import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bernstein_funcs import *
from structs import *
from helper_funcs import *

os.system('cls' if os.name=='nt' else 'clear')

# ============================================================
# Problem
# ============================================================
t_max = 120
dTi = 0.1
Nsim = int(t_max/dTi)

traj_rad = 2.2
start = np.array([traj_rad*np.cos(0), traj_rad*np.sin(0), 1.0])
goal  = np.array([traj_rad*np.cos(0), traj_rad*np.sin(0), 1.0])

H = 40                          # horizon
rho_obs = 100                   # Obstacle avoidance penalty
rho_ineq = 20.0                 # max/min vel and acc penalty
max_iter = 100                  # max iterations per update

w_goal = 1000.0                 # gain to reduce dist between agent and goal
w_smooth = 0.1                  # gain to reduce high accelerations

v_max = 2                       # max velocity constraint 
a_max = 1                       # max acceleration constraint 

obstacle_center = np.array([0, 2.0, 1.0])
obstacle_radius = np.diag([0.5, 0.5, 0.5])

drone_1 = DroneState(H)
drone_1.pos = start
drone_1.velocity = np.array([0,0,0])
drone_1.acceleration = np.array([0,0,0])

verbose = False
save_mp4 = False

# ===========================================================
# Variable layout
#
# x =
#   [
#       c0x,c0y,c0z,
#       c1x,c1y,c1z,
#       ...
#       cnx,cny,cnz
#   ]
# the path is defined via bernstein polynomials instead of [x,y,z]
# being the number of elements of the matrix R^[n_ctrl x 3]
# ============================================================
n_degree = 10               # bernstein polynomial degree
n_ctrl = n_degree + 1
dim = 3 * n_ctrl

# bernstein basis matrix construction
# obtained from the Paper:
# On the Derivatives of Bernstein Polynomials: 
# An Application for the Solution of High Even-Order Differential Equations
t = np.linspace(0, 1, H)
W0 = bernstein_basis_matrix(t, n_degree)

B = bernstein_basis_matrix(t, n_degree-1)
D1 = np.zeros((n_degree, n_degree+1))
for i in range(n_degree-1):
    D1[i,i]   = -1
    D1[i,i+1] = 1
W1 = n_degree * (B @ D1)

B = bernstein_basis_matrix(t, n_degree-2)
D2 = np.zeros((n_degree-1, n_degree+1))
for i in range(n_degree-1):
    D2[i,i]   = 1
    D2[i,i+1] = -2
    D2[i,i+2] = 1
W2 = n_degree * (n_degree-1) * (B @ D2)

# ============================================================
# Smoothness Hessian
#
# sum ||\ddot p(k)||²
# the smoothness hessian is defined as a cost function that reduces high value 
# accelerations
# ============================================================
Q1 = w_smooth * W2.T @ W2

Q = np.zeros((dim, dim))
for axis in range(3):
    idx = np.arange(axis, dim, 3)
    Q[np.ix_(idx, idx)] = Q1

# ============================================================
# Goal cost
#
# w_goal ||p_N - goal||²
# The goal cost is the cost associated with the distance between the agent 
# and the goal position
# ============================================================
wN = W0[-1].reshape(-1,1)

Qg = w_goal * np.outer(wN, wN)
q = np.zeros([dim,1])
for axis in range(3):
    idx = np.arange(axis, dim, 3)
    Q[np.ix_(idx, idx)] += Qg
    q[idx] = -w_goal * goal[axis] * wN

# ============================================================
# Equality constraints
# ============================================================

# [p(0),v(0),acc(0)] = start
# This equality constraint defines that the system must have at horizon 1
# the current pos, vel and acc in account
# A_eq =
# [ W0(0)  0      0    ]
# [ 0      W0(0)  0    ]  position
# [ 0      0      W0(0)]
# ----------------------
# [ W1(0)  0      0    ]
# [ 0      W1(0)  0    ]  velocity
# [ 0      0      W1(0)]
# ----------------------
# [ W2(0)  0      0    ]
# [ 0      W2(0)  0    ]  acceleration
# [ 0      0      W2(0)]
A_eq = np.zeros((9, dim))
for axis in range(3):
    cols = np.arange(axis, dim, 3)
    A_eq[axis, cols]     = W0[0]   # position
    A_eq[axis+3, cols]   = W1[0]   # velocity
    A_eq[axis+6, cols]   = W2[0]   # acceleration

# This equality constraint defines that the system must along the entire horizon
# respect the enforcement on collision avoidance.
# A_obs =
# [ W0  0   0 ] x
# [ 0   W0  0 ] y  position
# [ 0   0   W0] z
A_obs = np.zeros((3*H, dim))
for axis in range(3):
    cols = np.arange(axis, dim, 3)
    A_obs[H*axis:H*(axis+1), cols] = W0   # position

# Obstacle avoidance constraints
traj = np.tile(drone_1.pos, (H,1))
b_obs = build_b_obs(H, obstacle_radius, obstacle_center, traj)

# ============================================================
# Inequality constraints
#
# G x <= h
# The inequality is given as a restriction in velocity
# the inequality can be introduced directly the slack term 's' in
# Gx-h+s = 0. The constrain is given as a cube around the agent 
# ============================================================

# G_tilde =
# [ W1  0   0 ]  x
# [ 0   W1  0 ]  y
# [ 0   0   W1]  z

# Velocity constraints
G_vel = np.zeros((3 * H, dim))
for axis in range(3):
    idx = np.arange(axis, dim, 3)
    rows = np.arange(axis * H, (axis + 1) * H)
    G_vel[np.ix_(rows, idx)] = W1

# Acceleration constraints
G_acc = np.zeros((3 * H, dim))
for axis in range(3):
    idx = np.arange(axis, dim, 3)
    rows = np.arange(axis * H, (axis + 1) * H)
    G_acc[np.ix_(rows, idx)] = W2

# Assemble G
G = np.vstack((
    G_vel,
    -G_vel,
    G_acc,
    -G_acc,
))

# Assemble h
h = np.vstack((
    np.ones((6 * H, 1)) * v_max,
    np.ones((6 * H, 1)) * a_max,
))

s = np.zeros(G.shape[0]).reshape(-1,1)

# ============================================================
# Augmented Lagrangian
# ============================================================
lambda_obs = np.zeros([A_obs.shape[0],1])
lambda_ineq = np.zeros([G.shape[0],1])
x = np.zeros(dim)

goal_history = []
executed_path = [start.copy()]
velocity_history = []
acc_history = []

for step in range(Nsim):
    print(f"Current iter: {step}")
    goal  = np.array([traj_rad*np.cos(0.1*step*dTi), traj_rad*np.sin(0.1*step*dTi), 1.0])
    goal_history.append(goal)
    # =====================================================
    # Equality constraint becomes current position
    # =====================================================
    b_eq = np.hstack([drone_1.pos, drone_1.velocity, drone_1.acceleration]).reshape(-1,1)
    
    lambda_obs[:] = 0
    lambda_ineq[:] = 0

    # =====================================================
    # Solve AL subproblem
    # =====================================================
    for iteration in range(max_iter):

        # Step 1 - solve trajectory
        A_bar = (
            Q
            + rho_obs * A_obs.T @ A_obs
            + rho_ineq * G.T @ G
        )

        b_bar = (
            - q
            - A_obs.T @ lambda_obs
            + rho_obs * A_obs.T @ b_obs
            - G.T @ lambda_ineq
            + rho_ineq * G.T @ (h - s)
        )

        KKT = np.block([
            [A_bar, A_eq.T],
            [A_eq, np.zeros((A_eq.shape[0], A_eq.shape[0]))]
        ])

        rhs = np.concatenate([
            b_bar,
            b_eq
        ])

        solution = np.linalg.solve(KKT, rhs)

        n = A_bar.shape[0]
        x = solution[:n]

        # Step 2 - Update alpha, beta and d
        C = x.flatten().reshape(n_ctrl,3)
        traj = W0 @ C
        b_obs = build_b_obs(H, obstacle_radius, obstacle_center, traj)

        # Step 3 - Update slack term
        s = np.maximum(0,h - G @ x - lambda_ineq/rho_ineq)

        # calculate residuals
        residual_eq  = A_eq @ x - b_eq
        residual_ineq = G @ x - h + s
        residual_obs = A_obs @ x - b_obs

        # Step 4 - Update lambdas        
        lambda_ineq += rho_ineq * residual_ineq
        lambda_obs += rho_obs * residual_obs

        if (
            np.linalg.norm(residual_eq)
            + np.linalg.norm(residual_ineq)
            + np.linalg.norm(residual_obs)
            < 1e-6
        ):
            break

    C = x.flatten().reshape(n_ctrl,3)
    traj = W0 @ C
    vel = W1 @ C
    acc = W2 @ C
 
    if verbose:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(
            traj[:,0],
            traj[:,1],
            traj[:,2],
            linewidth=3,
            label='executed path'
        )

        ax.scatter(
            traj[:,0], traj[:,1], traj[:,2],
            s=100
        )

        ax.quiver(traj[:,0], traj[:,1], traj[:,2], vel[:,0], vel[:,1], vel[:,2], length=0.1, normalize=True, color='red') 

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.legend()
        plt.show()

    # executing next step velocity
    drone_1.acceleration = acc[1]
    drone_1.velocity = vel[1]
    drone_1.pos = traj[1]

    acc_history.append(drone_1.acceleration)
    velocity_history.append(drone_1.velocity)
    executed_path.append(drone_1.pos)

    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        q[idx] = -w_goal * goal[axis] * wN

goal_history = np.array(goal_history)
executed_path = np.array(executed_path)
velocity_history = np.array(velocity_history)
acc_history = np.array(acc_history)

# ============================================================
# Extract trajectory
# ============================================================

print("\nFinal point:")
print(executed_path[-1])

print("\nGoal:")
print(goal)

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    executed_path[:,0],
    executed_path[:,1],
    executed_path[:,2],
    linewidth=3,
    label='executed path'
)

ax.scatter(
    start[0], start[1], start[2],
    s=100,
    label='start'
)

ax.scatter(
    goal[0], goal[1], goal[2],
    s=100,
    label='goal'
)

u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:15j]

x = obstacle_radius[0,0]*(np.cos(u) * np.sin(v)) + obstacle_center[0]
y = obstacle_radius[1,1]*(np.sin(u) * np.sin(v)) + obstacle_center[1]
z = obstacle_radius[2,2]*(np.cos(v)) + obstacle_center[2]

ax.plot_surface(
    x,
    y,
    z,
    color='red',
    alpha=0.3
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.axis('equal')
ax.legend()
plt.show(block=False)

# ============================================================
# Velocity plot
# ============================================================

time = np.linspace(0, Nsim, len(velocity_history[:,0]))

plt.figure(figsize=(8,5))
plt.plot(time, velocity_history[:,0], 'r-o', label='vx')
plt.plot(time, velocity_history[:,1], 'g-o', label='vy')
plt.plot(time, velocity_history[:,2], 'b-o', label='vz')

plt.axhline(v_max, color='k', linestyle='--', alpha=0.6, label='v_max')
plt.axhline(-v_max, color='k', linestyle='--', alpha=0.6)

plt.xlabel("Normalized time")
plt.ylabel("Velocity")
plt.title("Velocity Profile")
plt.grid(True)
plt.legend()
plt.show(block=False)

# ============================================================
# Acceleration plot
# ============================================================

time = np.linspace(0, Nsim, len(velocity_history[:,0]))

plt.figure(figsize=(8,5))
plt.plot(time, acc_history[:,0], 'r-o', label='ax')
plt.plot(time, acc_history[:,1], 'g-o', label='ay')
plt.plot(time, acc_history[:,2], 'b-o', label='az')

plt.axhline(a_max, color='k', linestyle='--', alpha=0.6, label='a_max')
plt.axhline(-a_max, color='k', linestyle='--', alpha=0.6)

plt.xlabel("Normalized time")
plt.ylabel("Acceleration")
plt.title("Acceleration Profile")
plt.grid(True)
plt.legend()
plt.show(block=False)

# ============================================================
# Animation plot
# ============================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot obstacle
u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:15j]
xs = obstacle_radius[0,0]*(np.cos(u) * np.sin(v)) + obstacle_center[0]
ys = obstacle_radius[1,1]*(np.sin(u) * np.sin(v)) + obstacle_center[1]
zs = obstacle_radius[2,2]*(np.cos(v)) + obstacle_center[2]

ax.plot_surface(xs, ys, zs, color='red', alpha=0.3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Animated objects
path_line, = ax.plot([], [], [], 'b-', lw=3, label="Executed Path")
drone, = ax.plot([], [], [], 'bo', markersize=8)
start_pt = ax.scatter(*start, color='green', s=80, label='Start')
drone_goal, = ax.plot([], [], [], 'o', color='orange', markersize=8, label='Goal')

ax.axis('equal')
ax.legend()

def init():
    path_line.set_data([], [])
    path_line.set_3d_properties([])
    drone.set_data([], [])
    drone.set_3d_properties([])
    drone_goal.set_data([], [])
    drone_goal.set_3d_properties([])
    return path_line, drone

def update(frame):
    path_line.set_data(
        executed_path[:frame+1,0],
        executed_path[:frame+1,1]
    )
    path_line.set_3d_properties(
        executed_path[:frame+1,2]
    )

    drone.set_data(
        [executed_path[frame,0]],
        [executed_path[frame,1]]
    )
    drone.set_3d_properties(
        [executed_path[frame,2]]
    )

    drone_goal.set_data(
        [goal_history[frame,0]],
        [goal_history[frame,1]]
    )
    drone_goal.set_3d_properties(
        [goal_history[frame,2]]
    )

    # ---- Dynamically adjust limits ----
    pts = np.vstack((executed_path[:frame+1], goal_history[:frame+1]))
    margin = 1.0

    ax.set_xlim(pts[:,0].min()-margin, pts[:,0].max()+margin)
    ax.set_ylim(pts[:,1].min()-margin, pts[:,1].max()+margin)
    ax.set_zlim(pts[:,2].min()-margin, pts[:,2].max()+margin)

    return path_line, drone, drone_goal

ani = FuncAnimation(
    fig,
    update,
    frames=len(goal_history),
    init_func=init,
    interval=50,      # milliseconds between frames
    blit=False
)

if save_mp4: ani.save("trajectory.mp4", writer="ffmpeg", fps=24)
plt.show()