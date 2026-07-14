import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bernstein_funcs import *
from Structs import *
from helper_funcs import *
from obstacle_avoidance_funcs import init_obstacle_avoidance

# This program has objective have 1 drone restricted in vel and acc not collide
# with a fixed obstacle, the obstacle location is hard coded

os.system('cls' if os.name=='nt' else 'clear')

# ============================================================
# Problem
# ============================================================
param = Param()
param.sim.nD = 1

Nsim = param.sim.Nsim
dTi = param.sim.dTi

traj_rad = 2.2
start = np.array([traj_rad*np.cos(0), traj_rad*np.sin(0), 1.0])
goal  = np.array([traj_rad*np.cos(0), traj_rad*np.sin(0), 1.0])

drone_1 = DroneState(param)
drone_1.position = start
drone_1.velocity = np.array([0,0,0])
drone_1.acceleration = np.array([0,0,0])

verbose = False
save_mp4 = False

chaser_list = [drone_1]
vars = init_obstacle_avoidance(chaser_list, param)
Q = vars["Q"]
A_eq = vars["A_eq"]
A_obs = vars["A_obs"]
G = vars["G"]
h = vars["h"]
W0 = vars["W0"]
W1 = vars["W1"]
W2 = vars["W2"]
n_ctrl = vars["n_ctrl"]
dim = vars["dim"]

wN = W0[-1].reshape(-1,1)
x = np.zeros(dim)

drone_1.constraints.s = np.zeros(G.shape[0]).reshape(-1,1)
drone_1.constraints.lambda_obs = np.zeros([A_obs.shape[0],1])
drone_1.constraints.lambda_ineq = np.zeros([G.shape[0],1])

goal_history = []
executed_path = [start.copy()]
velocity_history = []
acc_history = []

for step in range(Nsim):
    t = step*param.sim.dTi
    print(f"Current iter: {step}")
    goal  = np.array([traj_rad*np.cos(t*0.1), traj_rad*np.sin(t*0.1), 1.0])
    goal_history.append(goal)
    # =====================================================
    # Equality constraint becomes current position
    # =====================================================
    b_eq = np.hstack([drone_1.position, drone_1.velocity, drone_1.acceleration]).reshape(-1,1)
    
    drone_1.constraints.lambda_obs[:] = 0
    drone_1.constraints.lambda_ineq[:] = 0

    # =====================================================
    # Solve AL subproblem
    # =====================================================
    for iteration in range(param.max_iter):
        # Step 1 - solve trajectory
        A_bar = (
            Q
            + param.rho_obs * A_obs.T @ A_obs
            + param.rho_ineq * G.T @ G
        )

        b_bar = (
            - drone_1.constraints.q
            - A_obs.T @ drone_1.constraints.lambda_obs
            + param.rho_obs * A_obs.T @ drone_1.constraints.b_obs
            - G.T @ drone_1.constraints.lambda_ineq
            + param.rho_ineq * G.T @ (h - drone_1.constraints.s)
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
        drone_1.constraints.b_obs = build_fixed_b_obs(
            param.horizon, 
            drone_1.constraints.obstacle_radius, 
            drone_1.constraints.obstacle_center, 
            traj
        )

        # Step 3 - Update slack term
        drone_1.constraints.s = np.maximum(0,h - G @ x - drone_1.constraints.lambda_ineq/param.rho_ineq)

        # calculate residuals
        residual_eq  = A_eq @ x - b_eq
        residual_ineq = G @ x - h + drone_1.constraints.s
        residual_obs = A_obs @ x - drone_1.constraints.b_obs

        # Step 4 - Update lambdas        
        drone_1.constraints.lambda_ineq += param.rho_ineq * residual_ineq
        drone_1.constraints.lambda_obs += param.rho_obs * residual_obs

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
    drone_1.position = traj[1]

    acc_history.append(drone_1.acceleration)
    velocity_history.append(drone_1.velocity)
    executed_path.append(drone_1.position)

    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        drone_1.constraints.q[idx] = -param.w_goal * goal[axis] * wN

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

x = drone_1.constraints.obstacle_radius[0,0]*(np.cos(u) * np.sin(v)) + drone_1.constraints.obstacle_center[0]
y = drone_1.constraints.obstacle_radius[1,1]*(np.sin(u) * np.sin(v)) + drone_1.constraints.obstacle_center[1]
z = drone_1.constraints.obstacle_radius[2,2]*(np.cos(v)) + drone_1.constraints.obstacle_center[2]

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

plt.axhline(param.v_max, color='k', linestyle='--', alpha=0.6, label='v_max')
plt.axhline(-param.v_max, color='k', linestyle='--', alpha=0.6)

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

plt.axhline(param.a_max, color='k', linestyle='--', alpha=0.6, label='a_max')
plt.axhline(-param.a_max, color='k', linestyle='--', alpha=0.6)

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
xs = drone_1.constraints.obstacle_radius[0,0]*(np.cos(u) * np.sin(v)) + drone_1.constraints.obstacle_center[0]
ys = drone_1.constraints.obstacle_radius[1,1]*(np.sin(u) * np.sin(v)) + drone_1.constraints.obstacle_center[1]
zs = drone_1.constraints.obstacle_radius[2,2]*(np.cos(v)) + drone_1.constraints.obstacle_center[2]

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