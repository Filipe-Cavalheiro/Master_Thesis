import numpy as np
from scipy.linalg import block_diag
import os

from bernstein_funcs import * 
from helper_funcs import *
from structs import *

os.system('cls' if os.name=='nt' else 'clear')

gravity = 9.81

# params
horizon = 5
n_degree = 10
m = n_degree + 1
max_vel = 1.73
max_acc = 1.5*gravity
omega_g = 2
omega_s = 1
rho = 1

max_iter = 1000
tol = 1e-2

goal = np.array([10,10,10], dtype=float)

drone_1 = DroneState(horizon)
drone_1.pos = np.array([0,0,0])
drone_1.position_horizon = np.zeros([horizon,3])
update_horizon(horizon, goal, drone_1)

coeffs, d_coeffs, dd_coeffs = fit_bernstein_curve(
    drone_1.position_horizon, n_degree
)
# coeffs [3(xyz) x n_degree + 1]
d_coeffs =  np.pad(d_coeffs, ((0, 0), (0, 1)), mode='constant')  # d_coeffs [3(xyz) x n_degree + 1]
dd_coeffs =  np.pad(dd_coeffs, ((0, 0), (0, 2)), mode='constant') # dd_coeffs [3(xyz) x n_degree + 1]

t = np.linspace(0,1,horizon)
W0 = bernstein_basis_matrix(t, n_degree) #[horizon x n_degree + 1]
B = bernstein_basis_matrix(t, n_degree-1)
D1 = np.zeros((n_degree, n_degree+1))
for i in range(n_degree-1):
    D1[i,i]   = 1
    D1[i,i+1] = -1
W1 = B @ D1

B = bernstein_basis_matrix(t, n_degree-2)
D2 = np.zeros((n_degree-1, n_degree+1))
for i in range(n_degree-1):
    D2[i,i]   = 1
    D2[i,i+1] = -2
    D2[i,i+2] = 1
W2 = B @ D2

# one stationary obstacle
A_tilde = np.vstack([W0, W1, W2]) #[horizon x (n_degree+1)]
A = block_diag(A_tilde,
               A_tilde,
               A_tilde) #9*horizon × 3*(n_degree+1)

zeta_1 = np.concatenate(coeffs).reshape(-1,1)

curve = W0 @ coeffs.T
d_curve = W1 @ d_coeffs.T
dd_curve = W2 @ dd_coeffs.T

# Q and q are obtained from the transform of 1a into bernstein format W@coef
# we are using the goal transformation: omega_g*curve.T*curve
# with a negative weight for high values of acceleration: omega_s*dd_curve.T*dd_curve
Qe = omega_g*W0.T@W0 # + omega_s*W2.T@W2
Q = block_diag(Qe,Qe,Qe)

qx = 2*omega_g*(W0@drone_1.position_horizon[0:11])
qy = 2*omega_g*(W0@drone_1.position_horizon[11:22])
qz = 2*omega_g*(W0@drone_1.position_horizon[22:33])
q = np.concatenate([qx, qy, qz]).reshape(-1, 1)

C = np.vstack([
    np.kron(np.eye(3), W0[0]),
    np.kron(np.eye(3), W1[0]),
    np.kron(np.eye(3), W2[0]),
])
#current [pos vel acc] matrix basis
e = np.concatenate([
    curve[1],
    d_curve[1],
    dd_curve[1]
]).reshape(-1,1)# current [pos vel acc]

rel = curve - goal
alpha_c, beta_c = cart2angles(rel)
alpha_v, beta_v = cart2angles(d_curve)
alpha_a, beta_a = cart2angles(dd_curve + np.array([0, 0, gravity]))

d_c, d_v, d_a, omega_c, omega_v, omega_a = update_magnitudes(curve, d_curve, dd_curve, alpha_c, beta_c,
                    alpha_v, beta_v, alpha_a, beta_a,
                    goal, gravity, max_vel, max_acc)
b = build_b(d_c, d_v, d_a, omega_c, omega_v, omega_a, goal, gravity, horizon)

# -------------------------------------------------
# dual / slack
# -------------------------------------------------
lambda_i = np.zeros((A.shape[0],1))

for k in range(20):
    print(f"iter {k}")
    print(f"pos {drone_1.pos}")
    # Update q based on receding horizon
    drone_1.pos = curve[1]
    drone_1.position_horizon = zeta_1

    # ── S1: Primal update (solve QP) ──────────────────────────────────
    # Augmented Lagrangian:
    #   min  0.5 zᵀQz + qᵀz
    #        + λ_eqᵀ(Az-b)   + ρ/2 ||Az-b||²
    # Gradient = 0 → (Q + ρ AᵀA) ζ = -q - Aᵀλ_eq + ρAᵀb

    A_bar = Q + rho * A.T @ A

    b_bar = - q + rho * A.T @ b - A.T @ lambda_i

    # KKT with equality constraints C @ zeta = e
    KKT = np.block([
        [A_bar, C.T],
        [C,     np.zeros((C.shape[0], C.shape[0]))]
    ])
    e = np.concatenate([curve[1], d_curve[1], dd_curve[1]]).reshape(-1, 1)
    rhs = np.vstack([b_bar, e])

    sol = np.linalg.lstsq(A_bar, b_bar, rcond=None)[0]
    zeta_1 = sol[:A_bar.shape[0]]

    # Reconstruct curves
    coeffs   = zeta_1.reshape(3, m)
    d_coeffs = n_degree * np.diff(coeffs, axis=1)
    d_coeffs = np.pad(d_coeffs, ((0,0),(0,1)), mode='constant')
    dd_coeffs = n_degree * (n_degree-1) * np.diff(coeffs, n=2, axis=1)
    dd_coeffs = np.pad(dd_coeffs, ((0,0),(0,2)), mode='constant')

    curve    = W0 @ coeffs.T
    d_curve  = W1 @ d_coeffs.T
    dd_curve = W2 @ dd_coeffs.T

    # ── S2: Direction update ───────────────────────────────────────────
    rel = curve - goal
    alpha_c, beta_c = cart2angles(rel)
    alpha_v, beta_v = cart2angles(d_curve)
    alpha_a, beta_a = cart2angles(dd_curve + np.array([0, 0, gravity]))

    # ── S3: RHS rebuild ───────────────────────────────────────────────

    # d_c, d_v, d_a, omega_c, omega_v, omega_a = update_magnitudes(curve, d_curve, dd_curve, alpha_c, beta_c,
    #                   alpha_v, beta_v, alpha_a, beta_a,
    #                   goal, gravity, max_vel, max_acc)
    # b = build_b(d_c, d_v, d_a, omega_c, omega_v, omega_a, goal, gravity, horizon)

    # b = b.reshape(-1, 1)

    # ── Convergence check ─────────────────────────────────────────────
    r = A @ zeta_1 - b           # shape (9*horizon,1)
    lambda_i = lambda_i + rho * r

    # update_horizon(horizon, goal, drone_1)
    qx = 2*omega_g*(W0@drone_1.position_horizon[0:11])
    qy = 2*omega_g*(W0@drone_1.position_horizon[11:22])
    qz = 2*omega_g*(W0@drone_1.position_horizon[22:33])
    q = np.concatenate([qx, qy, qz]).reshape(-1, 1)

    residual = np.linalg.norm(r)
    # update_horizon(horizon, goal, drone_1)

    if residual < tol:
        print("Converged!")
        break

print(drone_1.position_horizon)