import numpy as np
from scipy.linalg import block_diag
import os

from bernstein_funcs import * 
from helper_funcs import *
from structs import *

os.system('cls' if os.name=='nt' else 'clear')

gravity = 9.81

# params
horizon = 2
n_degree = 10
m = n_degree + 1
max_vel = 1.73
max_acc = 1.5*gravity
omega_g = 7000
omega_s = 100

rho = 0
max_iter = 1000
tol = 1e-2

obstacle = np.array([5,5,5], dtype=float)
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
W1 = bernstein_basis_matrix(t, n_degree-1) 
W1 = np.pad(W1, ((0, 0), (0, 1)), mode='constant') #[horizon x n_degree + 1]
W2 = bernstein_basis_matrix(t, n_degree-2) 
W2 = np.pad(W2, ((0, 0), (0, 2)), mode='constant') #[horizon x n_degree + 1]

# one stationary obstacle
A_tilde = np.vstack([W1, W2, W0]) #[horizon x (n_degree+1)]
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
Qe = omega_g*W0.T@W0 + omega_s*W2.T@W2
Q = block_diag(Qe,Qe,Qe)

qx = 2*omega_g*(drone_1.position_horizon[:,0].T@W0)
qy = 2*omega_g*(drone_1.position_horizon[:,1].T@W0)
qz = 2*omega_g*(drone_1.position_horizon[:,2].T@W0)
q = np.concatenate([qx,qy,qz]).reshape(-1,1)

C = np.vstack([
    np.kron(np.eye(3), W0[0]),
    np.kron(np.eye(3), W1[0]),
    np.kron(np.eye(3), W2[0]),
])
#current [pos vel acc] matrix basis
e = np.concatenate([
    curve[0],
    d_curve[0],
    dd_curve[0]
]).reshape(-1,1)# current [pos vel acc]

# -------------------------------------------------
# zeta_2 : angles
# -------------------------------------------------
rel = curve - obstacle
[r_xy, alpha_c, beta_c] = cart2pol_3D(rel)
[_, alpha_v, beta_v] = cart2pol_3D(d_curve)
[_, alpha_a, beta_a] = cart2pol_3D(dd_curve + np.array([0,0,gravity]))

# -------------------------------------------------
# zeta_3 : magnitudes
# -------------------------------------------------
# pos
omega_c, omega_v, omega_a, h, b = build_rhs(
        rel, d_curve, dd_curve,
        alpha_c, beta_c,
        alpha_v, beta_v,
        alpha_a, beta_a,
        max_vel, max_acc, horizon, gravity
    )

def build_G(W, omega):
    m = W.shape[1]
    H = W.shape[0]

    G = np.zeros((H, 3*m))
    for i in range(H):
        G[i,      :m] = omega[i, 0] * W[i]
        G[i,   m:2*m] = omega[i, 1] * W[i]
        G[i, 2*m:3*m] = omega[i, 2] * W[i]
    return G

G_v = build_G(W1, omega_v)
G_a = build_G(W2, omega_a)
G_c = build_G(W0, omega_c)

G = np.vstack((G_v, G_a, G_c))

# -------------------------------------------------
# dual / slack
# -------------------------------------------------
s = 0
lambda_i = 100

for k in range(15):
    print(f"iter {k}")
    print(f"pos {drone_1.position_horizon}")
    # Update q based on receding horizon
    drone_1.pos = curve[0]
    update_horizon(horizon, goal, drone_1)
    qx = -2*omega_g*(drone_1.position_horizon[:,0].T @ W0)
    qy = -2*omega_g*(drone_1.position_horizon[:,1].T @ W0)
    qz = -2*omega_g*(drone_1.position_horizon[:,2].T @ W0)
    q = np.concatenate([qx, qy, qz]).reshape(-1, 1)

    # ── S1: Primal update (solve QP) ──────────────────────────────────
    # Augmented Lagrangian:
    #   min  0.5 zᵀQz + qᵀz
    #        + λ_eqᵀ(Az-b)   + ρ/2 ||Az-b||²
    #        + λ_ineqᵀ(Gz-h+s) + ρ/2 ||Gz-h+s||²
    # Gradient = 0 → (Q + ρ AᵀA + ρ GᵀG) ζ = -q - Aᵀλ_eq - Gᵀλ_ineq + ρAᵀb + ρGᵀ(h-s)

    A_bar = Q + rho * (A.T @ A + G.T @ G)
    b_bar = q + rho * G.T @ (h - s) + rho * A.T @ b - lambda_i

    # KKT with equality constraints C @ zeta = e
    KKT = np.block([
        [A_bar, C.T],
        [C,     np.zeros((C.shape[0], C.shape[0]))]
    ])
    e = np.concatenate([curve[0], d_curve[0], dd_curve[0]]).reshape(-1, 1)
    rhs = np.vstack([b_bar, e])

    sol = np.linalg.lstsq(KKT, rhs)[0]
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
    rel = curve - obstacle
    alpha_c, beta_c = cart2angles(rel)
    alpha_v, beta_v = cart2angles(d_curve)
    alpha_a, beta_a = cart2angles(dd_curve + np.array([0, 0, gravity]))

    # ── S3: RHS rebuild ───────────────────────────────────────────────

    d_c, d_v, d_a, omega_c, omega_v, omega_a = update_magnitudes(curve, d_curve, dd_curve, alpha_c, beta_c,
                      alpha_v, beta_v, alpha_a, beta_a,
                      obstacle, gravity, max_vel, max_acc)
    b = build_b(d_c, d_v, d_a, omega_c, omega_v, omega_a, obstacle, gravity, horizon)

    b = b.reshape(-1, 1)
    h = h.reshape(-1, 1)

    G_v = build_G(W1, omega_v)
    G_a = build_G(W2, omega_a)
    G_c = build_G(W0, omega_c)
    G   = np.vstack((G_v, G_a, G_c))

    # ── S4: Slack update ──────────────────────────────────────────────
    # s >= 0, and G @ zeta + s = h  →  s = max(0, h - G @ zeta)
    s = np.maximum(0, -(G @ zeta_1 - h)) 

    # ── S5: Dual update ──────────
    residual_eq   = A @ zeta_1 - b
    residual_ineq = G @ zeta_1 - h + s
    lambda_i = lambda_i - (rho/2) * A.T @ residual_eq \
                        - (rho/2) * G.T @ residual_ineq

    # ── Convergence check ─────────────────────────────────────────────
    residual = np.linalg.norm(residual_eq) + np.linalg.norm(residual_ineq)

    #rho = min(1.3**k, 5e5)
    if residual < tol:
        print("Converged!")
        break