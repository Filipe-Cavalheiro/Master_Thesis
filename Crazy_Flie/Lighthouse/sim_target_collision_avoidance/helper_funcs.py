import numpy as np
from Structs import *
from math import comb

def initialize_qp(
    drone_states: DroneState,
    goal,
    param: Param
):
    """
    Initialize all matrices and variables that remain constant
    during the MPC execution.

    Returns a dictionary containing everything required by the
    optimization loop.
    """

    H = param.H
    n_degree = param.n_degree
    n_ctrl = n_degree + 1
    dim = 3 * n_ctrl

    # ==========================================================
    # Bernstein basis matrices
    # ==========================================================

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

    # ==========================================================
    # Cost function
    # ==========================================================

    Q1 = param.w_smooth * W2.T @ W2

    Q = np.zeros((dim, dim))

    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        Q[np.ix_(idx, idx)] = Q1

    wN = W0[-1].reshape(-1,1)
    Qg = param.w_goal * np.outer(wN, wN)
    q = np.zeros([dim,1])
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        Q[np.ix_(idx, idx)] += Qg
        q[idx] = -param.w_goal * goal[axis] * wN

    # ==========================================================
    # Equality constraints
    # ==========================================================

    A_eq = np.zeros((9, dim))
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        A_eq[axis, idx] = W0[0]
        A_eq[axis + 3, idx] = W1[0]
        A_eq[axis + 6, idx] = W2[0]

    A_obs = np.zeros((3 * H, dim))
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        A_obs[axis * H:(axis + 1) * H, idx] = W0

    traj = np.tile(drone_states.position, (H,1))
    b_obs = build_b_obs(H, param.obstacle_radius, param.obstacle_center, traj)


    # ==========================================================
    # Inequality constraints
    # ==========================================================

    G_vel = np.zeros((3 * H, dim))
    G_acc = np.zeros((3 * H, dim))

    for axis in range(3):

        idx = np.arange(axis, dim, 3)
        rows = np.arange(axis * H, (axis + 1) * H)

        G_vel[np.ix_(rows, idx)] = W1
        G_acc[np.ix_(rows, idx)] = W2

    G = np.vstack((
        G_vel,
        -G_vel,
        G_acc,
        -G_acc
    ))

    h = np.vstack((
        np.ones((6 * H, 1)) * param.v_max,
        np.ones((6 * H, 1)) * param.a_max
    ))

    # ==========================================================
    # Augmented Lagrangian variables
    # ==========================================================

    lambda_obs = np.zeros((A_obs.shape[0], 1))
    lambda_ineq = np.zeros((G.shape[0], 1))
    s = np.zeros((G.shape[0], 1))
    x = np.zeros((dim, 1))

    return {
        "W0": W0,
        "W1": W1,
        "W2": W2,
        "Q": Q,
        "q": q,
        "A_eq": A_eq,
        "A_obs": A_obs,
        "G": G,
        "h": h,
        "b_obs": b_obs,
        "lambda_obs": lambda_obs,
        "lambda_ineq": lambda_ineq,
        "s": s,
        "x": x,
        "dim": dim,
        "n_ctrl": n_ctrl,
    }


def bernstein_basis_matrix(t, n_degree):
    t = np.asarray(t)

    M = np.zeros((len(t), n_degree + 1))

    for k in range(n_degree + 1):
        M[:, k] = comb(n_degree, k) * t**k * (1 - t)**(n_degree - k)

    return M

def cart2pol_3D(coords):
    """
    Cartesian -> spherical

    Parameters
    ----------
    coords : (..., 3) array
        [..., x, y, z]

    Returns
    -------
    (..., 3) array
        [..., r, alpha, theta]
    """
    coords = np.asarray(coords)

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    alpha = np.arctan2(y, x)
    beta = np.arccos(z/r)

    return np.stack((r, alpha, beta), axis=-1)

def omega(a, b):
    return np.column_stack([
        np.cos(a) * np.sin(b),
        np.sin(a) * np.sin(b),
        np.cos(b)
    ])

def build_b_obs(H, obstacle_radius, obstacle_center, traj):
    b_obs = np.zeros((3 * H, 1))
    for k in range(H):
        delta = np.linalg.inv(obstacle_radius) @ (traj[k] - obstacle_center).T
        d, alpha, beta = cart2pol_3D(delta)
        d = max(d, 1.0)
        b = obstacle_center + (d * omega(alpha, beta))@obstacle_radius
        rows = np.arange(k, 3 * H, H)
        b_obs[rows,0] = b

    return b_obs