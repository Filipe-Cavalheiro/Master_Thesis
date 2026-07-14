import numpy as np
from Structs import *

def cart2pol_3D(coords):
    """
    Cartesian -> spherical

    Parameters
    ----------
    coords : (3, ...) array
        [..., x, y, z]

    Returns
    -------
    (3, ...) array
        [..., r, alpha, theta]
    """
    coords = np.asarray(coords)

    x = coords[0, ...]
    y = coords[1, ...]
    z = coords[2, ...]

    r = np.sqrt(x**2 + y**2 + z**2)

    alpha = np.arctan2(y, x)
    beta = np.arccos(z/r)

    return r, alpha, beta


def pol2cart_3D(coords):
    """
    Spherical -> Cartesian

    Parameters
    ----------
    coords : (..., 3) array
        [..., rho, phi, theta]

    Returns
    -------
    (..., 3) array
        [..., x, y, z]
    """
    coords = np.asarray(coords)

    rho = coords[..., 0]
    phi = coords[..., 1]
    theta = coords[..., 2]

    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)

    return np.stack((x, y, z), axis=-1)

def omega(a, b):
    return np.column_stack([
        np.cos(a) * np.sin(b),
        np.sin(a) * np.sin(b),
        np.cos(b)
    ]).reshape(3,1)

def update_horizon(horizon, goal: np.ndarray, drone: DroneState):
    delta = goal - drone.pos
    dist = np.linalg.norm(delta)

    if dist < 1e-6:
        drone.position_horizon = np.repeat(drone.pos[None, :], horizon, axis=0)
        return

    direction = delta / dist
    lookahead = min(dist, 1.0)
    end = drone.pos + direction * lookahead

    drone.position_horizon = np.linspace(drone.pos, end, horizon)
def normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm > 1e-8, v / norm, 0.0)

def cart2angles(v):
    v = normalize(v)
    beta = np.arccos(np.clip(v[:, 2], -1, 1))
    alpha = np.arctan2(v[:, 1], v[:, 0])
    return alpha, beta

# S3: closed-form magnitude update (scalar QP per timestep, then clip)
def update_magnitudes(curve, d_curve, dd_curve, alpha_c, beta_c,
                      alpha_v, beta_v, alpha_a, beta_a,
                      obstacle, gravity, max_vel, max_acc):
    omega_c = np.column_stack([np.cos(alpha_c)*np.sin(beta_c),
                                np.sin(alpha_c)*np.sin(beta_c),
                                np.cos(beta_c)])
    omega_v = np.column_stack([np.cos(alpha_v)*np.sin(beta_v),
                                np.sin(alpha_v)*np.sin(beta_v),
                                np.cos(beta_v)])
    omega_a = np.column_stack([np.cos(alpha_a)*np.sin(beta_a),
                                np.sin(alpha_a)*np.sin(beta_a),
                                np.cos(beta_a)])

    rel = curve - obstacle   # (horizon, 3)
    acc_g = dd_curve + np.array([0, 0, gravity])

    # Each d is: argmin_d ||r - d*omega||^2, clipped to feasible set
    # Solution: d* = dot(r, omega), then clip
    d_c = np.sum(rel   * omega_c, axis=1, keepdims=True)  # (horizon,1)
    d_v = np.sum(d_curve * omega_v, axis=1, keepdims=True)
    d_a = np.sum(acc_g   * omega_a, axis=1, keepdims=True)

    # Clip to feasible sets C_zeta3
    d_c = np.maximum(1.0, d_c)          # collision: d >= 1 (unit ellipsoid)
    d_v = np.clip(d_v, 0, max_vel)      # velocity:  0 <= d <= v_max
    d_a = np.clip(d_a, 0, max_acc)      # accel:     0 <= d <= f_max

    return d_c, d_v, d_a, omega_c, omega_v, omega_a

def build_fixed_b_obs(H, obstacle_radius, obstacle_center, traj):
    b_obs = np.zeros((3 * H, 1))
    for k in range(H):
        delta = np.linalg.inv(obstacle_radius) @ (traj[k] - obstacle_center).reshape(-1,1)
        d, alpha, beta = cart2pol_3D(delta)
        d = max(d, 1.0)
        b = obstacle_center + (d * omega(alpha, beta))@obstacle_radius
        rows = np.arange(k, 3 * H, H)
        b_obs[rows,0] = b

    return b_obs

def build_horizon_b_obs(H, obstacle_radius, obs_horizon, traj):
    b_obs = np.zeros((3 * H, 1))
    for k in range(H):
        delta = np.linalg.inv(obstacle_radius) @ (traj[k, :] - obs_horizon[k,:]).reshape(-1,1)
        d, alpha, beta = cart2pol_3D(delta)
        d = max(d, 1.0)
        b = obs_horizon[k,:] + (d * omega(alpha, beta))@obstacle_radius
        rows = np.arange(k, 3 * H, H)
        b_obs[rows,0] = b

    return b_obs

def build_constraints(H, dim, W0, W1, W2,
                      traj, obstacle_center,
                      R, v_max, a_max):
    """
    Build inequality constraints Gx <= h.

    Parameters
    ----------
    H : int
        Prediction horizon.
    dim : int
        Optimization variable dimension (typically 3*H).
    W0, W1, W2 : ndarray
        Position, velocity, and acceleration mapping matrices.
    traj : ndarray, shape (H,3)
        trajectory drone position.
    obstacle_center : ndarray, shape (3,)
        Obstacle center.
    R : float
        Safety radius.
    v_max : float
        Maximum velocity.
    a_max : float
        Maximum acceleration.

    Returns
    -------
    G : ndarray
        Inequality constraint matrix.
    h : ndarray
        Inequality constraint vector.
    """

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

    # Position mapping (used for obstacle constraints)
    G_pos = np.zeros((3 * H, dim))
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        rows = np.arange(axis * H, (axis + 1) * H)
        G_pos[np.ix_(rows, idx)] = W0

    # Obstacle avoidance constraints
    G_obs = []
    h_obs = []

    for k in range(H):
        pk = traj[k]
        delta = np.abs(pk - obstacle_center)
        dist = np.linalg.norm(delta)
        normal = delta / dist

        row = np.zeros(dim)
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            row[idx] = normal[axis] * W0[k]

        G_obs.append(-row)
        h_obs.append(-(R - dist))

    G_obs = np.array(G_obs)
    h_obs = np.array(h_obs).reshape(-1, 1)

    # Assemble G
    G = np.vstack((
        G_vel,
        -G_vel,
        G_acc,
        -G_acc,
        G_obs
    ))

    # Assemble h
    h = np.vstack((
        np.ones((6 * H, 1)) * v_max,
        np.ones((6 * H, 1)) * a_max,
        h_obs
    ))

    return G, h