
import numpy as np
from scipy.linalg import expm

import numpy as np

def skew(x):
    """
    Obtain skew-symmetric matrix from vector.

    Supports:
      - n = 3: so(3)
      - n = 1: so(2)

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    X : ndarray
        Skew-symmetric matrix.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if n == 3:
        X = np.array([
            [0,    -x[2],  x[1]],
            [x[2],  0,   -x[0]],
            [-x[1], x[0], 0]
        ])

    elif n == 1:
        X = np.array([
            [0,    -x[0]],
            [x[0],  0]
        ])

    else:
        raise ValueError(
            "SKEW function not implemented for input dimensions other than "
            "1 or 3 (i.e., so(2) and so(3))."
        )

    return X

def unskew(M):
    """
    Extracts the vector from a 3x3 skew-symmetric matrix.
    """
    return np.array([
        [M[2, 1]],
        [M[0, 2]],
        [M[1, 0]]
    ])

def rot_integrate(R, om, dT):
    """
    Integrate rotation matrix using angular velocity.
    
    Parameters:
        R : ndarray, shape (3,3)
            Initial rotation matrix
        om : array-like, shape (3,)
            Angular velocity vector
        dT : float
            Time step
            
    Returns:
        Rp : ndarray, shape (3,3)
            Updated rotation matrix
    """
    nsteps = 10
    Rp = R.copy()

    for _ in range(nsteps):
        Rp = Rp @ expm((dT / nsteps) * skew(om))

    return Rp

import numpy as np

def Euler2R(ang):
    """
    Converts an Euler angles vector [phi, theta, psi] into a rotation matrix
    of the Z-Y-X representation:
        R = rot_z(psi) * rot_y(theta) * rot_x(phi)

    Parameters:
        ang : array-like
            Euler angles [phi, theta, psi] in radians

    Returns:
        rot : numpy.ndarray
            3x3 rotation matrix
    """

    phi, theta, psi = ang

    rotx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])

    roty = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0,              1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])

    rot = rotz @ roty @ rotx

    return rot

def omega(a, b):
    sa = np.sin(a)
    ca = np.cos(a)
    sb = np.sin(b)
    cb = np.cos(b)

    return np.array([
        ca * sb,
        sa * sb,
        cb
    ])
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

def spherical_gaussian(reference, vectors, sigma=1.5):
    """
    Assigns Gaussian values based on angular distance
    from the reference vector.
    """
    # dot product gives cos(theta) for unit vectors
    cos_theta = vectors @ reference

    # avoid numerical problems
    cos_theta = np.clip(cos_theta, -1, 1)

    # angular distance
    theta = np.arccos(cos_theta)

    # Gaussian weight
    values = 1/np.exp(-(theta**2) / (2*sigma**2))

    return values

def spherical_gaussian(reference, vectors, sigma=1.5):
    """
    Assigns Gaussian values based on angular distance
    from the reference vector.
    """
    # dot product gives cos(theta) for unit vectors
    cos_theta = vectors @ reference

    # avoid numerical problems
    cos_theta = np.clip(cos_theta, -1, 1)

    # angular distance
    theta = np.arccos(cos_theta)

    # Gaussian weight
    values = np.exp((theta**2) / (2*sigma**2))

    return values

def _exp_so3(phi):
    """Rodrigues formula: rotation vector -> rotation matrix."""
    phi = np.asarray(phi).ravel()
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-9:
        return np.eye(3) + K + 0.5 * K @ K
    return (np.eye(3)
            + np.sin(theta) / theta * K
            + (1 - np.cos(theta)) / theta**2 * K @ K)

def rk4_step_rigid_body(deriv, p, v, R, om, dt):
    """
    One RK4 step for a rigid body.

    deriv(p, v, R, om) -> (dot_p, dot_v, dot_om)
    R is a 3x3 rotation matrix; om is the body-frame angular velocity (R_dot = R @ hat(om)).
    Position, velocity and angular velocity use standard RK4. The attitude is
    propagated on SO(3) using intermediate rotations and an RK4-weighted
    angular velocity, so R stays orthonormal.
    """
    # Stage 1
    k1p, k1v, k1o = deriv(p, v, R, om)
    om1 = om

    # Stage 2
    om2 = om + 0.5 * dt * k1o
    R2 = R @ _exp_so3(0.5 * dt * om1)
    k2p, k2v, k2o = deriv(p + 0.5 * dt * k1p, v + 0.5 * dt * k1v, R2, om2)

    # Stage 3
    om3 = om + 0.5 * dt * k2o
    R3 = R @ _exp_so3(0.5 * dt * om2)
    k3p, k3v, k3o = deriv(p + 0.5 * dt * k2p, v + 0.5 * dt * k2v, R3, om3)

    # Stage 4
    om4 = om + dt * k3o
    R4 = R @ _exp_so3(dt * om3)
    k4p, k4v, k4o = deriv(p + dt * k3p, v + dt * k3v, R4, om4)

    p_new  = p  + dt / 6 * (k1p + 2*k2p + 2*k3p + k4p)
    v_new  = v  + dt / 6 * (k1v + 2*k2v + 2*k3v + k4v)
    om_new = om + dt / 6 * (k1o + 2*k2o + 2*k3o + k4o)

    om_eff = (om1 + 2*om2 + 2*om3 + om4) / 6
    R_new = R @ _exp_so3(dt * om_eff)

    return p_new, v_new, R_new, om_new