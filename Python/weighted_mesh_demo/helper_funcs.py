
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

def directional_alignment(generator_positions,
                          target_position,
                          target_velocity):
    """
    Calculate directional alignment metrics for a spherical formation.

    Parameters
    ----------
    generator_positions : np.ndarray
        Array of shape (N, 3) containing generator positions.
    target_position : np.ndarray
        Target position, shape (3,).
    target_velocity : np.ndarray
        Target velocity vector, shape (3,).

    Returns
    -------
    mean_alignment : float
        Mean directional alignment A, in [-1, 1].

    forward_fraction : float
        Fraction of generators in the forward hemisphere.

    angles : np.ndarray
        Angular displacement of each generator from the
        direction of motion, in radians.

    individual_alignment : np.ndarray
        Alignment value for each generator.
    """

    # Relative position of each generator with respect to the target
    relative_positions = generator_positions - target_position

    # Norm of each relative position
    distances = np.linalg.norm(relative_positions, axis=1)

    # Unit velocity vector
    velocity_norm = np.linalg.norm(target_velocity)

    if velocity_norm == 0:
        raise ValueError(
            "Target velocity must be non-zero to calculate "
            "directional alignment."
        )

    velocity_direction = target_velocity / velocity_norm

    # Normalised generator directions
    generator_directions = (
        relative_positions / distances[:, np.newaxis]
    )

    # Dot product with velocity direction
    dot_products = np.sum(
        generator_directions * velocity_direction,
        axis=1
    )

    # Protect arccos against floating-point errors
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Angular displacement from direction of motion
    angles = np.arccos(dot_products)

    # Individual alignment values
    individual_alignment = dot_products

    # Mean directional alignment
    mean_alignment = np.mean(individual_alignment)

    # Fraction of generators in the forward hemisphere
    forward_fraction = np.mean(dot_products > 0.0)

    return (
        mean_alignment,
        forward_fraction,
        angles,
        individual_alignment
    )