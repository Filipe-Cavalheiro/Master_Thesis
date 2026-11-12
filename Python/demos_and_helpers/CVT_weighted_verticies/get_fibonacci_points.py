import numpy as np

from Structs import Param 

def get_fibonacci_points(sphere_center, param: Param, verbose=False):
    """
    This function returns the desired trajectory for drones to form a sphere
    around a point using Fibonacci sphere distribution.

    Parameters:
        P : int
            Number of drones (must be odd and >= 3)
        R : float
            Radius of the sphere
        sphere_center : array-like, optional
            Center of the sphere (default: [0, 0, 0])
        verbose : bool, optional
            Print generated points if True

    Returns:
        fibonacci_points : ndarray
            3xP array containing Cartesian coordinates of drone positions
    """ 
    P = param.sim.nD                # number of drones
    R = param.sim.sphere_radius     # sphere radius

    if P < 3:
        raise ValueError("Not enough elements (P must be >= 3).")

    if P % 2 == 0:

        raise ValueError(
            f"Even number of elements is not allowed. "
            f"Currently {param.sim.nD} elements have been assigned."
        )

    if sphere_center is None:
        sphere_center = np.array([0, 0, 0])

    sphere_center = np.asarray(sphere_center).reshape(3, 1)

    # Number of fibonacci points
    N = (P - 1) // 2

    # Golden ratio
    phi_golden = (1 + np.sqrt(5)) / 2

    # Generate indices
    idx = np.arange(-N, N + 1)

    # Latitude and longitude
    theta_pos = np.arccos(2 * idx / P)
    theta = np.diff(theta_pos)

    phi_pos = 2 * np.pi * idx / phi_golden
    phi = np.diff(phi_pos)[0]

    theta_tmp = theta_pos[0]
    phi_tmp = phi_pos[0]

    x_fibonacci = np.zeros(P)
    y_fibonacci = np.zeros(P)
    z_fibonacci = np.zeros(P)

    for i in range(P - 1):
        # Cartesian coordinates
        x_fibonacci[i] = R * np.sin(theta_tmp) * np.cos(phi_tmp)
        y_fibonacci[i] = R * np.sin(theta_tmp) * np.sin(phi_tmp)
        z_fibonacci[i] = R * np.cos(theta_tmp)

        theta_tmp += theta[i]
        phi_tmp += phi

    # Last point
    x_fibonacci[P - 1] = R * np.sin(theta_tmp) * np.cos(phi_tmp)
    y_fibonacci[P - 1] = R * np.sin(theta_tmp) * np.sin(phi_tmp)
    z_fibonacci[P - 1] = R * np.cos(theta_tmp)

    fibonacci_points = np.vstack(
        (x_fibonacci, y_fibonacci, z_fibonacci)
    ) + sphere_center

    if verbose:
        print("fibonacci_points:")
        print(fibonacci_points)

    return fibonacci_points