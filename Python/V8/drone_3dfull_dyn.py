import numpy as np
from helper_funcs import *
from Structs import Param

def drone_3dfull_dyn(
    v, R, om, thrust, tau, param: Param,
    f_external=None,
    tau_external=None,
):
    """
    Full 3D drone dynamics.

    Parameters
    ----------
    v : np.ndarray
        Velocity in world frame (3,)
    R : np.ndarray
        Rotation matrix body -> world (3,3)
    om : np.ndarray
        Angular velocity in body frame (3,)
    thrust : float
        thrust magnitude
    tau : np.ndarray
        Body torque (3,)
    param : object
        Drone parameters

    Returns
    -------
    dp : np.ndarray
        Position derivative
    dv : np.ndarray
        Velocity derivative
    dR : np.ndarray
        Rotation matrix derivative
    dom : np.ndarray
        Angular velocity derivative
    """

    # Column vectors
    zW = np.array([[0.0], [0.0], [1.0]])
    zB = R[:, 2:3]

    f_external = (
        np.zeros((3, 1))
        if f_external is None else
        np.asarray(f_external, dtype=float).reshape(3, 1)
    )

    tau_external = (
        np.zeros((3, 1))
        if tau_external is None else
        np.asarray(tau_external, dtype=float).reshape(3, 1)
    )

    tau = np.asarray(tau, dtype=float).reshape(3, 1)
    thrust = float(np.asarray(thrust).item())

    # Translational drag: assumes D is expressed in the body frame
    D = np.abs(param.sim.drag_matrix)
    drag_force_world = -R @ D @ R.T @ v

    # Translational dynamics
    d_p = v

    d_v = (
        -param.sim.gravity * zW
        + (thrust / param.sim.drone_mass) * zB
        + (f_external + drag_force_world)
          / param.sim.drone_mass
    )

    # Rotational dynamics
    J = param.sim.J

    angular_momentum = J @ om
    gyroscopic = np.cross(om.ravel(), angular_momentum.ravel()).reshape(3, 1)

    d_om = np.linalg.solve(
        J,
        -gyroscopic + tau + tau_external
    )

    return d_p, d_v, d_om