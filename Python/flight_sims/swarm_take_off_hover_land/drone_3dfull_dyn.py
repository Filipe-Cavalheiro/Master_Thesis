import numpy as np

from Structs import Param


def drone_3dfull_dyn(v, R, om, thrust, tau, param: Param, f_external=None, tau_external=None):
    """
    Full 3D drone dynamics: v world velocity (3,1), R body -> world, om body
    angular velocity (3,1), thrust magnitude, tau body torque (3,1).
    Returns (dot_p, dot_v, dot_om).
    """
    zW = np.array([[0.0], [0.0], [1.0]])
    zB = R[:, 2:3]

    f_external = np.zeros((3, 1)) if f_external is None else np.asarray(f_external, dtype=float).reshape(3, 1)
    tau_external = np.zeros((3, 1)) if tau_external is None else np.asarray(tau_external, dtype=float).reshape(3, 1)

    tau = np.asarray(tau, dtype=float).reshape(3, 1)
    thrust = float(np.asarray(thrust).item())

    # Translational drag: assumes D is expressed in the body frame
    D = np.abs(param.sim.drag_matrix)
    drag_force_world = -R @ D @ R.T @ v

    d_p = v
    d_v = (
        -param.sim.gravity * zW
        + (thrust / param.sim.drone_mass) * zB
        + (f_external + drag_force_world) / param.sim.drone_mass
    )

    J = param.sim.J
    angular_momentum = J @ om
    gyroscopic = np.cross(om.ravel(), angular_momentum.ravel()).reshape(3, 1)
    d_om = np.linalg.solve(J, -gyroscopic + tau + tau_external)

    return d_p, d_v, d_om
