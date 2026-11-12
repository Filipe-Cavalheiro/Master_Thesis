import numpy as np

from Structs import DroneState, Param
from helper_funcs import skew, unskew


def drone_mellinger_ctrl(agent: DroneState, dpsi_d, param: Param):
    """Mellinger controller: (thrust, body torque) for the desired state of agent."""
    zW = np.array([[0.0], [0.0], [1.0]])   # world frame zW is up
    R = agent.attitude.reshape(3, 3)
    zB = R[:, 2].reshape(3, 1)             # drone frame zB

    # Translation errors
    e_p = agent.position - agent.desired.position
    e_v = agent.velocity - agent.desired.velocity

    # Desired force vector
    f_d = (-param.controller.Mellinger.kp @ e_p
           - param.controller.Mellinger.kv @ e_v
           + param.sim.drone_mass * param.sim.gravity * zW
           + param.sim.drone_mass * agent.desired.acceleration)

    # Thrust, clipped to the motor limits
    thrust = np.squeeze(f_d.T @ zB)
    thrust = np.clip(thrust, 4 * param.sim.thrust_min, 4 * param.sim.thrust_max)

    # Desired rotation matrix
    norm_fd = np.linalg.norm(f_d)
    if norm_fd < 1e-8:
        norm_fd = 1e-8
    zB_d = f_d / norm_fd
    xC_d = np.array([np.cos(agent.desired.psi), np.sin(agent.desired.psi), 0.0]).reshape(3, 1)
    yB_d = (skew(zB_d) @ xC_d) / (np.linalg.norm(skew(zB_d) @ xC_d))
    xB_d = skew(yB_d) @ zB_d
    R_d = np.column_stack((xB_d, yB_d, zB_d))

    # Desired angular velocity
    hw_d = (param.sim.drone_mass / thrust) * (
        agent.desired.jerk - (zB_d.T @ agent.desired.jerk) * zB_d
    )
    ang_p_d = (-hw_d.T @ yB_d)[0]
    ang_q_d = (hw_d.T @ xB_d)[0]
    ang_r_d = (dpsi_d * zW.T @ zB_d)[0]     # minimum-snap style yaw rate projection
    om_d = np.array([ang_p_d, ang_q_d, ang_r_d])

    # Torques
    e_om = agent.ang_vel - om_d
    e_R = 0.5 * unskew((R_d.T @ R) - (R.T @ R_d))
    torque = (
        -param.controller.Mellinger.kR @ e_R
        - param.controller.Mellinger.kom @ e_om
    )

    return thrust, torque
