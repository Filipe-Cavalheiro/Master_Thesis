import numpy as np
from Structs_sim import Param
from Structs_agent import DroneState
from helper_funcs import skew, unskew

def drone_mellinger_ctrl(agent: DroneState, dpsi_d, param: Param):
    # Auxiliary variables
    zW = np.array([[0.0], [0.0], [1.0]])   # world frame zW is up
    R = agent.attitude.reshape(3,3)
    zB = R[:,2].reshape(3,1)                # drone frame zB

    # Translation errors
    e_p = agent.position - agent.desired.position
    e_v = agent.velocity - agent.desired.velocity

    # Mass of the feedforward terms: the controller's own mass (Crazyflie
    # firmware: ctrlMel.mass, set by the flight code) if given, else the
    # model mass. The dynamics always use the model mass (param.sim.drone_mass).
    mel = param.controller.Mellinger
    ff_mass = mel.get("mass") or param.sim.drone_mass

    # Desired force vector with attitude
    f_d = (-mel["kp"] @ e_p
           - mel["kv"] @ e_v
           + ff_mass * param.sim.gravity * zW
           + ff_mass * agent.desired.acceleration)

    # Integral of the position error (Crazyflie firmware controller_mellinger.c:
    # i_error += (setpoint - position) * dt, clamped to +-i_range per axis,
    # force += ki * i_error), if the gains have "ki" / "i_range"
    ki = mel.get("ki")
    if ki is not None:
        i_range = np.asarray(mel["i_range"], dtype=float).reshape(3, 1)
        agent.pos_i_error = np.clip(agent.pos_i_error - e_p * param.sim.dTi, -i_range, i_range)
        f_d = f_d + ki @ agent.pos_i_error

    # Optional tilt limit (None = no limit): the horizontal part of the
    # desired force is scaled so the commanded tilt stays below max_tilt_deg
    max_tilt = getattr(param.controller, "max_tilt_deg", None)
    if max_tilt is not None and f_d[2, 0] > 0:
        f_h = np.linalg.norm(f_d[:2, 0])
        f_h_max = f_d[2, 0] * np.tan(np.radians(max_tilt))
        if f_h > f_h_max:
            f_d[:2, 0] *= f_h_max / f_h

    # Compute thrust
    thrust =  np.squeeze(f_d.T @ zB)
    # clip it since the drone as a maximum and minimum values
    thrust = np.clip(thrust, 4*param.sim.thrust_min, 4*param.sim.thrust_max)

    # Compute desired rotation matrix
    norm_fd = np.linalg.norm(f_d)
    if norm_fd < 1e-8:
        norm_fd = 1e-8

    zB_d = f_d / norm_fd

    # Desired yaw direction
    xC_d = np.array([np.cos(agent.desired.psi), np.sin(agent.desired.psi), 0.0]).reshape(3,1)

    # yB_d = skew(zB_d) * xC_d / norm(...)
    yB_d = (skew(zB_d)@xC_d)/(np.linalg.norm(skew(zB_d)@xC_d))

    xB_d = skew(yB_d)@zB_d

    R_d = np.column_stack((xB_d, yB_d, zB_d))

    # Compute desired angular velocity
    hw_d = (ff_mass / thrust) * (
        agent.desired.jerk - (zB_d.T@agent.desired.jerk) * zB_d
    )

    ang_p_d = (-hw_d.T@ yB_d)[0]
    ang_q_d = (hw_d.T@ xB_d)[0]

    # Minimum-snap style yaw rate projection
    ang_r_d = (dpsi_d * zW.T@ zB_d)[0]

    om_d = np.array([
        ang_p_d,
        ang_q_d,
        ang_r_d
    ])

    # Compute torques
    e_om = agent.ang_vel - om_d

    e_R = 0.5 * unskew(
        (R_d.T @ R) - (R.T @ R_d)
    )

    torque = (
        -param.controller.Mellinger["kR"] @ e_R
        -param.controller.Mellinger["kom"] @ e_om
    )

    return thrust, torque
