import numpy as np

from drone_mellinger_ctrl import drone_mellinger_ctrl
from drone_3dfull_dyn import drone_3dfull_dyn
from helper_funcs import rk4_step_rigid_body
from Structs_sim import Param
from Structs_agent import DroneState

def mellinger_command(agent: DroneState, param: Param):
    """Thrust and body torque from the Mellinger controller (updates the yaw reference)."""
    # Reference yaw rate
    dpsi_d = (agent.desired.psi - agent.desired.old_psi) / param.sim.dTi
    agent.desired.old_psi = agent.desired.psi

    # Mellinger controller
    return drone_mellinger_ctrl(
        agent,
        dpsi_d,
        param
    )


def rk4_propagate(agent: DroneState, thrust, torque, param: Param):
    """Propagates the drone one timestep with RK4 (attitude on SO(3)), thrust and torque held constant."""
    # Drone dynamics wrapped for RK4
    def deriv(p_, v_, R_, om_):
        return drone_3dfull_dyn(
            v_,
            R_,
            om_,
            thrust,
            torque,
            param,
        )

    p_new, v_new, R_new, om_new = rk4_step_rigid_body(
        deriv,
        agent.position,
        agent.velocity,
        agent.attitude.reshape(3, 3),
        agent.ang_vel,
        param.sim.dTi,
    )

    agent.position = p_new
    agent.velocity = v_new
    agent.attitude = R_new.reshape(-1, 1)
    agent.ang_vel = om_new


def actuate_control(agent: DroneState, param: Param):
    if param.controller.selected_controller == "Mellinger":
        thrust, torque = mellinger_command(agent, param)

        # RK4 integration of the continuous-time drone dynamics
        rk4_propagate(agent, thrust, torque, param)

    else:
        raise ValueError(
            f"Unknown controller type: {param.controller}"
        )

    return agent


def actuate_passive(agent: DroneState, param: Param):
    """
    Dropped (powered-off) drone: passive dynamics until it reaches the ground,
    then it stays there.
    """
    pos = agent.position

    if pos[2] <= 0:
        agent.position = np.array(
            [pos[0,0], pos[1,0], 0]
        ).reshape(-1,1)

        agent.velocity = np.array([0,0,0]).reshape(-1,1)

    else:
        # motors off: no thrust nor torque
        rk4_propagate(agent, 0, np.zeros([3,1]), param)

    return agent
