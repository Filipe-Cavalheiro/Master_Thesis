from Structs import DroneState, Param
from drone_3dfull_dyn import drone_3dfull_dyn
from helper_funcs import *
from drone_mellinger_ctrl import drone_mellinger_ctrl

def agent_motion(agent: DroneState, param: Param):
    """
    Propagates the agent drone one timestep.

    Parameters
    ----------
    agent : dict
        agent state with keys:
            - "pos" : (3,1)
            - "vel" : (3,1)
            - "attitude" : (3,3) rotation matrix
            - "ang_vel" : (3,1)
    p_d : ndarray
        Desired position (3,1)
    v_d : ndarray
        Desired velocity (3,1)
    Param : dict
        Simulation parameters.
    """

    # Extract state
    pos = agent.position
    vel = agent.velocity
    R = agent.attitude.reshape(3,3)
    om = agent.ang_vel

    # Desired quantities (trajectory tracking)
    dpsi_d = (agent.desired.psi - agent.desired.old_psi)/param.sim.dTi
    agent.desired.old_psi = agent.desired.psi

    # Controller
    thrust, torque = drone_mellinger_ctrl(
        agent,
        dpsi_d, 
        param
    )

    # Drone dynamics wrapped for RK4
    def deriv(p_, v_, R_, om_):
        dot_p, dot_v, dot_om = drone_3dfull_dyn(
            v_,
            R_,
            om_,
            thrust,
            torque,
            param,
        )
        return dot_p, dot_v, dot_om

    # RK4 integration
    dt = param.sim.dTi

    p_new, v_new, R_new, om_new = rk4_step_rigid_body(deriv, pos, vel, R, om, dt)

    agent.position = p_new
    agent.velocity = v_new
    agent.attitude = R_new.reshape(-1, 1)
    agent.ang_vel = om_new

def passive_motion(agent: DroneState, param: Param):
    """Motors stopped (stop_flight): the dynamics with no thrust and no torque, one timestep."""
    def deriv(p_, v_, R_, om_):
        return drone_3dfull_dyn(v_, R_, om_, 0.0, np.zeros((3, 1)), param)

    p_new, v_new, R_new, om_new = rk4_step_rigid_body(
        deriv, agent.position, agent.velocity, agent.attitude.reshape(3, 3), agent.ang_vel, param.sim.dTi)
    agent.position = p_new
    agent.velocity = v_new
    agent.attitude = R_new.reshape(-1, 1)
    agent.ang_vel = om_new


def ground_contact(agent: DroneState, z_floor):
    """Drone resting on the floor: no penetration, no sliding, level (yaw kept). True if in contact."""
    if agent.position[2, 0] < z_floor or (agent.position[2, 0] <= z_floor + 1e-9 and agent.velocity[2, 0] < 0):
        agent.position[2, 0] = z_floor
        agent.velocity[:] = 0.0
        R = agent.attitude.reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        agent.attitude = Euler2R([0.0, 0.0, yaw]).reshape(-1, 1)
        agent.ang_vel[:] = 0.0
        return True
    return False
