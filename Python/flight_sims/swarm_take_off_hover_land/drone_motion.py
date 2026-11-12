import numpy as np
from scipy.spatial.transform import Rotation

from Structs import DroneState, Param
from drone_3dfull_dyn import drone_3dfull_dyn
from drone_mellinger_ctrl import drone_mellinger_ctrl
from helper_funcs import *


def drone_motion(drone: DroneState, param: Param):
    """Propagates the drone one time step (param.sim.dTi): Mellinger controller + dynamics, RK4."""
    pos = drone.position
    vel = drone.velocity
    R = drone.attitude.reshape(3, 3)
    om = drone.ang_vel

    dpsi_d = (drone.desired.psi - drone.desired.old_psi) / param.sim.dTi
    drone.desired.old_psi = drone.desired.psi

    thrust, torque = drone_mellinger_ctrl(drone, dpsi_d, param)

    def deriv(p_, v_, R_, om_):
        return drone_3dfull_dyn(v_, R_, om_, thrust, torque, param)

    p_new, v_new, R_new, om_new = rk4_step_rigid_body(deriv, pos, vel, R, om, param.sim.dTi)

    drone.position = p_new
    drone.velocity = v_new
    drone.attitude = R_new.reshape(-1, 1)
    drone.ang_vel = om_new


def passive_motion(agent: DroneState, param: Param):
    """Motors stopped: the dynamics with no thrust and no torque."""
    def deriv(p_, v_, R_, om_):
        return drone_3dfull_dyn(v_, R_, om_, 0.0, np.zeros((3, 1)), param)
    p, v, R, om = rk4_step_rigid_body(deriv, agent.position, agent.velocity,
                                      agent.attitude.reshape(3, 3), agent.ang_vel, param.sim.dTi)
    agent.position, agent.velocity, agent.attitude, agent.ang_vel = p, v, R.reshape(-1, 1), om


def ground_contact(agent: DroneState, z_floor):
    """Drone resting on the floor: no penetration, no sliding, level (yaw kept)."""
    if agent.position[2, 0] < z_floor or (agent.position[2, 0] <= z_floor + 1e-9 and agent.velocity[2, 0] < 0):
        agent.position[2, 0] = z_floor
        agent.velocity[:] = 0.0
        yaw = Rotation.from_matrix(agent.attitude.reshape(3, 3)).as_euler("ZYX")[0]
        agent.attitude = Rotation.from_euler("z", yaw).as_matrix().reshape(-1, 1)
        agent.ang_vel[:] = 0.0
        return True
    return False
