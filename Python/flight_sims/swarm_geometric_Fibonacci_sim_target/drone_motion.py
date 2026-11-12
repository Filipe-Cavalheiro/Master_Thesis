from Structs import DroneState, Param
from drone_3dfull_dyn import drone_3dfull_dyn
from helper_funcs import *
from drone_mellinger_ctrl import drone_mellinger_ctrl

def drone_motion(drone: DroneState, param: Param):
    """
    Propagates the drone drone one timestep.

    Parameters
    ----------
    drone : dict
        drone state with keys:
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
    pos = drone.position
    vel = drone.velocity
    R = drone.attitude.reshape(3,3)
    om = drone.ang_vel

    # Desired quantities (trajectory tracking)
    dpsi_d = (drone.desired.psi - drone.desired.old_psi)/param.sim.dTi
    drone.desired.old_psi = drone.desired.psi

    # Controller
    thrust, torque = drone_mellinger_ctrl(
        drone,
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

    drone.position = p_new
    drone.velocity = v_new
    drone.attitude = R_new.reshape(-1, 1)
    drone.ang_vel = om_new

def passive_motion(drone: DroneState, param: Param):
    """Motors stopped (stop_flight): the dynamics with no thrust and no torque, one timestep."""
    def deriv(p_, v_, R_, om_):
        return drone_3dfull_dyn(v_, R_, om_, 0.0, np.zeros((3, 1)), param)

    p_new, v_new, R_new, om_new = rk4_step_rigid_body(
        deriv, drone.position, drone.velocity, drone.attitude.reshape(3, 3), drone.ang_vel, param.sim.dTi)
    drone.position = p_new
    drone.velocity = v_new
    drone.attitude = R_new.reshape(-1, 1)
    drone.ang_vel = om_new


def ground_contact(drone: DroneState, z_floor):
    """Drone resting on the floor: no penetration, no sliding, level (yaw kept). True if in contact."""
    if drone.position[2, 0] < z_floor or (drone.position[2, 0] <= z_floor + 1e-9 and drone.velocity[2, 0] < 0):
        drone.position[2, 0] = z_floor
        drone.velocity[:] = 0.0
        R = drone.attitude.reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        drone.attitude = Euler2R([0.0, 0.0, yaw]).reshape(-1, 1)
        drone.ang_vel[:] = 0.0
        return True
    return False
