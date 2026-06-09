import time
from dataclasses import dataclass, field
from typing import List
import math
import threading

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie

@dataclass
class DroneState:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attitude: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # roll,pitch,yaw
    log_conf: LogConfig = field(init=False)

# CONSTS
DEFAULT_HEIGHT = 1.3    # Default takeoff altitude (m)
DEFAULT_SPEED = 0.8    # Default speed (m/s)
BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [
    BASE_URI + f"{i}"
    for i in range(1, 5)   # 01 → 04
]
SPHERE_RADIUS = 0.5  # circle radius
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
DT = 0.5 # time step
DEADLINE_TIME = 10 # max time an action can take (sec)

def uri_is_alive(uri, timeout=0.5):
    """
    Try connecting briefly to check if a Crazyflie exists.
    """
    try:
        cf = Crazyflie(rw_cache='./cache')

        with SyncCrazyflie(uri, cf=cf) as scf:
            # If we get here, connection succeeded
            return True

    except Exception:
        return False


def discover_active_uris():
    active = []

    for uri in CANDIDATE_URIS:
        print(f"Probing {uri} ...")
        if uri_is_alive(uri):
            print(f"  ✔ Found: {uri}")
            active.append(uri)
        else:
            print(f"  ✖ Not found: {uri}")

    return active

def stop_logs(scf, state: DroneState):
    drone_states[scf.cf.link_uri].log_conf.stop()
    
def get_position_callback(uri, data):
    state = drone_states[uri]

    state.position[0] = data['stateEstimate.x']
    state.position[1] = data['stateEstimate.y']
    state.position[2] = data['stateEstimate.z']

def init_position_getting(scf: SyncCrazyflie):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(lambda _timestamp,
                    data,
                    _logconf: get_position_callback(scf.cf.link_uri, data))
    log_conf.start()

    drone_states[scf.cf.link_uri].log_conf = log_conf
    
def light_check(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)
    time.sleep(2)
    scf.cf.param.set_value('led.bitmask', 0)

def arm(scf):
    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

def take_off(scf, state: DroneState):
    cf = scf.cf
    deadline = time.time() + DEADLINE_TIME
    while state.position[2] < DEFAULT_HEIGHT*0.9:
        if time.time() > deadline:
            land(scf, state)
            raise RuntimeError("Takeoff timeout")
        cf.commander.send_position_setpoint(state.position[0], state.position[1], DEFAULT_HEIGHT, 0)
        time.sleep(DT)

def land(scf: SyncCrazyflie, state: DroneState):
    cf = scf.cf
    x, y = state.position[0], state.position[1]

    deadline = time.time() + DEADLINE_TIME
    while state.position[2] > 0.12 and time.time() < deadline:
        cf.commander.send_position_setpoint(x, y, 0.1, 0)
        time.sleep(DT)

    cf.commander.send_position_setpoint(x, y, 0.0, 0)
    time.sleep(0.3)

    cf.commander.send_notify_setpoint_stop()

def move_fibonacci_circle(scf: SyncCrazyflie):
    cf = scf.cf
    uri = cf.link_uri
    iD = active_uris.index(uri)

    start_time = time.time()
    end_time = start_time + 30
    base_x = fib_pos[0, iD]
    base_y = fib_pos[1, iD]
    base_z = fib_pos[2, iD]
    
    while time.time() < end_time:
        t = time.time() - start_time
        cf.commander.send_position_setpoint(base_x + target.position[0],
                                            base_y + target.position[1],
                                            base_z + target.position[2],
                                            0)
        time.sleep(DT)

import numpy as np

def get_fibonacci_points(P, R = 1.0, sphere_center = None, verbose = False):
    """
    Generate Fibonacci-like points on a sphere.

    Parameters:
    - P: number of drones/points (must be odd and >= 3)
    - R: radius of sphere
    - sphere_center: (3,) array-like center of sphere
    - verbose: print output if True

    Returns:
    - fibonacci_points: (3, P) array
    """

    if P < 3:
        raise RuntimeError("Need at least 3 drones")

    if P % 2 == 0:
        raise RuntimeError("Need an odd number of drones")

    if sphere_center is None:
        sphere_center = np.array([0.0, 0.0, 0.0])
    else:
        sphere_center = np.asarray(sphere_center, dtype=float).reshape(3)

    # Number of points on one side
    N = (P - 1) // 2  # since P = 2N + 1

    # Golden ratio
    phi_golden = (1 + np.sqrt(5)) / 2

    # Index array
    idx = np.arange(-N, N + 1)

    # Latitude (theta) and longitude (phi)
    theta_pos = np.arccos(2 * idx / P)
    phi_pos = 2 * np.pi * idx / phi_golden

    # Differences (as in MATLAB diff)
    theta = np.diff(theta_pos)
    phi = np.diff(phi_pos)
    phi_step = phi[0]  # constant step used in your code

    # Initial values
    theta_tmp = theta_pos[0]
    phi_tmp = phi_pos[0]

    x = np.zeros(P)
    y = np.zeros(P)
    z = np.zeros(P)

    for i in range(P - 1):
        x[i] = R * np.sin(theta_tmp) * np.cos(phi_tmp)
        y[i] = R * np.sin(theta_tmp) * np.sin(phi_tmp)
        z[i] = R * np.cos(theta_tmp)

        theta_tmp += theta[i]
        phi_tmp += phi_step

    # Last point
    x[P - 1] = R * np.sin(theta_tmp) * np.cos(phi_tmp)
    y[P - 1] = R * np.sin(theta_tmp) * np.sin(phi_tmp)
    z[P - 1] = R * np.cos(theta_tmp)

    fibonacci_points = np.vstack([x, y, z]) + sphere_center.reshape(3, 1)

    if verbose:
        print("fibonacci_points:")
        print(fibonacci_points)

    return fibonacci_points

def update_virtual_target(target: DroneState, lock: threading.Lock):
    start_time = time.time()

    while True:
        with lock:
            t = time.time() - start_time

            target.position[0] = math.sin(DEFAULT_SPEED * t)
            target.position[1] = math.cos(DEFAULT_SPEED * t)
            target.position[2] = DEFAULT_HEIGHT 
        time.sleep(DT)

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    
    # get the drones on radio range (powered on)
    active_uris = discover_active_uris()
    if not active_uris:
        raise RuntimeError("No drones found")

    # initialize drone state
    drone_states = {
        uri: DroneState()
        for uri in active_uris
    }

    fib_pos = get_fibonacci_points(len(active_uris), SPHERE_RADIUS)

    # initiate the target motion
    lock = threading.Lock()
    target = DroneState()
    target_thread = threading.Thread(
        target=update_virtual_target,
        args=(target,lock),
        daemon=True
    )
    target_thread.start()
    
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(active_uris, factory=factory) as swarm:
        print("warning light")
        swarm.parallel_safe(light_check)
        swarm.reset_estimators()

        # store vars in Ground Control memory
        swarm.parallel_safe(init_position_getting)

        # arm vehicles
        swarm.parallel_safe(arm)

        print("Take off")
        swarm.parallel_safe(take_off, 
                            args_dict={uri: (state,) for uri, state in drone_states.items()})
        swarm.parallel_safe(move_fibonacci_circle)
        print("Sleepy time")
        swarm.parallel_safe(land, 
                            args_dict={uri: (state,) for uri, state in drone_states.items()})
        
        # end of flight clean up
        print("Landed: stopping coms")
        swarm.parallel_safe(stop_logs, 
                            args_dict={uri: (state,) for uri, state in drone_states.items()})