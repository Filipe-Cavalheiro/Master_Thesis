import time
import threading
import numpy as np

from Structs import *
from voronoi_mesh_funcs import *

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie

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
    state.log_conf.stop()
    
def get_position_callback(data, state: DroneState):
    state.position[0] = data['stateEstimate.x']
    state.position[1] = data['stateEstimate.y']
    state.position[2] = data['stateEstimate.z']

def init_position_getting(scf: SyncCrazyflie, state: DroneState):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(lambda _timestamp,
                    data,
                    _logconf: get_position_callback(data, state))
    log_conf.start()

    state.log_conf = log_conf
    
def light_check(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)
    time.sleep(2)
    scf.cf.param.set_value('led.bitmask', 0)

def arm(scf: SyncCrazyflie):
    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

def take_off(scf, state: DroneState, param: Param):
    cf = scf.cf
    deadline = time.time() + param.deadline_time
    while state.position[2] < param.default_height*0.9:
        if time.time() > deadline:
            land(scf, state, param)
            raise RuntimeError("Takeoff timeout")
        cf.commander.send_position_setpoint(state.position[0], state.position[1], param.default_height, 0)
        time.sleep(param.dTi)

def land(scf: SyncCrazyflie, state: DroneState, param: Param):
    cf = scf.cf
    x, y = state.position[0], state.position[1]

    deadline = time.time() + param.deadline_time
    while state.position[2] > 0.12 and time.time() < deadline:
        cf.commander.send_position_setpoint(x, y, 0.1, 0)
        time.sleep(param.dTi)

    cf.commander.send_position_setpoint(x, y, 0.0, 0)
    time.sleep(0.3)

    cf.commander.send_notify_setpoint_stop()

def controller(
    scf: SyncCrazyflie,
    active_uris,
    state: DroneState,
    param: Param
):  
    """
    Runs continuously on each CF.
    """
    cf = scf.cf
    my_idx = active_uris.index(cf.link_uri)
    if my_idx != len(active_uris)-1: # last drone is target
        # === Chasers code ===
        end_time = time.time() + param.Tend
        while time.time() < end_time:
            with desired_lock:
                dp = desired_positions
                if dp is None or my_idx >= len(dp):
                    # hover in place
                    cf.commander.send_position_setpoint(
                        state.position[0],
                        state.position[1],
                        state.position[2],
                        0.0
                    )
                    time.sleep(param.dTi)
                    continue

                goal = dp[my_idx]

            cf.commander.send_position_setpoint(
                float(goal[0]),
                float(goal[1]),
                float(goal[2]),
                0.0
            )

            time.sleep(param.dTi)
    else:
        # === Target code ===
        start_time = time.time()
        end_time = start_time + param.Tend
        while time.time() < end_time:
            t = time.time() - start_time

            pos = param.ref_target_traj["pos"](t)

            cf.commander.send_position_setpoint(
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                0.0
            )

            time.sleep(param.dTi)


# ============================================================
# Shared desired positions
# ============================================================

desired_positions = None
desired_lock = threading.Lock()

# ============================================================
# Voronoi thread
# ============================================================
def voronoi_worker(
    target_state: DroneState,
    target_lock: threading.Lock,
    param: Param,
    stop_event: threading.Event

):
    global desired_positions
    voronoi = Voronoi(param)

    p_d, swarm_center, voronoi = pre_compute_desired_pos(
        voronoi,
        param
    )

    with desired_lock:
        desired_positions = p_d.copy()

    k = 0
    while not stop_event.is_set():        
        with target_lock:
            target_pos = np.array([
                target_state.position[0],
                target_state.position[1],
                target_state.position[2]
            ])

        voronoi.vertices, voronoi.faces, voronoi.mesh_pos = (
            pos_based_mesh_transform(
                target_pos,
                voronoi,
                param
            )
        )

        if k == 1 or k % 5 == 0:
            voronoi.sites = runCVT(voronoi, voronoi.sites)

        p_d = voronoi.vertices[voronoi.sites, :]

        with desired_lock:
            desired_positions = p_d.copy()

        k += 1
        time.sleep(param.dTi)