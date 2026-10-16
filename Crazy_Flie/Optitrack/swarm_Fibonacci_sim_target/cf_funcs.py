import time
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment

from Structs import *
from fiboancci_funcs import *
from helper_funcs import *

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie

# ============================================================
# PER-DRONE COMMANDS (run via swarm.parallel_safe)
# ============================================================

def capture_home(scf: SyncCrazyflie, drone_state: DroneState, position_lock: threading.Lock):
    uri = scf.cf.link_uri
    pos = get_position(drone_state, position_lock)
    with position_lock:
        drone_state.local_origin = pos
    yaw_deg = math.degrees(get_yaw(drone_state, position_lock))
    print(f"{uri}: home = {[round(p, 3) for p in pos]}, yaw = {yaw_deg:.1f} deg")
    if abs(yaw_deg) > 15.0:
        print(f"WARNING {uri}: the Kalman filter assumes yaw ~ 0 at reset. "
              f"Point the drone along +X of the tracking frame.")

def arm(scf: SyncCrazyflie):
    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

def warning_lights(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)
    time.sleep(2)
    scf.cf.param.set_value('led.bitmask', 0)

def send_position_setpoint(scf, x, y, z, yaw=0.0):
    scf.cf.commander.send_position_setpoint(float(x), float(y), float(z), float(yaw))

def ramp_to(scf, target_pos, duration, label, state, position_lock, abort_event, respect_abort=True):
    """Smoothly move from the current measured position to `target_pos`."""
    uri = scf.cf.link_uri
    start = get_position(state, position_lock)
    steps = max(1, int(duration / SETPOINT_PERIOD))

    for i in range(steps):
        if respect_abort and abort_event.is_set():
            return False
        if not safety_check(state, position_lock, abort_event):
            return False
        f = (i + 1) / steps
        f = f * f * (3.0 - 2.0 * f)  # smoothstep
        sp = [s + (t - s) * f for s, t in zip(start, target_pos)]
        send_position_setpoint(scf, sp[0], sp[1], sp[2], 0.0)
        time.sleep(SETPOINT_PERIOD)

    print(f"{uri}: {label} complete.")
    return True

def take_off(scf, state: DroneState, position_lock:threading.Lock, abort_event: threading.Event, param: Param):
    hx, hy, _ = state.local_origin
    return ramp_to(scf, [hx, hy, param.default_height], TAKEOFF_TIME, "takeoff", state, position_lock, abort_event)

def return_origin(scf, state: DroneState, position_lock:threading.Lock, abort_event: threading.Event, param: Param):
    """Return to a point above home at takeoff height."""
    uri = scf.cf.link_uri
    hx, hy, _ = state.local_origin
    return ramp_to(scf, [hx, hy, param.default_height], RETURN_TIME, "return to home", state, position_lock, abort_event)

def land(scf, state: DroneState, position_lock:threading.Lock, abort_event: threading.Event):
    """Always runs (ignores abort) as long as this drone is still tracked."""
    hx, hy, _ = state.local_origin
    return ramp_to(scf, [hx, hy, 0.0], LAND_TIME, "landing", state, position_lock, abort_event, respect_abort=False)

def stop_flight(scf):
    try:
        scf.cf.commander.send_stop_setpoint()
        scf.cf.commander.send_notify_setpoint_stop()
        scf.cf.supervisor.send_arming_request(False)
        print(f"{scf.cf.link_uri}: disarmed.")
    except Exception as e:
        print(f"{scf.cf.link_uri}: could not disarm: {e}")

def get_position_callback(data, state: DroneState):
    state.position[0] = data['stateEstimate.x']
    state.position[1] = data['stateEstimate.y']
    state.position[2] = data['stateEstimate.z']

def update_virtual_target(
    target: DroneState,
    lock: threading.Lock,
    param: Param,
    stop_event: threading.Event
):
    start_time = time.time()

    while not stop_event.is_set():

        t = time.time() - start_time

        pos = param.ref_target_traj["pos"](t)

        with lock:
            target.position[:] = pos

        time.sleep(param.dTi)

def get_target_offset(target: DroneState, target_lock: threading.Lock):
    with target_lock:
        return list(target.position)

def controller(
    scf: SyncCrazyflie,
    target_lock: threading.Lock,
    target: DroneState, 
    chaser_list_ref_lock: threading.Lock,
    chaser_list_ref, 
    fib_ref_points, 
    min_safe_dist, 
    active_uris,
    state: DroneState,
    param: Param
):
    """
    Runs continuously on each CF.
    """
    cf = scf.cf
    iD = active_uris.index(cf.link_uri)
    end_time = time.time() + param.Tend
    while time.time() < end_time:
        with target_lock:
            target_pos = np.array([
                target.position[0],
                target.position[1],
                target.position[2]
            ]).reshape(-1,1)
        
        with chaser_list_ref_lock:
            q_ref_list = chaser_list_ref.copy()

        chaser_pos = state.position.reshape(-1,1)
        q_s = chaser_pos - target_pos
        state.radii_to_target = np.linalg.norm(q_s)
        if state.radii_to_target < 1e-8:
            continue
        
        q_s = chaser_pos - target_pos
        # Desired reduced-attitude direction: fibonacci point relative to the
        # target, projected onto the unit sphere (q* = (p* - p_T)/||p* - p_T||)
        q_d = fib_ref_points[iD, :].reshape(3,1)
        q_d = q_d / np.linalg.norm(q_d)

        # Position reference generation: integrate the guidance state (eq. (11))
        q_ref_list[iD], e_q = linear_motion_ref(q_ref_list[iD], q_d, param)

        # Safe guard collisions: APF displacement applied to the guidance
        # references (eq. (14)), so references deflect around each other
        q_ref_list[iD] = apf_safe_guard(iD, q_ref_list[iD], q_ref_list,
                                        min_safe_dist, param)
        q_hat = q_ref_list[iD]

        with chaser_list_ref_lock:
            chaser_list_ref[iD] = q_ref_list[iD]

        # Update sphere radius reference (integrates its own state, eq. (15))
        state.desired.radii_to_target = update_radius(state.desired.radii_to_target, e_q, param)

        # ---------- Enforce S² constraint (local) ----------
        goal = state.desired.radii_to_target * q_hat + target_pos
        cf.commander.send_position_setpoint(
            float(goal[0, 0]),
            float(goal[1, 0]),
            float(goal[2, 0]),
            0.0
        )

        time.sleep(param.dTi)
  
# ============================================================
# TRACKING HELPERS
# ============================================================

def uri_is_alive(uri):
    """True if a Crazyflie answers on this URI (opens and closes a link)."""
    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')):
            return True
    except Exception:
        return False

def discover_active_uris():
    active = []
    for uri in CANDIDATE_URIS:
        print(f"Probing {uri} ...")
        if uri_is_alive(uri):
            print(f"  [OK] Found: {uri}")
            active.append(uri)
        else:
            print(f"  [--] Not found: {uri}")
        time.sleep(0.5)   # let the radio settle before the next link
    return active

def get_position(drone_state: DroneState, position_lock: threading.Lock):
    with position_lock:
        return drone_state.position

def get_yaw(drone_state: DroneState, position_lock: threading.Lock):
    with position_lock:
        return drone_state.attitude[2]

def all_tracking_is_valid(drone_states: dict[str, DroneState], position_lock: threading.Lock):
    """True if the given drone (or all drones if uri is None) has fresh data."""
    now = time.monotonic()
    with position_lock:
        states = list(drone_states.values())
        return bool(states) and all(
            s.last_update > 0.0 and now - s.last_update < TRACKING_TIMEOUT
            for s in states
        )

def one_tracking_is_valid(drone_state: DroneState, position_lock: threading.Lock):
    """True if the given drone (or all drones if uri is None) has fresh data."""
    now = time.monotonic()
    with position_lock:
        return bool(drone_state) and drone_state.last_update > 0.0 and now - drone_state.last_update < TRACKING_TIMEOUT

def wait_for_tracking(timeout, drone_states: dict[str, DroneState], position_lock: threading.Lock, active_uris):
    print("Waiting for valid OptiTrack tracking of all drones...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all_tracking_is_valid(drone_states, position_lock):
            print("OptiTrack tracking acquired for all drones.")
            return True
        time.sleep(0.05)
    for uri in active_uris:
        if not one_tracking_is_valid(drone_states[uri], position_lock):
            print(f"ERROR: no fresh tracking for {uri} (rigid body {RIGID_BODY_IDS[uri]})")
    return False

def safety_check(drone_state, position_lock, abort_event):
    if not one_tracking_is_valid(drone_state, position_lock):
        print(f"!!! OPTITRACK TRACKING LOST !!!")
        abort_event.set()
        return False
    if get_position(drone_state, position_lock)[2] > MAX_HEIGHT:
        print(f"!!! HEIGHT LIMIT EXCEEDED !!!")
        abort_event.set()
        return False
    return True

# ============================================================
# EXTERNAL POSITION (one thread per drone)
# ============================================================

def _extpos_loop(cf, uri, drone_state, stop_event, abort_event, position_lock):
    while not stop_event.is_set():
        if one_tracking_is_valid(drone_state, position_lock):
            x, y, z = get_position(drone_state, position_lock)
            cf.extpos.send_extpos(x, y, z)
        elif not abort_event.is_set():
            print(f"Stale tracking for {uri}: aborting.")
            abort_event.set()
        time.sleep(EXT_POS_PERIOD)

def start_extpos(
    scf: SyncCrazyflie,
    drone_state: DroneState,
    extpos_threads,
    stop_event: threading.Event,
    abort_event: threading.Event,
    position_lock: threading.Lock,
):
    t = threading.Thread(
        target=_extpos_loop,
        args=(
            scf.cf,
            scf.cf.link_uri,
            drone_state,
            stop_event,
            abort_event,
            position_lock,
        ),
        daemon=True,
    )
    t.start()
    extpos_threads.append(t)