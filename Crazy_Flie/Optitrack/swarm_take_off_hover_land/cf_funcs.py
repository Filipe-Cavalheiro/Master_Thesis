import time
import math
import threading
from pathlib import Path

import numpy as np

from Structs import *
import ctrl_c

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

CACHE_DIR = str(Path(__file__).resolve().parent / "cache")

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
    if abs(yaw_deg) > MAX_START_YAW_DEG:
        raise RuntimeError(f"{uri}: yaw {yaw_deg:.1f} deg, the Kalman filter assumes yaw ~ 0 at reset. "
                           f"Point the drone along +X of the tracking frame.")

def set_onboard_controller(scf: SyncCrazyflie):
    if ONBOARD_ESTIMATOR is not None:
        scf.cf.param.set_value('stabilizer.estimator', str(ONBOARD_ESTIMATOR))
    if ONBOARD_CONTROLLER is not None:
        scf.cf.param.set_value('stabilizer.controller', str(ONBOARD_CONTROLLER))
    time.sleep(0.1)

def check_battery(scf: SyncCrazyflie):
    """Raises if the battery voltage (at rest) is below MIN_BATTERY_VOLTAGE."""
    log = LogConfig(name='battery', period_in_ms=100)
    log.add_variable('pm.vbat', 'float')
    with SyncLogger(scf, log) as logger:
        for _, data, _ in logger:
            vbat = data['pm.vbat']
            break
    print(f"{scf.cf.link_uri}: battery {vbat:.2f} V")
    if vbat < MIN_BATTERY_VOLTAGE:
        raise RuntimeError(f"{scf.cf.link_uri}: battery {vbat:.2f} V < {MIN_BATTERY_VOLTAGE} V")

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
        if ctrl_c.killed():
            return False
        if respect_abort and abort_event.is_set():
            return False
        if not safety_check(state, position_lock, abort_event, check_arena=respect_abort):
            return False
        f = (i + 1) / steps
        f = f * f * (3.0 - 2.0 * f)  # smoothstep
        sp = [s + (t - s) * f for s, t in zip(start, target_pos)]
        send_position_setpoint(scf, sp[0], sp[1], sp[2], 0.0)
        time.sleep(SETPOINT_PERIOD)

    print(f"{uri}: {label} complete.")
    return True

def take_off(scf, state: DroneState, position_lock: threading.Lock, abort_event: threading.Event):
    hx, hy, _ = state.local_origin
    return ramp_to(scf, [hx, hy, TAKEOFF_HEIGHT], TAKEOFF_TIME, "takeoff", state, position_lock, abort_event)

def return_origin(scf, state: DroneState, position_lock: threading.Lock, abort_event: threading.Event,
                  duration: float):
    """
    Fly to the assigned landing spot (above a takeoff point), same duration
    for every drone. Also runs after an abort (as long as this drone is still
    tracked).
    """
    return ramp_to(scf, list(state.land_spot), duration, "return to the takeoff points",
                   state, position_lock, abort_event, respect_abort=False)

def land(scf, state: DroneState, position_lock: threading.Lock, abort_event: threading.Event):
    """Vertical landing in place. Always runs (ignores abort) as long as this drone is still tracked."""
    x, y, _ = get_position(state, position_lock)
    return ramp_to(scf, [x, y, LAND_HEIGHT], LAND_TIME, "landing", state, position_lock, abort_event,
                   respect_abort=False)

def stop_flight(scf):
    try:
        scf.cf.commander.send_stop_setpoint()
        scf.cf.commander.send_notify_setpoint_stop()
        scf.cf.supervisor.send_arming_request(False)
        print(f"{scf.cf.link_uri}: disarmed.")
    except Exception as e:
        print(f"{scf.cf.link_uri}: could not disarm: {e}")

# ============================================================
# TRACKING HELPERS
# ============================================================

def uri_is_alive(uri):
    """True if a Crazyflie answers on this URI (opens and closes a link)."""
    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=CACHE_DIR)):
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
        return [float(c) for c in np.ravel(drone_state.position)]

def get_yaw(drone_state: DroneState, position_lock: threading.Lock):
    with position_lock:
        return drone_state.yaw

def all_tracking_is_valid(drone_states: dict[str, DroneState], position_lock: threading.Lock):
    """True if every drone has fresh data."""
    now = time.monotonic()
    with position_lock:
        states = list(drone_states.values())
        return bool(states) and all(
            s.last_update > 0.0 and now - s.last_update < TRACKING_TIMEOUT
            for s in states
        )

def one_tracking_is_valid(drone_state: DroneState, position_lock: threading.Lock):
    """True if the given drone has fresh data."""
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

def inside_arena(drone_state: DroneState, position_lock: threading.Lock):
    p = np.array(get_position(drone_state, position_lock))
    return bool(np.all(p[:2] >= ARENA_MIN[:2]) and np.all(p <= ARENA_MAX))

def safety_check(drone_state, position_lock, abort_event, check_arena=True):
    if not one_tracking_is_valid(drone_state, position_lock):
        print(f"!!! OPTITRACK TRACKING LOST !!!")
        abort_event.set()
        return False
    if check_arena and not inside_arena(drone_state, position_lock):
        print(f"!!! DRONE OUTSIDE THE ARENA !!!")
        abort_event.set()
        return False
    return True

# ============================================================
# EXTERNAL POSITION (one thread per drone)
# ============================================================

def _extpos_loop(cf, uri, drone_state, stop_event, abort_event, position_lock):
    """
    Sends the OptiTrack position to the Kalman filter every EXT_POS_PERIOD;
    sets abort_event (once) if the tracking becomes stale.
    """
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
    """Starts the daemon thread streaming the external position to one drone."""
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
