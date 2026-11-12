import time
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment

from Structs import *
import ctrl_c
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

def set_mellinger(scf: SyncCrazyflie):
    """Kalman estimator and Mellinger controller, with the gains of MELLINGER_PARAMS
    (call before swarm.reset_estimators)."""
    scf.cf.param.set_value("stabilizer.estimator", str(ONBOARD_ESTIMATOR))
    scf.cf.param.set_value("stabilizer.controller", str(ONBOARD_CONTROLLER))
    for name, value in MELLINGER_PARAMS.items():
        scf.cf.param.set_value(name, str(value))
    time.sleep(0.2)

def read_firmware_params(scf: SyncCrazyflie, firmware_params: dict):
    """The LOGGED_FIRMWARE_PARAMS of this drone, as the firmware has them (None if it lacks one)."""
    values = {}
    for name in LOGGED_FIRMWARE_PARAMS:
        try:
            v = scf.cf.param.get_value(name)
            values[name] = float(v) if v is not None else None
        except Exception:
            values[name] = None
    firmware_params[scf.cf.link_uri] = values
    est, ctrl = values.get("stabilizer.estimator"), values.get("stabilizer.controller")
    if est != ONBOARD_ESTIMATOR or ctrl != ONBOARD_CONTROLLER:
        raise RuntimeError(f"{scf.cf.link_uri}: estimator {est}, controller {ctrl} "
                           f"(expected {ONBOARD_ESTIMATOR}, {ONBOARD_CONTROLLER}): not arming")
    print(f"{scf.cf.link_uri}: controller {values.get('stabilizer.controller')}, "
          f"kp_xy {values.get('ctrlMel.kp_xy')}, kd_xy {values.get('ctrlMel.kd_xy')}")

def init_guidance(drone_states: dict[str, DroneState], active_uris, param: Param):
    """
    Initial guidance state of every drone (call after capture_home): q_ref is
    the direction from the target's start position to the drone's takeoff
    point (home x, home y, default_height), and the radius state its distance,
    so the first controller setpoint is where the takeoff left the drone.
    Returns q_ref_list in active_uris order.
    """
    target0 = np.asarray(param.initial_target_pos, dtype=float).reshape(3, 1)
    q_ref_list = []
    for uri in active_uris:
        state = drone_states[uri]
        hx, hy, _ = np.asarray(state.local_origin, dtype=float).reshape(-1)[:3]
        q_s = np.array([[hx], [hy], [param.default_height]]) - target0
        norm = float(np.linalg.norm(q_s))
        if norm > 1e-6:
            q_ref_list.append(q_s / norm)
        else:   # takeoff point on the target: choose a fallback direction
            q_ref_list.append(np.array([[1.0], [0.0], [0.0]]))
        state.desired.radii_to_target = norm
    return q_ref_list

def flight_metadata(param: Param, active_uris, drone_states: dict[str, DroneState], firmware_params: dict,
                    fib_ref_points, min_safe_dist, q_ref_list):
    """What the simulation of this flight needs (saved in <log>_meta.json)."""
    drones = {}
    for iD, uri in enumerate(active_uris):
        st = drone_states[uri]
        drones[str(RIGID_BODY_IDS[uri])] = {
            "uri": uri,
            "guidance_index": iD,                      # iD in controller(): its Fibonacci point
            "home": np.asarray(st.local_origin, dtype=float).reshape(-1)[:3],
            "home_yaw_deg": math.degrees(float(np.asarray(st.attitude).reshape(-1)[2])),
            "fibonacci_point": np.asarray(fib_ref_points[iD], dtype=float).reshape(-1),
            "q_ref0": np.asarray(q_ref_list[iD], dtype=float).reshape(-1),
            "radius0": float(st.desired.radii_to_target),
            "firmware": firmware_params.get(uri, {}),
        }
    return {
        "flight_code": "swarm_geometric_Fibonacci_sim_target",
        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "onboard_controller": ONBOARD_CONTROLLER,
        "mellinger_params_set": MELLINGER_PARAMS,
        "timing_s": {
            "SETPOINT_PERIOD": SETPOINT_PERIOD, "EXT_POS_PERIOD": EXT_POS_PERIOD,
            "TAKEOFF_TIME": TAKEOFF_TIME, "RETURN_TIME": RETURN_TIME, "LAND_TIME": LAND_TIME,
            "ESTIMATOR_SETTLE_TIME": ESTIMATOR_SETTLE_TIME, "TRACKING_TIMEOUT": TRACKING_TIMEOUT,
            "LOG_SAMPLE_PERIOD": LOG_SAMPLE_PERIOD,
            "controller_period_nominal": param.dTi,     # controller(): sleep(param.dTi) per tick
            "target_period_nominal": param.dTi,         # update_virtual_target(): sleep(param.dTi)
            "Tend": param.Tend,
        },
        "guidance": {
            "nD": param.nD, "dTi": param.dTi, "sphere_radius": param.sphere_radius,
            "k_geometric": param.k_geometric, "k_rad": param.k_rad, "k_apf": param.k_apf,
            "min_safe_dist": float(min_safe_dist),
            "q_ref_init": "direction from the target start to (home x, home y, default_height); radius0 = its distance",
        },
        "target": {
            "traj": param.traj, "target_traj_radius": param.target_traj_radius,
            "default_speed": param.default_speed, "default_height": param.default_height,
            "initial_position": np.asarray(param.initial_target_pos, dtype=float).reshape(-1),
        },
        "max_height": MAX_HEIGHT,
        "drones": drones,
    }

# Flight log of the setpoints (TrajectoryLogger.record_setpoint), set by main.py
SETPOINT_LOG = None

def set_setpoint_logger(logger):
    global SETPOINT_LOG
    SETPOINT_LOG = logger

def send_position_setpoint(scf, x, y, z, yaw=0.0, target=None, radius=None, q_ref=None):
    scf.cf.commander.send_position_setpoint(float(x), float(y), float(z), float(yaw))
    if SETPOINT_LOG is not None:
        SETPOINT_LOG.record_setpoint(scf.cf.link_uri, (x, y, z), target, radius, q_ref)

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
    """Return to a point above home at takeoff height. Also runs after an abort."""
    uri = scf.cf.link_uri
    hx, hy, _ = state.local_origin
    return ramp_to(scf, [hx, hy, param.default_height], RETURN_TIME, "return to home", state, position_lock, abort_event,
                   respect_abort=False)

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
    param: Param,
    abort_event: threading.Event,
):
    """
    Runs continuously on each CF (until Tend or an abort).
    """
    cf = scf.cf
    iD = active_uris.index(cf.link_uri)
    end_time = time.time() + param.Tend
    while time.time() < end_time:
        if abort_event.is_set():   # 1st Ctrl+C / tracking lost: hand over to return + land
            return False
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
        send_position_setpoint(scf, goal[0, 0], goal[1, 0], goal[2, 0], 0.0,
                               target=target_pos.ravel(),
                               radius=state.desired.radii_to_target,
                               q_ref=q_hat.ravel())

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