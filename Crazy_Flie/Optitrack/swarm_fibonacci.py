"""
Centralized Fibonacci-sphere formation with direct motion of drones to
the desired points (no path planner).
"""
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import List
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from natnet import DataDescriptions, DataFrame, NatNetClient

import ctrl_c   # 1st Ctrl+C: return home + land, 2nd: motors off

# ============================================================
# CONFIGURATION
# ============================================================

BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [BASE_URI + f"{i}" for i in range(1, 5)]   # 01 -> 04

# Rigid-body ID in Motive for EACH Crazyflie (EDIT to match Motive).
RIGID_BODY_IDS = {uri: int(uri[-2:]) for uri in CANDIDATE_URIS}
ID_TO_URI = {rb_id: uri for uri, rb_id in RIGID_BODY_IDS.items()}

MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

TARGET_HEIGHT = 1.00      # takeoff height / bottom of the sphere (m)
MAX_HEIGHT = 2.5          # abort if OptiTrack z exceeds this (m)
DEFAULT_SPEED = 0.8       # max target speed (m/s)

# Fibonacci sphere + moving virtual target
SPHERE_RADIUS = 0.5                               # m
SPHERE_CENTER_Z = TARGET_HEIGHT + SPHERE_RADIUS   # lowest point sits at TARGET_HEIGHT
CIRCLE_RADIUS = 1.0                               # radius of the target's circular path (m)
CIRCLE_TIME = 30.0                                # s
APPROACH_TIME = 4.0                               # s, fly from takeoff point to sphere point
TARGET_PERIOD = 0.01                              # s, virtual target update

EXT_POS_PERIOD = 0.01     # 100 Hz
SETPOINT_PERIOD = 0.05    # 20 Hz

TRACKING_TIMEOUT = 0.25
ESTIMATOR_SETTLE_TIME = 3.0
TAKEOFF_TIME = 3.0
RETURN_TIME = 3.0
LAND_TIME = 4.0

# ============================================================
# GLOBAL STATE
# (must be defined at module level; `global x` at module level does nothing)
# ============================================================

@dataclass
class DroneState:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attitude: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # roll, pitch, yaw (rad)
    local_origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    last_update: float = 0.0

active_uris: List[str] = []          # drones found on the radio
drone_states = {}                    # uri -> DroneState
fib_pos = None                       # (3, P) sphere points, absolute tracking frame
target = DroneState()                # virtual moving target (offset of the sphere centre)

position_lock = threading.Lock()
target_lock = threading.Lock()
stop_event = threading.Event()       # stops background threads
abort_event = threading.Event()      # aborts the flight phases
extpos_threads = []

logging.basicConfig(level=logging.ERROR)

# ============================================================
# NATNET
# ============================================================

def on_data_description(desc: DataDescriptions):
    print("Received NatNet data descriptions.")

def quaternion_to_euler(x, y, z, w):
    """Quaternion (x, y, z, w) -> (roll, pitch, yaw) in radians."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw

def on_data_frame(frame: DataFrame):
    """Store the latest pose of every configured rigid body."""
    for rb in frame.rigid_bodies:
        uri = ID_TO_URI.get(rb.id_num)
        if uri is None or not rb.tracking_valid:
            continue

        mx, my, mz = rb.pos
        qx, qy, qz, qw = rb.rot

        # Motive (Y-up) -> Crazyflie (Z-up): (x, y, z) = (Zm, Xm, Ym).
        roll, pitch, yaw = quaternion_to_euler(qz, qx, qy, qw)

        with position_lock:
            st = drone_states.get(uri)
            if st is None:           # not an active drone
                continue
            st.position = [mz, mx, my]
            st.attitude = [roll, pitch, yaw]
            st.last_update = time.monotonic()

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

def get_position(uri):
    with position_lock:
        return list(drone_states[uri].position)

def get_yaw(uri):
    with position_lock:
        return drone_states[uri].attitude[2]

def tracking_is_valid(uri=None):
    """True if the given drone (or all drones if uri is None) has fresh data."""
    now = time.monotonic()
    with position_lock:
        states = [drone_states[uri]] if uri else list(drone_states.values())
        return bool(states) and all(
            s.last_update > 0.0 and now - s.last_update < TRACKING_TIMEOUT
            for s in states
        )

def wait_for_tracking(timeout=5.0):
    print("Waiting for valid OptiTrack tracking of all drones...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tracking_is_valid():
            print("OptiTrack tracking acquired for all drones.")
            return True
        time.sleep(0.05)
    for uri in active_uris:
        if not tracking_is_valid(uri):
            print(f"ERROR: no fresh tracking for {uri} (rigid body {RIGID_BODY_IDS[uri]})")
    return False

def safety_check(uri):
    if not tracking_is_valid(uri):
        print(f"!!! OPTITRACK TRACKING LOST ({uri}) !!!")
        abort_event.set()
        return False
    if get_position(uri)[2] > MAX_HEIGHT:
        print(f"!!! HEIGHT LIMIT EXCEEDED ({uri}) !!!")
        abort_event.set()
        return False
    return True

# ============================================================
# EXTERNAL POSITION (one thread per drone)
# ============================================================

def _extpos_loop(cf, uri):
    while not stop_event.is_set():
        if tracking_is_valid(uri):
            x, y, z = get_position(uri)
            cf.extpos.send_extpos(x, y, z)
        elif not abort_event.is_set():
            print(f"Stale tracking for {uri}: aborting.")
            abort_event.set()
        time.sleep(EXT_POS_PERIOD)


def start_extpos(scf: SyncCrazyflie):
    t = threading.Thread(target=_extpos_loop,
                         args=(scf.cf, scf.cf.link_uri), daemon=True)
    t.start()
    extpos_threads.append(t)

# ============================================================
# HELPER FUNCS
# ============================================================

def update_virtual_target():
    """
    Moves the sphere centre along a circle. The offset starts at (0, 0, 0)
    so there is no jump when the circle phase begins.
    """
    w = DEFAULT_SPEED / CIRCLE_RADIUS    # angular rate -> linear speed = DEFAULT_SPEED
    start = time.monotonic()

    while not stop_event.is_set():
        t = time.monotonic() - start
        with target_lock:
            target.position[0] = CIRCLE_RADIUS * math.sin(w * t)
            target.position[1] = CIRCLE_RADIUS * (math.cos(w * t) - 1.0)
            target.position[2] = 0.0
        time.sleep(TARGET_PERIOD)

def get_target_offset():
    with target_lock:
        return list(target.position)

def get_fibonacci_points(P, R=1.0, sphere_center=None, verbose=False):
    """
    Generate Fibonacci-lattice points on a sphere (works for any P >= 1;
    for odd P it is identical to the original +/-N index formulation).

    Returns: (3, P) array
    """
    if P < 1:
        raise RuntimeError("Need at least 1 drone")

    if sphere_center is None:
        sphere_center = np.zeros(3)
    else:
        sphere_center = np.asarray(sphere_center, dtype=float).reshape(3)

    phi_golden = (1 + np.sqrt(5)) / 2

    idx = np.arange(P) - (P - 1) / 2.0           # symmetric indices
    theta = np.arccos(2.0 * idx / P)             # polar angle
    phi = 2.0 * np.pi * idx / phi_golden         # azimuth

    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    fibonacci_points = np.vstack([x, y, z]) + sphere_center.reshape(3, 1)

    if verbose:
        print("fibonacci_points:")
        print(fibonacci_points)

    return fibonacci_points

# ============================================================
# PER-DRONE COMMANDS (run via swarm.parallel_safe)
# ============================================================

def capture_home(scf: SyncCrazyflie):
    uri = scf.cf.link_uri
    pos = get_position(uri)
    with position_lock:
        drone_states[uri].local_origin = pos
    yaw_deg = math.degrees(get_yaw(uri))
    print(f"{uri}: home = {[round(p, 3) for p in pos]}, yaw = {yaw_deg:.1f} deg")
    if abs(yaw_deg) > 15.0:
        print(f"WARNING {uri}: the Kalman filter assumes yaw ~ 0 at reset. "
              f"Point the drone along +X of the tracking frame.")


def configure_kalman(scf: SyncCrazyflie):
    cf = scf.cf
    cf.param.set_value('stabilizer.estimator', '2')   # 2 = Kalman
    time.sleep(0.5)
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

def arm(scf: SyncCrazyflie):
    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)


def warning_lights(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)
    time.sleep(2)
    scf.cf.param.set_value('led.bitmask', 0)


def send_position_setpoint(scf, x, y, z, yaw=0.0):
    scf.cf.commander.send_position_setpoint(float(x), float(y), float(z), float(yaw))


def ramp_to(scf, target_pos, duration, label, respect_abort=True):
    """Smoothly move from the current measured position to `target_pos`."""
    uri = scf.cf.link_uri
    start = get_position(uri)
    steps = max(1, int(duration / SETPOINT_PERIOD))

    for i in range(steps):
        if ctrl_c.killed():
            return False
        if respect_abort and abort_event.is_set():
            return False
        if not safety_check(uri):
            return False
        f = (i + 1) / steps
        f = f * f * (3.0 - 2.0 * f)  # smoothstep
        sp = [s + (t - s) * f for s, t in zip(start, target_pos)]
        send_position_setpoint(scf, sp[0], sp[1], sp[2], 0.0)
        time.sleep(SETPOINT_PERIOD)

    print(f"{uri}: {label} complete.")
    return True

def take_off(scf):
    uri = scf.cf.link_uri
    hx, hy, _ = drone_states[uri].local_origin
    return ramp_to(scf, [hx, hy, TARGET_HEIGHT], TAKEOFF_TIME, "takeoff")

def go_to_fibonacci_point(scf):
    """Fly directly from the takeoff point to this drone's sphere point."""
    uri = scf.cf.link_uri
    i = active_uris.index(uri)
    goal = [fib_pos[0, i], fib_pos[1, i], fib_pos[2, i]]
    return ramp_to(scf, goal, APPROACH_TIME, "approach to sphere point")

def move_fibonacci_circle(scf):
    """Hold the sphere formation while its centre follows the virtual target."""
    uri = scf.cf.link_uri
    i = active_uris.index(uri)
    base = fib_pos[:, i]

    start = time.monotonic()
    while time.monotonic() - start < CIRCLE_TIME:
        if abort_event.is_set() or not safety_check(uri):
            return False
        off = get_target_offset()
        send_position_setpoint(scf,
                               base[0] + off[0],
                               base[1] + off[1],
                               base[2] + off[2],
                               0.0)
        time.sleep(SETPOINT_PERIOD)
    return True

def return_origin(scf):
    """Return to a point above home at takeoff height. Also runs after an abort."""
    uri = scf.cf.link_uri
    hx, hy, _ = drone_states[uri].local_origin
    return ramp_to(scf, [hx, hy, TARGET_HEIGHT], RETURN_TIME, "return to home",
                   respect_abort=False)


def land(scf):
    """Always runs (ignores abort) as long as this drone is still tracked."""
    uri = scf.cf.link_uri
    hx, hy, _ = drone_states[uri].local_origin
    return ramp_to(scf, [hx, hy, 0.0], LAND_TIME, "landing", respect_abort=False)


def stop_flight(scf):
    try:
        scf.cf.commander.send_stop_setpoint()
        scf.cf.commander.send_notify_setpoint_stop()
        scf.cf.supervisor.send_arming_request(False)
        print(f"{scf.cf.link_uri}: disarmed.")
    except Exception as e:
        print(f"{scf.cf.link_uri}: could not disarm: {e}")

# ============================================================
# MAIN
# ============================================================

def main():
    global active_uris, fib_pos     # main() rebinds these module-level names

    print("\n==============================================")
    print(" Crazyflie swarm + OptiTrack (Kalman, ext. position)")
    print("==============================================\n")
    print(f"  Candidate URIs : {CANDIDATE_URIS}")
    print(f"  Rigid bodies   : {RIGID_BODY_IDS}")
    print(f"  Motive / local : {MOTIVE_IP} / {LOCAL_IP}\n")

    ctrl_c.install(abort_event)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    # Discover existing drones
    active_uris = discover_active_uris()
    if not active_uris:
        raise RuntimeError("No drones found")

    drone_states.clear()
    drone_states.update({uri: DroneState() for uri in active_uris})

    client = NatNetClient(
        server_ip_address=MOTIVE_IP,
        local_ip_address=LOCAL_IP,
        use_multicast=False,
    )
    client.on_data_description_received_event.handlers.append(on_data_description)
    client.on_data_frame_received_event.handlers.append(on_data_frame)

    target_thread = threading.Thread(target=update_virtual_target, daemon=True)

    with client:
        client.request_modeldef()
        client.run_async()

        if not wait_for_tracking(timeout=5.0):
            print("ABORT: no valid OptiTrack tracking.")
            return

        print("\nConnecting to Crazyflies...")
        with Swarm(active_uris, factory=factory) as swarm:
            print("Swarm connected.\n")
            stop_event.clear()
            abort_event.clear()
            armed = False

            try:
                swarm.parallel_safe(capture_home)

                # Sphere centred over the centroid of the home positions.
                center = [0, 0, SPHERE_CENTER_Z]
                fib_pos = get_fibonacci_points(len(active_uris), SPHERE_RADIUS,
                                               sphere_center=center, verbose=True)

                swarm.parallel_safe(start_extpos)
                swarm.parallel_safe(configure_kalman)

                print(f"Waiting {ESTIMATOR_SETTLE_TIME} s for estimator...")
                time.sleep(ESTIMATOR_SETTLE_TIME)

                if not tracking_is_valid() or abort_event.is_set():
                    print("ERROR: tracking lost during estimator settling.")
                    return

                ctrl_c.flying([scf.cf for scf in swarm._cfs.values()])
                swarm.parallel_safe(arm)
                armed = True
                print("Armed.")

                swarm.parallel_safe(warning_lights)
                swarm.parallel_safe(take_off)
                swarm.parallel_safe(go_to_fibonacci_point)

                # start the moving target only once everyone is on the sphere
                if not abort_event.is_set():
                    target_thread.start()
                    swarm.parallel_safe(move_fibonacci_circle)


            except KeyboardInterrupt:
                print("\nKeyboard interrupt received.")
                abort_event.set()
            except Exception as e:
                print(f"\nERROR DURING FLIGHT: {e}")
                abort_event.set()
            finally:
                # Always (also after an abort / 1st Ctrl+C): return above the
                # local home, land (per drone, as long as it is still tracked), disarm.
                if armed:
                    time.sleep(0.3)   # let aborted phases exit
                    try:
                        try:
                            swarm.parallel_safe(return_origin)
                        except Exception as e:
                            print(f"ERROR DURING RETURN: {e}: landing in place.")
                        swarm.parallel_safe(land)
                        time.sleep(1.0)
                    finally:
                        swarm.parallel_safe(stop_flight)
                        ctrl_c.landed()
                        time.sleep(1.0)

                stop_event.set()
                for t in extpos_threads:
                    t.join(timeout=1.0)
                if target_thread.is_alive():
                    target_thread.join(timeout=1.0)

            if not abort_event.is_set():
                print("\nFlight test completed successfully.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nProgram terminated.")