import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import List

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from natnet import DataDescriptions, DataFrame, NatNetClient

import ctrl_c   # 1st Ctrl+C: return home + land, 2nd: motors off

# ============================================================
# CONFIGURATION
# ============================================================

URIS = [
    'radio://0/10/2M/E7E7E7E701',
    'radio://0/10/2M/E7E7E7E702',
    'radio://0/10/2M/E7E7E7E704',
    # Same radio => same channel for every URI in the swarm.
]

# Rigid-body ID in Motive for EACH Crazyflie (EDIT to match Motive).
RIGID_BODY_IDS = {
    'radio://0/10/2M/E7E7E7E701': 1,
    'radio://0/10/2M/E7E7E7E702': 2,
    'radio://0/10/2M/E7E7E7E704': 4,
}
ID_TO_URI = {rb_id: uri for uri, rb_id in RIGID_BODY_IDS.items()}

MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

TARGET_HEIGHT = 1.00      # height of the lowest drone (m)
LAYER_SPACING = 0.50      # extra height per drone index (vertical separation)
MAX_HEIGHT = 2.5          # abort if OptiTrack z exceeds this (m)
DEFAULT_SPEED = 0.8       # max trajectory speed (m/s)

BOX_LIMIT = 1.0           # max X/Y excursion from home during the figure 8 (m)
FIG8_A = 0.8              # x amplitude (clamped to BOX_LIMIT)
FIG8_B = 0.5              # y amplitude (clamped to BOX_LIMIT)
FIG8_TIME = 12.0          # s
FIG8_FADE_IN = 2.0        # s, amplitude ramps up so there is no jump

EXT_POS_PERIOD = 0.01     # 100 Hz
SETPOINT_PERIOD = 0.05    # 20 Hz

TRACKING_TIMEOUT = 0.25
ESTIMATOR_SETTLE_TIME = 3.0
TAKEOFF_TIME = 3.0
RETURN_TIME = 3.0
LAND_TIME = 4.0

# ============================================================
# GLOBAL STATE
# ============================================================

@dataclass
class DroneState:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attitude: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # roll, pitch, yaw (rad)
    local_origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    last_update: float = 0.0

drone_states = {uri: DroneState() for uri in URIS}
position_lock = threading.Lock()
stop_event = threading.Event()    # stops background threads
abort_event = threading.Event()   # aborts the flight phases
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
        # This is a cyclic permutation (right-handed), so the quaternion's
        # vector part must be permuted the same way before taking yaw.
        roll, pitch, yaw = quaternion_to_euler(qz, qx, qy, qw)

        with position_lock:
            st = drone_states[uri]
            st.position = [mz, mx, my]
            st.attitude = [roll, pitch, yaw]
            st.last_update = time.monotonic()

# ============================================================
# TRACKING HELPERS
# ============================================================

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
        return all(s.last_update > 0.0 and now - s.last_update < TRACKING_TIMEOUT
                   for s in states)


def wait_for_tracking(timeout=5.0):
    print("Waiting for valid OptiTrack tracking of all drones...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tracking_is_valid():
            print("OptiTrack tracking acquired for all drones.")
            return True
        time.sleep(0.05)
    for uri in URIS:
        if not tracking_is_valid(uri):
            print(f"ERROR: no fresh tracking for {uri} (rigid body {RIGID_BODY_IDS[uri]})")
    return False


def flight_height(uri):
    return TARGET_HEIGHT + LAYER_SPACING * URIS.index(uri)


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


def light_check(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)
    time.sleep(2)
    scf.cf.param.set_value('led.bitmask', 0)


def send_position_setpoint(scf, x, y, z, yaw=0.0):
    scf.cf.commander.send_position_setpoint(x, y, z, yaw)


def ramp_to(scf, target, duration, label, respect_abort=True):
    """Smoothly move from the current measured position to `target`."""
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
        sp = [s + (t - s) * f for s, t in zip(start, target)]
        send_position_setpoint(scf, sp[0], sp[1], sp[2], 0.0)
        time.sleep(SETPOINT_PERIOD)

    print(f"{uri}: {label} complete.")
    return True


def take_off(scf):
    uri = scf.cf.link_uri
    hx, hy, _ = drone_states[uri].local_origin
    return ramp_to(scf, [hx, hy, flight_height(uri)], TAKEOFF_TIME, "takeoff")


def move_8_traj(scf):
    """Figure 8 around the drone's own home, on its own altitude layer."""
    uri = scf.cf.link_uri
    idx = URIS.index(uri)
    h = flight_height(uri)

    A = min(FIG8_A, BOX_LIMIT)
    B = min(FIG8_B, BOX_LIMIT)
    w = DEFAULT_SPEED / max(A, 2.0 * B)      # keeps peak speed <= DEFAULT_SPEED
    sign = 1.0 if idx % 2 == 0 else -1.0     # mirror alternate drones

    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        if t >= FIG8_TIME or abort_event.is_set() or not safety_check(uri):
            break
        k = min(1.0, t / FIG8_FADE_IN)
        x = sign * k * A * math.sin(w * t)
        y = k * B * math.sin(2.0 * w * t)
        send_position_setpoint(scf, x, y, h, 0.0)
        time.sleep(SETPOINT_PERIOD)


def return_origin(scf):
    """Return to a point above home at flight height. Also runs after an abort."""
    uri = scf.cf.link_uri
    hx, hy, _ = drone_states[uri].local_origin
    return ramp_to(scf, [hx, hy, flight_height(uri)], RETURN_TIME, "return to home",
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
    print("\n==============================================")
    print(" Crazyflie swarm + OptiTrack (Kalman, ext. position)")
    print("==============================================\n")
    print(f"  URIs          : {URIS}")
    print(f"  Rigid bodies  : {RIGID_BODY_IDS}")
    print(f"  Motive / local: {MOTIVE_IP} / {LOCAL_IP}\n")

    ctrl_c.install(abort_event)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    client = NatNetClient(
        server_ip_address=MOTIVE_IP,
        local_ip_address=LOCAL_IP,
        use_multicast=False,
    )
    client.on_data_description_received_event.handlers.append(on_data_description)
    client.on_data_frame_received_event.handlers.append(on_data_frame)

    with client:
        client.request_modeldef()
        client.run_async()

        if not wait_for_tracking(timeout=5.0):
            print("ABORT: no valid OptiTrack tracking.")
            return

        print("\nConnecting to Crazyflies...")
        with Swarm(URIS, factory=factory) as swarm:
            print("Swarm connected.\n")
            stop_event.clear()
            abort_event.clear()
            armed = False

            try:
                swarm.parallel_safe(capture_home)
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

                swarm.parallel_safe(light_check)
                swarm.parallel_safe(take_off)
                swarm.parallel_safe(move_8_traj)

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

            if not abort_event.is_set():
                print("\nFlight test completed successfully.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nProgram terminated.")