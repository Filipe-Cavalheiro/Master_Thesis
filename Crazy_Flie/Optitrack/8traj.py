import math
import threading
import time
from datetime import datetime
from pathlib import Path
import csv
import atexit
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig

from natnet import DataDescriptions, DataFrame, NatNetClient

import ctrl_c   # 1st Ctrl+C: return home + land, 2nd: motors off

from camera_logger import CameraLogger

# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Crazyflie
# ------------------------------------------------------------

URI = uri_helper.uri_from_env(
    default='radio://0/10/2M/E7E7E7E701'
)
    
# Desired height ABOVE the OptiTrack home position.
TARGET_HEIGHT = 1.00
DEFAULT_SPEED = 0.8    # Default speed (m/s)

# Figure 8 (Main Controller): x = TRAJ_A sin(w t), y = TRAJ_B sin(2 w t),
# w = DEFAULT_SPEED rad/s. Read by Python/flight_sims/8traj/main.py.
TRAJ_A = 1.5           # m
TRAJ_B = 1.0           # m
# Soft start / stop of the figure 8: the phase rate goes linearly 0 -> w over
# the first TRAJ_EASE_TIME s and w -> 0 over the last ones, so the drone starts
# from and stops at rest on the curve (0 = the old abrupt start / stop at ~1.5-2 m/s).
TRAJ_EASE_TIME = 4.0   # s

# Maximum allowed X/Y-axis displacement before reversing direction.
BOX_LIMIT = 1.0

# ------------------------------------------------------------
# OptiTrack / Motive
# ------------------------------------------------------------

MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

# Rigid body ID assigned to the Crazyflie in Motive.
RIGID_BODY_ID = 1

# Rigid-body ID in Motive of the filming camera (phone), logged to
# <log>_camera.csv for the video overlay; None = not logged.
CAMERA_RIGID_BODY_ID = 0

# ------------------------------------------------------------
# Logs
# ------------------------------------------------------------

# External position update rate.
SAMPLING_RATE = 0.01       # 100 Hz

# ------------------------------------------------------------
# Control
# ------------------------------------------------------------

# External position update rate.
EXT_POS_PERIOD = 0.01       # 100 Hz

# Position setpoint rate.
SETPOINT_PERIOD = 0.05      # 20 Hz

# ------------------------------------------------------------
# Safety
# ------------------------------------------------------------

# Abort if OptiTrack has not provided a valid measurement
# within this amount of time.
TRACKING_TIMEOUT = 0.25

# Time allowed for estimator to initialize.
ESTIMATOR_SETTLE_TIME = 3.0

MAIN_CONTROL_TIME = 30.0

# How long to take to reach the target height.
TAKEOFF_TIME = 3.0

# How long to descend.
LAND_TIME = 2.0

# Return to the origin after the figure 8: smoothstep ramp from the current
# position to (0, 0, TARGET_HEIGHT), peak speed RETURN_SPEED, at least LAND_TIME.
# (The old step to the origin, up to ~1.6 m of position error at once, flipped
# the drone on 15 and 20 Oct.)
RETURN_SPEED = 0.5     # m/s

# Safety limit. If estimator reports something wildly wrong,
# abort the flight.
MAX_ESTIMATED_HEIGHT = 2.0

# ============================================================
# GLOBAL STATE
# ============================================================

# Latest raw OptiTrack position.
optitrack_position = [0.0, 0.0, 0.0]

# Latest yaw from OptiTrack.
optitrack_yaw = 0.0

# Time of last valid OptiTrack update.
last_tracking_time = 0.0

# True once a valid rigid-body measurement has been received.
tracking_event = threading.Event()

# Protects OptiTrack shared state.
position_lock = threading.Lock()

# Tells background threads to stop.
stop_event = threading.Event()

# Indicates that the flight should be aborted.
abort_event = threading.Event()

# ============================================================
# LOGGING
# ============================================================
# Extract state (position, velocity, attitude, angular velocity)

class TrajectoryLogger:
    """
    Logs, at a fixed rate, into a single CSV:
      - OptiTrack position (ground truth, same frame sent to the EKF)
      - Kalman estimate: position, velocity, attitude (deg), gyro (deg/s)
    """

    COLUMNS = [
        "time_s", "flight_state", "id",
        "opti_x", "opti_y", "opti_z",
        "est_x", "est_y", "est_z",
        "est_vx", "est_vy", "est_vz",
        "roll_deg", "pitch_deg", "yaw_deg",
        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    ]

    # Crazyflie log variable -> CSV column
    VAR_TO_COL = {
        "stateEstimate.x": "est_x",
        "stateEstimate.y": "est_y",
        "stateEstimate.z": "est_z",
        "stateEstimate.vx": "est_vx",
        "stateEstimate.vy": "est_vy",
        "stateEstimate.vz": "est_vz",
        "stateEstimate.roll": "roll_deg",
        "stateEstimate.pitch": "pitch_deg",
        "stateEstimate.yaw": "yaw_deg",
        "gyro.x": "gyro_x_dps",
        "gyro.y": "gyro_y_dps",
        "gyro.z": "gyro_z_dps",
    }

    def __init__(self, position, position_lock,
                 output_dir=None, run_name=None, sample_period=SAMPLING_RATE):
        self.position = position
        self.position_lock = position_lock
        self.sample_period = sample_period

        # One folder per run, in the repository Results folder:
        #   Results/<run_name>_trajectory_<timestamp>/trajectory_<timestamp>.csv
        # (the files derived from the log, _sim.csv, _camera.csv, ..., go next to it)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "Results"
        run_name = run_name or Path(__file__).stem
        self.run_dir = output_dir / f"{run_name}_trajectory_{timestamp}"
        self.output_path = self.run_dir / f"trajectory_{timestamp}.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.stop_evt = threading.Event()
        self.thread = None
        self.start_time = None
        self.current_state = ""

        # Latest values received from the Crazyflie
        self.latest = {}
        self.latest_lock = threading.Lock()

        self.log_configs = []
        self.cf = None
        self.file = None
        self.writer = None

    # ---------------- Crazyflie log blocks ----------------
    def _make_log_configs(self):
        # A log packet holds max 26 bytes -> 6 floats (24 B) per block
        pos_vel = LogConfig(name="KalmanPosVel", period_in_ms=20)
        for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                  "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
            pos_vel.add_variable(v, "float")

        att_gyro = LogConfig(name="AttGyro", period_in_ms=20)
        for v in ("stateEstimate.roll", "stateEstimate.pitch",
                  "stateEstimate.yaw",
                  "gyro.x", "gyro.y", "gyro.z"):
            att_gyro.add_variable(v, "float")

        return [pos_vel, att_gyro]

    def _on_log_data(self, timestamp, data, logconf):
        with self.latest_lock:
            for var, value in data.items():
                col = self.VAR_TO_COL.get(var)
                if col:
                    self.latest[col] = value

    # ---------------- Control ----------------
    def start(self, cf):
        """Start logging. Call after the Crazyflie is connected."""
        self.cf = cf
        self.start_time = time.monotonic()

        self.file = self.output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.COLUMNS)
        self.writer.writeheader()

        for lc in self._make_log_configs():
            cf.log.add_config(lc)
            lc.data_received_cb.add_callback(self._on_log_data)
            lc.start()
            self.log_configs.append(lc)

        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"Started logging at {1/SAMPLING_RATE} Hz.")

    def mark_state(self, state_name):
        self.current_state = state_name
        print(f"[Log] State: {state_name}")

    def stop(self):
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

        for lc in self.log_configs:
            try:
                lc.stop()
                lc.delete()
            except Exception:
                pass
        self.log_configs.clear()

        if self.file is not None:
            self.file.close()
            self.file = None

        print(f"Log saved to: {self.output_path}")

    # ---------------- Sampling thread ----------------
    def _run(self):
        next_sample = time.monotonic()
        rows_since_flush = 0

        while not self.stop_evt.is_set():
            with self.position_lock:
                ox, oy, oz = self.position
            with self.latest_lock:
                est = dict(self.latest)

            row = {
                "time_s": time.monotonic() - self.start_time,
                "flight_state": self.current_state,
                "id": RIGID_BODY_ID,
                "opti_x": ox, "opti_y": oy, "opti_z": oz,
            }
            row.update(est)   # missing keys are written as empty
            self.writer.writerow(row)

            rows_since_flush += 1
            if rows_since_flush >= 50:      # flush about once per second
                self.file.flush()
                rows_since_flush = 0

            next_sample += self.sample_period
            delay = next_sample - time.monotonic()
            if delay > 0:
                self.stop_evt.wait(delay)
            else:
                next_sample = time.monotonic()

# ============================================================
# NATNET CALLBACKS
# ============================================================

def on_data_description(desc: DataDescriptions):
    """
    Called when Motive sends the NatNet model description.
    """

    print("Received NatNet data descriptions.")


def quaternion_to_yaw(x, y, z, w):
    """
    Convert quaternion (x, y, z, w) to yaw in radians.
    """

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(sin_yaw, cos_yaw)

def on_data_frame(frame: DataFrame):
    """
    NatNet callback.

    Finds the configured Crazyflie rigid body and stores
    its latest position and orientation.
    """

    global last_tracking_time
    global optitrack_position
    global optitrack_yaw

    rigid_body = None

    for rb in frame.rigid_bodies:

        if rb.id_num == RIGID_BODY_ID:
            rigid_body = rb
            break

    if rigid_body is None:
        return

    if not rigid_body.tracking_valid:
        return

    x, y, z = rigid_body.pos

    qx, qy, qz, qw = rigid_body.rot

    yaw = quaternion_to_yaw(
        qx,
        qy,
        qz,
        qw
    )

    with position_lock:

        optitrack_position[0] = z
        optitrack_position[1] = x
        optitrack_position[2] = y

        optitrack_yaw = yaw

        last_tracking_time = time.monotonic()

    tracking_event.set()


# ============================================================
# OPTITRACK HELPERS
# ============================================================

def get_optitrack_position():
    """
    Return a copy of the latest OptiTrack position.
    """

    with position_lock:
        return list(optitrack_position)


def get_optitrack_yaw():
    """
    Return latest OptiTrack yaw.
    """

    with position_lock:
        return optitrack_yaw


def tracking_is_valid():
    """
    Return True if OptiTrack has updated recently.
    """

    with position_lock:

        if last_tracking_time == 0.0:
            return False

        age = time.monotonic() - last_tracking_time

    return age < TRACKING_TIMEOUT


def wait_for_tracking(timeout=5.0):
    """
    Wait until a valid OptiTrack rigid-body measurement
    is received.
    """

    print("Waiting for valid OptiTrack tracking...")

    if not tracking_event.wait(timeout):

        print("ERROR: No OptiTrack data received.")

        return False

    if not tracking_is_valid():

        print("ERROR: OptiTrack data is already stale.")

        return False

    print("OptiTrack tracking acquired.")

    return True

# ============================================================
# EXTERNAL POSITION THREAD
# ============================================================

def send_optitrack_position(scf):
    """
    Continuously send the OptiTrack position to the
    Crazyflie's external position interface.

    The position is converted to a home-relative frame.
    """

    cf = scf.cf

    print("External-position thread started.")

    while not stop_event.is_set():

        if tracking_is_valid():

            raw_position = get_optitrack_position()

            cf.extpos.send_extpos(
                raw_position[0],
                raw_position[1],
                raw_position[2]
            )

        else:

            # Do not feed stale tracking data.
            abort_event.set()

        time.sleep(EXT_POS_PERIOD)

    print("External-position thread stopped.")

# ============================================================
# ESTIMATOR CONFIGURATION
# ============================================================

def configure_kalman(scf):
    """
    Configure the Crazyflie to use the Kalman estimator.
    """

    print("Configuring Kalman estimator...")

    # stabilizer.estimator:
    #
    # 1 = complementary
    # 2 = Kalman
    #
    scf.cf.param.set_value(
        'stabilizer.estimator',
        '2'
    )

    time.sleep(1.0)

    print("Kalman estimator selected.")

# ============================================================
# CRAZYFLIE COMMANDS
# ============================================================

# SAFETY CHECKS
def safety_check(scf):
    """
    Perform safety checks during flight.
    """

    if not tracking_is_valid():

        print()
        print("!!! OPTITRACK TRACKING LOST !!!")
        print()

        abort_event.set()

        return False

    return True

# SEND POSITION SETPOINT
def send_position_setpoint(scf, x, y, z, yaw=0.0):
    """
    Send one position setpoint.
    """

    scf.cf.commander.send_position_setpoint(
        x,
        y,
        z,
        yaw
    )

# TAKEOFF
def takeoff(scf):
    """
    Gradually increase the position setpoint from 0 to
    TARGET_HEIGHT.
    """

    print()
    print("Starting takeoff...")
    print()

    steps = int(
        TAKEOFF_TIME / SETPOINT_PERIOD
    )

    for i in range(steps):

        if abort_event.is_set():
            return False

        if not safety_check(scf):
            return False

        fraction = (i + 1) / steps

        target_z = TARGET_HEIGHT * fraction

        send_position_setpoint(
            scf,
            0.0,
            0.0,
            target_z,
            0.0
        )

        print(
            f"Takeoff target Z: "
            f"{target_z:.3f} m"
        )

        time.sleep(SETPOINT_PERIOD)

    print()
    print("Takeoff complete.")

    return True

def figure8_phase(t):
    """
    Phase of the figure 8 at t s into the Main Controller, with the soft start / stop:
    rate 0 -> DEFAULT_SPEED over TRAJ_EASE_TIME, constant, then -> 0 at MAIN_CONTROL_TIME.
    (Same function in Python/flight_sims/8traj/main.py.)
    """
    w, T = DEFAULT_SPEED, MAIN_CONTROL_TIME
    E = min(TRAJ_EASE_TIME, T / 2)
    if E <= 0:
        return w * t
    if t < E:
        return w * t * t / (2 * E)
    if t < T - E:
        return w * (t - E / 2)
    tau = min(t, T) - (T - E)
    return w * (T - 1.5 * E + tau - tau * tau / (2 * E))


def move_8_traj(scf):
    A, B = TRAJ_A, TRAJ_B
    start_time = time.monotonic()

    while time.monotonic() - start_time < MAIN_CONTROL_TIME:
        if abort_event.is_set() or not safety_check(scf):
            return False
        t = time.monotonic() - start_time
        phi = figure8_phase(t)
        x = A * math.sin(phi)
        y = B * math.sin(2 * phi)
        send_position_setpoint(scf, x, y, TARGET_HEIGHT, 0.0)
        time.sleep(SETPOINT_PERIOD)
    return True

# RETURN HOME
def return_origin(scf):
    """
    Smooth return to [0, 0, TARGET_HEIGHT]: smoothstep ramp from the current
    OptiTrack position, duration max(1.5 * distance / RETURN_SPEED, LAND_TIME)
    (smoothstep peak speed = 1.5 x mean speed).
    """

    print()
    print("Returning to origin")
    print()

    start = np.array(get_optitrack_position(), dtype=float)
    goal = np.array([0.0, 0.0, TARGET_HEIGHT])
    distance = float(np.linalg.norm(goal - start))
    ramp_time = max(1.5 * distance / RETURN_SPEED, LAND_TIME)

    steps = int(
        ramp_time / SETPOINT_PERIOD
    )
    print(f"Return: {distance:.2f} m in {ramp_time:.1f} s")

    for i in range(steps):
        fraction = (i + 1) / steps
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        target = start + (goal - start) * fraction

        if ctrl_c.killed():
            break

        # During landing we still want to know whether
        # OptiTrack is functioning.
        if not tracking_is_valid():

            print("OptiTrack lost during landing.")

            break

        send_position_setpoint(
            scf,
            target[0],
            target[1],
            target[2],
            0.0
        )

        time.sleep(SETPOINT_PERIOD)

    # Hold the ground position briefly
    # rather than immediately stopping the setpoint stream.
    send_position_setpoint(
        scf,
        0.0,
        0.0,
        0.0,
        0.0
    )
    time.sleep(SETPOINT_PERIOD)

    print("On origin [X/Y] complete.")

# LAND
def land(scf):
    """
    Gradually reduce the position setpoint to zero.
    """

    print()
    print("Landing...")
    print()

    steps = int(
        LAND_TIME / SETPOINT_PERIOD
    )

    for i in range(steps):

        fraction = 1.0 - ((i + 1) / steps)

        target_z = TARGET_HEIGHT * fraction

        if ctrl_c.killed():
            break

        # During landing we still want to know whether
        # OptiTrack is functioning.
        if not tracking_is_valid():

            print("OptiTrack lost during landing.")

            break

        send_position_setpoint(
            scf,
            0.0,
            0.0,
            target_z,
            0.0
        )

        print(
            f"Landing target Z: "
            f"{target_z:.3f} m"
        )

        time.sleep(SETPOINT_PERIOD)
    print("Landing complete.")

# STOP / DISARM
def stop_flight(scf):
    """
    Stop position setpoints and disarm.
    """

    print()
    print("Stopping flight...")

    try:
        scf.cf.commander.send_stop_setpoint()
        scf.cf.commander.send_notify_setpoint_stop()
        scf.cf.supervisor.send_arming_request(False)
        print("Disarmed.")

    except Exception as e:

        print(
            f"Could not disarm: {e}"
        )

# ============================================================
# MAIN
# ============================================================
def main():

    global home_position

    print()
    print("==============================================")
    print(" Crazyflie + OptiTrack")
    print(" Kalman External Position Test")
    print("==============================================")
    print()

    print("Configuration:")
    print(
        f"  Crazyflie URI : {URI}"
    )
    print(
        f"  Motive IP     : {MOTIVE_IP}"
    )
    print(
        f"  Local IP      : {LOCAL_IP}"
    )
    print(
        f"  Rigid Body ID : {RIGID_BODY_ID}"
    )
    print(
        f"  Target height : {TARGET_HEIGHT:.2f} m"
    )
    print()

    # --------------------------------------------------------
    # Initialize Crazyflie drivers
    # --------------------------------------------------------

    ctrl_c.install(abort_event)
    cflib.crtp.init_drivers()

    # --------------------------------------------------------
    # Create NatNet client
    # --------------------------------------------------------

    print("Connecting to Motive...")

    client = NatNetClient(
        server_ip_address=MOTIVE_IP,
        local_ip_address=LOCAL_IP,
        use_multicast=False,
    )

    client.on_data_description_received_event.handlers.append(
        on_data_description
    )

    client.on_data_frame_received_event.handlers.append(
        on_data_frame
    )

    # Trajectory log (its run folder now, the Crazyflie log blocks once connected)
    trajectory_logger = TrajectoryLogger(
        position=optitrack_position,
        position_lock=position_lock,
        sample_period=SAMPLING_RATE,
    )

    # Pose of the filming camera (phone), from the same NatNet frames, from now on:
    # the phone gestures before the take off are recorded too (video sync)
    cam_log = None
    if CAMERA_RIGID_BODY_ID is not None:
        cam_log = CameraLogger(trajectory_logger.output_path, CAMERA_RIGID_BODY_ID)
        client.on_data_frame_received_event.handlers.append(cam_log.on_frame)
        atexit.register(cam_log.close)

    scf = None
    # --------------------------------------------------------
    # Start NatNet
    # --------------------------------------------------------
    with client:

        print("Starting NatNet...")

        client.request_modeldef()

        client.run_async()

        # ----------------------------------------------------
        # Wait for tracking
        # ----------------------------------------------------
        if not wait_for_tracking(timeout=5.0):
            print("ABORT: No valid OptiTrack tracking.")
            return

        # ----------------------------------------------------
        # Connect Crazyflie
        # ----------------------------------------------------
        print()
        print("Connecting to Crazyflie...")
        try:
            with SyncCrazyflie(
                URI,
                cf=Crazyflie(
                    rw_cache='./cache'
                )
            ) as scf:
                print("Crazyflie connected.")
                print()

                # ------------------------------------------------
                # Configure Kalman estimator
                # ------------------------------------------------

                configure_kalman(scf)

                # ------------------------------------------------
                # Capture home position
                # ------------------------------------------------

                if not tracking_is_valid():

                    print(
                        "ERROR: Tracking lost before "
                        "home position capture."
                    )

                    return

                # ------------------------------------------------
                # Start external position feed
                # ------------------------------------------------

                stop_event.clear()
                abort_event.clear()

                extpos_thread = threading.Thread(
                    target=send_optitrack_position,
                    args=(scf,),
                    daemon=True
                )

                extpos_thread.start()

                # ------------------------------------------------
                # Give estimator time to converge
                # ------------------------------------------------

                print()
                print(
                    "Waiting for Kalman estimator "
                    "to initialize..."
                )

                time.sleep(
                    ESTIMATOR_SETTLE_TIME
                )

                if not tracking_is_valid():

                    print()
                    print(
                        "ERROR: OptiTrack tracking "
                        "was lost."
                    )

                    abort_event.set()

                    return

                print()
                print(
                    "Estimator initialization complete."
                )

                # ------------------------------------------------
                # Final tracking check
                # ------------------------------------------------

                if not tracking_is_valid():

                    print()
                    print(
                        "ERROR: OptiTrack tracking "
                        "is not valid."
                    )

                    return

                # ------------------------------------------------
                # Configure Mellinger controller
                # ------------------------------------------------
                print("Setting Mellinger controller...")

                scf.cf.param.set_value(
                    "stabilizer.controller",
                    "2"
                )

                scf.cf.param.set_value("ctrlMel.mass", str(0.032))

                # Position proportional gains: kp = diag(1.1448, 1.1448, 1.1448)
                scf.cf.param.set_value("ctrlMel.kp_xy", "1.1448")
                scf.cf.param.set_value("ctrlMel.kp_z", "1.1448")

                # Velocity feedback gains: kv = diag(0.34344, 0.34344, 0.34344)
                # In the stock firmware these correspond to position D gains.
                scf.cf.param.set_value("ctrlMel.kd_xy", "0.34344")
                scf.cf.param.set_value("ctrlMel.kd_z", "0.34344")

                time.sleep(0.2)

                # ------------------------------------------------
                # Arm
                # ------------------------------------------------
                ctrl_c.flying([scf.cf])
                scf.cf.supervisor.send_arming_request(
                    True
                )

                time.sleep(1.0)

                print("Armed.")

                # --------------------------------------------------------
                # Logs
                # --------------------------------------------------------
                trajectory_logger.start(scf.cf)
                if cam_log is not None:
                    cam_log.mark_flight_start(trajectory_logger.start_time)

                # ------------------------------------------------
                # Main 
                # ------------------------------------------------
                trajectory_logger.mark_state("Take off")
                takeoff(scf)
                trajectory_logger.mark_state("Main Controller")
                move_8_traj(scf)
                trajectory_logger.mark_state("Return to origin")
                return_origin(scf)
                trajectory_logger.mark_state("Land")
                land(scf)
                trajectory_logger.stop()

                time.sleep(2)
                stop_flight(scf)
                ctrl_c.landed()
                time.sleep(2)

                # ------------------------------------------------
                # Stop external position thread
                # ------------------------------------------------
                stop_event.set()

                if extpos_thread.is_alive():
                    extpos_thread.join(timeout=1.0)

                print()
                print("Flight test completed successfully.")

        except KeyboardInterrupt:

            print()
            print("Keyboard interrupt received.")

            if scf is not None:

                try:
                    stop_flight(scf)
                except Exception:
                    pass

        except Exception as e:

            print()
            print("==============================================")
            print("ERROR DURING CRAZYFLIE OPERATION")
            print("==============================================")
            print()
            print(e)
            print()

            if scf is not None:

                try:
                    stop_flight(scf)
                except Exception:
                    pass

# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Program interrupted by user.")

    finally:

        print()
        print("Program terminated.")