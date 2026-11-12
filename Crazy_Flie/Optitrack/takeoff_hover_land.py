import logging
import math
import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

from natnet import DataDescriptions, DataFrame, NatNetClient

import ctrl_c   # 1st Ctrl+C: return home + land, 2nd: motors off


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
TARGET_HEIGHT = 0.50

# ------------------------------------------------------------
# OptiTrack / Motive
# ------------------------------------------------------------

MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

# Rigid body ID assigned to the Crazyflie in Motive.
RIGID_BODY_ID = 1

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

# How long to hold 0.5 m.
HOVER_TIME = 5.0

# How long to take to reach the target height.
TAKEOFF_TIME = 3.0

# How long to descend.
LAND_TIME = 2.0

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

# Home position captured before flight.
home_position = [0.0, 0.0, 0.0]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.ERROR)


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
# COORDINATE CONVERSION
# ============================================================

def optitrack_to_crazyflie_position(position):
    """
    Convert absolute Motive position to a position relative
    to the captured home position.

    This means:

        home = (0, 0, 0)

    and a point 0.5 m above home becomes:

        (0, 0, 0.5)

    Assumes Motive's Z axis is vertical.
    """

    return [
        position[0] - home_position[0],
        position[1] - home_position[1],
        position[2] - home_position[2],
    ]


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
# CRAZYFLIE LOGGING
# ============================================================

class CrazyflieLogger:
    """
    Logs the Crazyflie's Kalman position.

    This is particularly useful for determining whether the
    Crazyflie estimator is actually receiving the OptiTrack
    position correctly.
    """

    def __init__(self, scf):

        self.scf = scf
        self.log_config = None

    def start(self):

        try:

            self.log_config = self.scf.cf.log.create_log_config(
                name='KalmanPosition',
                period_in_ms=100
            )

            self.log_config.add_variable(
                'kalman.stateX',
                'float'
            )

            self.log_config.add_variable(
                'kalman.stateY',
                'float'
            )

            self.log_config.add_variable(
                'kalman.stateZ',
                'float'
            )

            self.log_config.data_received_cb.add_callback(
                self.data_received
            )

            self.log_config.error_cb.add_callback(
                self.error
            )

            self.log_config.start()

            print("Kalman position logging started.")

        except Exception as e:

            print()
            print("WARNING: Could not start Kalman logging.")
            print(e)
            print()

    def data_received(self, timestamp, data, logconf):

        x = data.get('kalman.stateX', 0.0)
        y = data.get('kalman.stateY', 0.0)
        z = data.get('kalman.stateZ', 0.0)

        print(
            f"Kalman | "
            f"X: {x:+.3f} m | "
            f"Y: {y:+.3f} m | "
            f"Z: {z:+.3f} m"
        )

    def error(self, logconf, msg):

        print(
            f"Kalman logging error: {msg}"
        )

    def stop(self):

        if self.log_config is not None:

            try:
                self.log_config.delete()

            except Exception:
                pass


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
# HOME POSITION
# ============================================================

def capture_home_position():
    """
    Capture the current OptiTrack position as the origin.
    """

    global home_position

    position = get_optitrack_position()

    home_position = list(position)

    print()
    print("Home position captured:")
    print(
        f"  X = {home_position[0]:+.3f} m"
    )
    print(
        f"  Y = {home_position[1]:+.3f} m"
    )
    print(
        f"  Z = {home_position[2]:+.3f} m"
    )
    print()

    return home_position


# ============================================================
# SAFETY CHECKS
# ============================================================

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


# ============================================================
# SEND POSITION SETPOINT
# ============================================================

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

# ============================================================
# TAKEOFF
# ============================================================

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


# ============================================================
# HOVER
# ============================================================

def hover(scf):
    """
    Hold the Crazyflie at TARGET_HEIGHT.
    """

    print()
    print(
        f"Holding {TARGET_HEIGHT:.2f} m "
        f"for {HOVER_TIME:.1f} seconds..."
    )
    print()

    end_time = time.monotonic() + HOVER_TIME

    while time.monotonic() < end_time:

        if abort_event.is_set():
            return False

        if not safety_check(scf):
            return False

        send_position_setpoint(
            scf,
            0.0,
            0.0,
            TARGET_HEIGHT,
            0.0
        )

        time.sleep(SETPOINT_PERIOD)

    print()
    print("Hover complete.")

    return True


# ============================================================
# LAND
# ============================================================

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

    # from the current height (an aborted takeoff is below TARGET_HEIGHT)
    start_z = min(TARGET_HEIGHT, max(0.0, get_optitrack_position()[2]))

    for i in range(steps):

        fraction = 1.0 - ((i + 1) / steps)

        target_z = start_z * fraction

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

    time.sleep(1)
    print("Landing complete.")

# ============================================================
# STOP / DISARM
# ============================================================

def stop_flight(scf):
    """
    Stop position setpoints and disarm.
    """

    print()
    print("Stopping flight...")

    try:
        # Tell Crazyflie that position setpoint stream
        # has ended.
        scf.cf.commander.send_stop_setpoint()
        scf.cf.commander.send_notify_setpoint_stop()

    except Exception as e:

        print(
            f"Could not send stop setpoint: {e}"
        )

    time.sleep(0.5)

    try:
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

    kalman_logger = None
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

            print()
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

                capture_home_position()

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
                # Start Kalman logging
                # ------------------------------------------------

                kalman_logger = CrazyflieLogger(scf)

                kalman_logger.start()

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
                # Arm
                # ------------------------------------------------
                ctrl_c.flying([scf.cf])
                scf.cf.supervisor.send_arming_request(
                    True
                )

                time.sleep(1.0)

                print("Armed.")

                # ------------------------------------------------
                # Takeoff
                # ------------------------------------------------

                # (an abort / 1st Ctrl+C skips to the landing)
                takeoff_ok = takeoff(scf)
                if not takeoff_ok:
                    print(
                        "Takeoff aborted: landing."
                    )

                # ------------------------------------------------
                # Hover
                # ------------------------------------------------

                if takeoff_ok and not hover(scf):
                    print(
                        "Hover aborted: landing."
                    )

                # ------------------------------------------------
                # Land
                # ------------------------------------------------

                land(scf)

                # ------------------------------------------------
                # Stop
                # ------------------------------------------------

                stop_flight(scf)
                ctrl_c.landed()

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

        finally:

            stop_event.set()
            abort_event.set()

            if kalman_logger is not None:

                try:
                    kalman_logger.stop()
                except Exception:
                    pass

            print()
            print("Flight control stopped.")


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