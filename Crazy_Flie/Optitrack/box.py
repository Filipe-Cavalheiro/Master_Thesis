import logging
import math
import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.positioning.motion_commander import MotionCommander

from natnet import DataDescriptions, DataFrame, NatNetClient

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

# Maximum allowed X/Y-axis displacement before reversing direction.
BOX_LIMIT = 1.0

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
BOX_TIME = 30.0

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

def move_box_limit(scf):
    """
    Continuously moves the Crazyflie back and forth between
    predefined X-axis boundaries.

    Args:
        scf (SyncCrazyflie):
            Active Crazyflie connection object.

    Behavior:
        - Takes off to DEFAULT_HEIGHT.
        - Starts moving forward.
        - Reverses direction when the estimated X position
          exceeds BOX_LIMIT or goes below -BOX_LIMIT.
        - Runs indefinitely until externally interrupted.
    """
    max_vel = 0.5

    body_x_cmd = max_vel
    body_y_cmd = max_vel

    end_time = time.monotonic() + BOX_TIME

    while time.monotonic() < end_time:

        position_estimate = get_optitrack_position()

        # X boundary
        if position_estimate[0] > BOX_LIMIT:
            body_x_cmd = -max_vel
        elif position_estimate[0] < -BOX_LIMIT:
            body_x_cmd = max_vel

        # Y boundary
        if position_estimate[1] > BOX_LIMIT:
            body_y_cmd = -max_vel
        elif position_estimate[1] < -BOX_LIMIT:
            body_y_cmd = max_vel

        # Send velocity command
        scf.cf.commander.send_velocity_world_setpoint(
            body_x_cmd,
            body_y_cmd,
            0,
            0
        )

        print(
            "position:",
            position_estimate[0],
            position_estimate[1],
            position_estimate[2]
        )

        time.sleep(SETPOINT_PERIOD)

    # Stop horizontal motion, but DON'T land
    scf.cf.commander.send_velocity_world_setpoint(
        0,
        0,
        0,
        0
    )

# RETURN HOME
def return_origin(scf):
    """
    go to [0,0, current height]
    """

    print()
    print("Returning to origin")
    print()

    steps = int(
        LAND_TIME / SETPOINT_PERIOD
    )

    for _ in range(steps):
        target_z = TARGET_HEIGHT

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
                # Arm
                # ------------------------------------------------
                scf.cf.supervisor.send_arming_request(
                    True
                )

                time.sleep(1.0)

                print("Armed.")

                # ------------------------------------------------
                # Main 
                # ------------------------------------------------
                takeoff(scf)
                move_box_limit(scf)
                return_origin(scf)
                land(scf)

                time.sleep(2)
                stop_flight(scf)
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