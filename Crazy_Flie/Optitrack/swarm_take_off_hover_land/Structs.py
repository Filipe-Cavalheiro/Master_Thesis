import numpy as np

# CONSTS
BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [BASE_URI + f"{i}" for i in range(1, 5)]   # 01 -> 04

# Rigid-body ID in Motive for EACH Crazyflie (EDIT to match Motive).
RIGID_BODY_IDS = {uri: int(uri[-2:]) for uri in CANDIDATE_URIS}
ID_TO_URI = {rb_id: uri for uri, rb_id in RIGID_BODY_IDS.items()}

# Rigid-body ID in Motive of the filming camera (phone), logged to
# <log>_camera.csv next to the trajectory log, for the video overlay
# (Python/demos_and_helpers/video_overlay/flight_overlay.py); None = not logged.
CAMERA_RIGID_BODY_ID = 0

MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

# Arena (OptiTrack frame, z up, origin at the centre of the floor): the room
# is 6 x 6 x 3 m, the drones only use the central 5 x 5 x 2.5 m. A drone
# measured outside it aborts the flight.
ARENA_MIN = np.array([-2.5, -2.5, 0.0])
ARENA_MAX = np.array([2.5, 2.5, 2.5])

# Onboard controller (stabilizer.controller): None = keep the drone's
# setting, 1 = PID, 2 = Mellinger
ONBOARD_CONTROLLER = 2
# Estimator (stabilizer.estimator) set before the reset: 2 = Kalman (fuses
# the OptiTrack external position)
ONBOARD_ESTIMATOR = 2
MAX_START_YAW_DEG = 10.0  # deg, |yaw| on the ground before arming (Kalman reset assumes 0)
MIN_BATTERY_VOLTAGE = 3.8  # V at rest before arming (1S LiPo)

EXT_POS_PERIOD = 0.01     # 100 Hz
SETPOINT_PERIOD = 0.05    # 20 Hz (takeoff, hover, return, landing)

TRACKING_TIMEOUT = 0.25
TAKEOFF_HEIGHT = 0.6      # m
TAKEOFF_TIME = 3.0
RETURN_SPEED = 0.3        # m/s, peak speed back to the takeoff points
LAND_TIME = 4.0
LAND_HEIGHT = 0.05        # m, motors stopped at the end of a landing


class  DroneState:
    """OptiTrack/flight data of a Crazyflie."""

    def __init__(self):
        self.position: np.ndarray = np.zeros(3)
        self.attitude: np.ndarray = np.zeros(3)       # roll, pitch, yaw (rad)
        self.yaw: float = 0.0
        self.local_origin: np.ndarray = np.zeros(3)   # takeoff point on the floor
        self.land_spot: np.ndarray = np.zeros(3)      # where it lands at the end
        self.last_update: float = 0.0                 # time.monotonic() of the last OptiTrack frame
