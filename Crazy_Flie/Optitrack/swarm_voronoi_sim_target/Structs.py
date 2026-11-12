import numpy as np

# CONSTS
BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [BASE_URI + f"{i}" for i in range(3, 5)]   # 01 -> 04

# Rigid-body ID in Motive for EACH Crazyflie (EDIT to match Motive).
RIGID_BODY_IDS = {uri: int(uri[-2:]) for uri in CANDIDATE_URIS}
ID_TO_URI = {rb_id: uri for uri, rb_id in RIGID_BODY_IDS.items()}

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

HOVER_TIME = 8.0     # s, hover above the takeoff points
TRACKING_TIMEOUT = 0.25
TAKEOFF_HEIGHT = 0.6      # m
TAKEOFF_TIME = 3.0
RETURN_SPEED = 0.3        # m/s, peak speed back to the takeoff points
LAND_TIME = 3.0
LAND_HEIGHT = 0.02        # m, motors stopped at the end of a landing

# Virtual target and Voronoi formation
TARGET_TRAJ = 2             # 1 = still, 2 = horizontal circle around the arena centre
TARGET_TRAJ_RADIUS = 1.6    # m (2.2 with lighthouse: with the formation around it, it left the 5 x 5 m arena)
TARGET_ANGULAR_SPEED = 0.3  # rad/s (0.48 m/s on the circle)
TARGET_HEIGHT = 1.3         # m
SPHERE_RADIUS = 0.4         # m, formation radius
MESH_POINTS = 1001
FORMATION_TIME = 30.0       # s, controllers running while the target moves
VORONOI_PERIOD = 0.1        # s, target and formation update
CVT_PRECONVERGE_ITERS = 30  # CVT iterations on the initial mesh, before arming
APPROACH_SPEED = 0.3        # m/s, peak speed to the initial formation slots
MAX_SETPOINT_SPEED = 0.8    # m/s, rate limit of the setpoint towards the formation goal
GEOFENCE_MARGIN = 0.3       # m, formation goals kept this far from the arena walls/ceiling
MIN_GOAL_HEIGHT = 0.3       # m
MIN_SEPARATION = 0.25       # m, two drones closer than this abort the flight

GOAL_MIN = np.array([ARENA_MIN[0] + GEOFENCE_MARGIN, ARENA_MIN[1] + GEOFENCE_MARGIN, MIN_GOAL_HEIGHT])
GOAL_MAX = ARENA_MAX - GEOFENCE_MARGIN


class DroneState:
    """OptiTrack/flight data of a Crazyflie."""

    def __init__(self):
        self.position: np.ndarray = np.zeros(3)
        self.attitude: np.ndarray = np.zeros(3)       # roll, pitch, yaw (rad)
        self.yaw: float = 0.0
        self.local_origin: np.ndarray = np.zeros(3)   # takeoff point on the floor
        self.land_spot: np.ndarray = np.zeros(3)      # where it lands at the end
        self.setpoint: np.ndarray = np.full(3, np.nan)  # last position setpoint sent
        self.last_update: float = 0.0                 # time.monotonic() of the last OptiTrack frame


class Param:
    """Formation/target parameters used by voronoi_mesh_funcs (same fields as the lighthouse version)."""

    def __init__(self, number_of_drones: int, traj: int = TARGET_TRAJ):
        self.Tend = FORMATION_TIME
        self.dTi = VORONOI_PERIOD

        self.nD = number_of_drones
        self.sphere_radius = SPHERE_RADIUS

        self.target_traj_radius = TARGET_TRAJ_RADIUS
        self.default_speed = TARGET_ANGULAR_SPEED
        self.default_height = TARGET_HEIGHT

        self.ref_target_traj = self._make_traj(traj)
        self.initial_target_pos = self.ref_target_traj["pos"](0)

    def _make_traj(self, traj):
        # The lighthouse trajectories 3 and 4 (z = 5 m, 5 m lobes) do not fit the arena
        if traj == 1:
            return {
                "pos": lambda t: np.array([self.target_traj_radius, 0.0, self.default_height]),
            }

        if traj == 2:
            return {
                "pos": lambda t: np.array([
                    self.target_traj_radius * np.sin(self.default_speed * t),
                    self.target_traj_radius * np.cos(self.default_speed * t),
                    self.default_height
                ])
            }

        raise ValueError("Unknown trajectory")


class Voronoi:
    def __init__(self, param: Param):
        self.mesh_pos = np.array(None)
        self.surf_direction = [1, 0, 0]
        self.v_circle = np.array(None)
        self.v_cone = np.array(None)
        self.v_sphere = np.array(None)
        self.vertices = np.array(None)
        self.faces = np.array(None)
        self.sites = np.array(None)
        self.ran_u2_eq_1 = False


class MeshVertex:
    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = np.array(pos, dtype=float)

        self.dist = np.inf
        self.origin_site = -1
        self.prev = -1

        self.neigh = []
        self.neighW = []
