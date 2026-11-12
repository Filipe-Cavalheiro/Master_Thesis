import os
import sys
from pathlib import Path

import numpy as np

# The arena experiment (trajectory 5) is shared with the Python simulation
_REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.append(str(_REPO / "Python" / "flight_sims" / "V17_Lisbon_aeroport" / "Python"))
import experiment_arena as arena

# CONSTS
BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [BASE_URI + f"{i}" for i in range(1, 5)]   # 01 -> 04

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
SETPOINT_PERIOD = 0.05   # 20 Hz (takeoff, control, return, landing)

HOVER_TIME = 8.0     # s, hover above the takeoff points
TRACKING_TIMEOUT = 0.25
TAKEOFF_HEIGHT = 0.6      # m
TAKEOFF_TIME = 3.0
RETURN_SPEED = 0.3        # m/s, peak speed back to the takeoff points
LAND_TIME = 3.0
LAND_HEIGHT = 0.02        # m, motors stopped at the end of a landing

# Virtual target and Voronoi formation
TARGET_TRAJ = 2             # 1 = still, 2 = horizontal circle around the arena centre, 5 = Lisbon
                            # trajectory 5 window (Python/experiment_arena.py, wall at the KMZ angle)
TARGET_TRAJ = int(os.environ.get("CF_TARGET_TRAJ", TARGET_TRAJ))   # flight_sim.py --dry-run --traj
# The target, chosen per flight (python main.py --target sim | real, which
# sets CF_TARGET; flight_sim.py sets it from the run folder name):
#   real: a Crazyflie (rigid-body ID REAL_TARGET_DRONE_ID) flies the target
#         path with full-state setpoints and the formation follows its
#         OptiTrack position and velocity
#   sim:  virtual target, the formation follows the path itself
REAL_TARGET_DRONE_ID = 4
TARGET_MODE = os.environ.get("CF_TARGET", "sim")
if TARGET_MODE not in ("sim", "real"):
    raise ValueError(f"CF_TARGET must be 'sim' or 'real', not {TARGET_MODE!r}")
TARGET_DRONE_ID = REAL_TARGET_DRONE_ID if TARGET_MODE == "real" else None
TARGET_URI = ID_TO_URI.get(TARGET_DRONE_ID) if TARGET_DRONE_ID is not None else None
TARGET_RADIUS_XY = 0.3      # m, MPC envelope around the target drone (sphere radius; cylinder: horizontal radius, half-height MPC_RADIUS_Z)
TARGET_TRAJ_RADIUS = 1.6    # m (2.2 with lighthouse: with the formation around it, it left the 5 x 5 m arena)
TARGET_ANGULAR_SPEED = 0.3  # rad/s (0.48 m/s on the circle)
TARGET_HEIGHT = 1.3         # m
SPHERE_RADIUS = 0.8         # m, formation radius
MESH_POINTS = 1001
FORMATION_TIME = 30.0       # s, controllers running while the target moves
if TARGET_TRAJ == 5:
    # the sphere radius, path and duration of the Python simulation (arena_traj = 5)
    SPHERE_RADIUS = arena.TRAJ5_RADIUS
    FORMATION_TIME = arena.ArenaTrajectory5().t_stop + arena.TRAJ5_HOLD
VORONOI_PERIOD = 0.1        # s, target and formation update
CVT_PRECONVERGE_ITERS = 30  # CVT iterations on the initial mesh, before arming
APPROACH_SPEED = 0.3        # m/s, peak speed to the initial formation slots
GEOFENCE_MARGIN = 0.3       # m, formation goals kept this far from the arena walls/ceiling
MIN_GOAL_HEIGHT = 0.3       # m
MIN_SEPARATION = 0.25       # m, two drones closer than this abort the flight

GOAL_MIN = np.array([ARENA_MIN[0] + GEOFENCE_MARGIN, ARENA_MIN[1] + GEOFENCE_MARGIN, MIN_GOAL_HEIGHT])
GOAL_MAX = ARENA_MAX - GEOFENCE_MARGIN


def clip_to_arena(pts):
    """
    Points (m, 3) inside the geofence and, with trajectory 5, not behind the
    wall (as ArenaAirspace.project of the Python simulation; the red border
    is outside the arena).
    """
    pts = np.array(pts, dtype=float, copy=True)
    if TARGET_TRAJ == 5:
        n = arena.TRAJ5_N_IN
        u = (pts[:, :2] - arena.ArenaTrajectory5().wall_point) @ n
        pts[:, :2] += np.maximum(-u, 0.0)[:, None] * n
    return np.clip(pts, GOAL_MIN, GOAL_MAX)

# MPC (Python/obstacle_avoidance_funcs.py, AdmmMPC): Bernstein polynomial
# horizon to the formation goal, solved with ADMM, with the other drones'
# horizons, the arena geofence and the velocity/acceleration norms as
# projection constraints. Same weights as the simulation, limits of the arena
# experiment (Python/experiment_arena.py).
MPC_PERIOD = 0.1            # s, replanning period (the setpoints are sampled from the plan every SETPOINT_PERIOD)
MPC_HORIZON_TIME = 1.0      # s, horizon duration
MPC_HORIZON = 5             # samples of the horizon
MPC_N_DEGREE = 5            # Bernstein polynomial degree
MPC_ITERS = 15              # ADMM iterations per replanning
MPC_RHO_OBS = 5000.0        # obstacle (other drones, geofence) penalty (100 in the simulation: the
                            # 0.4 m formation stacks the goals, and with 100 the goal cost kept the
                            # drones stacked; offline, 5000 separates them for 3-5 drones)
MPC_RHO_INEQ = 20.0         # vel/acc limit penalty
MPC_W_GOAL = 1000.0
MPC_W_SMOOTH = 0.1
MPC_W_VEL = 300.0           # goal velocity tracking (target velocity)
MPC_GOAL_REF_TAU = 0.5      # s, exponential approach to the goal over the horizon
MPC_V_MAX = 0.8             # m/s (0.5 in the arena experiment, with a 0.3 m/s target; this target flies 0.48 m/s)
MPC_A_MAX = 1.0             # m/s^2
if TARGET_TRAJ == 5:
    MPC_V_MAX = 3.0 * arena.TRAJ5_SCALE     # m/s, Lisbon 3 m/s scaled (0.48), as the simulation
    MPC_A_MAX = 5.0 * arena.TRAJ5_SCALE     # m/s^2, Lisbon 5 m/s^2 scaled (0.8)
# Safety envelope around the other drones (and the target drone): "sphere"
# of radius MPC_RADIUS_XY (the method of the paper, default), or the legacy
# downwash "cylinder" (radius MPC_RADIUS_XY, half-height 1.2 m) with which
# the 26 Oct 2026 flights were flown
MPC_ENVELOPE = "sphere"
MPC_RADIUS_XY = 0.3         # m, inter-drone radius (sphere radius / cylinder radius)
MPC_RADIUS_Z = 1.2 if MPC_ENVELOPE == "cylinder" else MPC_RADIUS_XY   # m, cylinder half-height (sphere: = radius)
MPC_WARM_MULTIPLIERS = False  # carry the ADMM multipliers between replannings (they wind up on real drones)
MPC_REPLAN_RESET_DIST = 0.2   # m, a drone this far from its plan is replanned from the measured state
SLOT_SETTLE_TIME = 5.0      # s, MPC run offline (target still) to get the approach slots
SLOT_MIN_DH = 0.2           # m, min horizontal distance between two slots less than MPC_RADIUS_Z apart vertically
SLOT_STACK_TOL = 0.05       # m, slots up to this much inside MPC_RADIUS_Z vertically are not stacked (the MPC
                            # parks stacked drones on the cylinder edge, 1.2 m apart)
PLAN_TIMEOUT = 0.5          # s, a plan older than this aborts the flight
VEL_FILTER_ALPHA = 0.3      # EMA of the OptiTrack finite-difference velocity
# Setpoint sent to the onboard controller: "full_state" (position, velocity
# and acceleration of the plan, needs Mellinger) or "position"
SETPOINT_MODE = "full_state"

class DroneState:
    """OptiTrack/flight data of a Crazyflie."""

    def __init__(self):
        self.position: np.ndarray = np.zeros(3)
        self.velocity: np.ndarray = np.zeros(3)       # OptiTrack finite differences, EMA filtered
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

        # Wall of the formation (circle -> cone -> sphere as the target
        # crosses it): normal and a point of it. Trajectories 1 and 2: x = 0
        self.wall_normal = np.array([1.0, 0.0, 0.0])
        self.wall_point = np.zeros(3)

        self.ref_target_traj = self._make_traj(traj)
        self.initial_target_pos = self.ref_target_traj["pos"](0)

    def _make_traj(self, traj):
        if traj == 1:
            return {
                "pos": lambda t: np.array([self.target_traj_radius, 0.0, self.default_height]),
                "vel": lambda t: np.zeros(3),
            }

        if traj == 2:
            return {
                "pos": lambda t: np.array([
                    self.target_traj_radius * np.sin(self.default_speed * t),
                    self.target_traj_radius * np.cos(self.default_speed * t),
                    self.default_height
                ]),
                "vel": lambda t: np.array([
                     self.target_traj_radius * self.default_speed * np.cos(self.default_speed * t),
                    -self.target_traj_radius * self.default_speed * np.sin(self.default_speed * t),
                    0.0
                ])
            }

        if traj == 5:
            # Lisbon trajectory 5 scaled to the arena, the wall at the KMZ
            # angle; same path as the Python simulation (arena_traj = 5)
            tr = arena.ArenaTrajectory5()
            self.wall_normal = np.array([*arena.TRAJ5_N_IN, 0.0])
            self.wall_point = np.array([*tr.wall_point, 0.0])
            return {
                "pos": lambda t: tr.ref_target_traj["pos"](t).ravel(),
                "vel": lambda t: tr.ref_target_traj["vel"](t).ravel(),
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
