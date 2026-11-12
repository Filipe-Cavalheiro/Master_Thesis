import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional

from cflib.crazyflie.log import LogConfig

# CONSTS
BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [BASE_URI + f"{i}" for i in range(1, 5)]   # 01 -> 04
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

# Rigid-body ID in Motive for EACH Crazyflie (EDIT to match Motive).
RIGID_BODY_IDS = {uri: int(uri[-2:]) for uri in CANDIDATE_URIS}
ID_TO_URI = {rb_id: uri for uri, rb_id in RIGID_BODY_IDS.items()}

MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

MAX_HEIGHT = 2.5          # abort if OptiTrack z exceeds this (m)
DEFAULT_SPEED = 0.8       # max target speed (m/s)

# Fibonacci sphere + moving virtual target
CIRCLE_RADIUS = 1.0                               # radius of the target's circular path (m)
CIRCLE_TIME = 30.0                                # s
APPROACH_TIME = 4.0                               # s, fly from takeoff point to sphere point
TARGET_PERIOD = 0.01                              # s, virtual target update

EXT_POS_PERIOD = 0.01     # 100 Hz
SETPOINT_PERIOD = 0.05    # 20 Hz

LOG_SAMPLE_PERIOD = 0.02  # s, trajectory log (50 Hz)

# Onboard controller: Mellinger (stabilizer.controller = 2), with the position
# gains of 8traj.py (kp = diag(1.1448), kv = diag(0.34344)): the ones of the
# Crazyflie plant of the simulation (Python/flight_sims/<name>/Structs.py ControllerConfig).
ONBOARD_CONTROLLER = 2
MELLINGER_PARAMS = {
    "ctrlMel.mass": 0.032,
    "ctrlMel.kp_xy": 1.1448, "ctrlMel.kp_z": 1.1448,
    "ctrlMel.kd_xy": 0.34344, "ctrlMel.kd_z": 0.34344,
}
# Firmware parameters read back after the setup and saved in <log>_meta.json
# (the values the drone actually flew with; a name the firmware lacks is saved as null)
LOGGED_FIRMWARE_PARAMS = [
    "stabilizer.controller", "stabilizer.estimator",
    "ctrlMel.mass", "ctrlMel.massThrust",
    "ctrlMel.kp_xy", "ctrlMel.kd_xy", "ctrlMel.ki_xy", "ctrlMel.i_range_xy",
    "ctrlMel.kp_z", "ctrlMel.kd_z", "ctrlMel.ki_z", "ctrlMel.i_range_z",
    "ctrlMel.kR_xy", "ctrlMel.kR_z", "ctrlMel.kw_xy", "ctrlMel.kw_z",
    "ctrlMel.ki_m_xy", "ctrlMel.ki_m_z", "ctrlMel.kd_omega_rp",
    "ctrlMel.i_range_m_xy", "ctrlMel.i_range_m_z",
]

TRACKING_TIMEOUT = 0.25
ESTIMATOR_SETTLE_TIME = 3.0
TAKEOFF_TIME = 3.0
RETURN_TIME = 3.0
LAND_TIME = 4.0

@dataclass
class EstimatedState:        
    target_position: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    target_velocity: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))

@dataclass
class DesiredState:
    position: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    snap: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    psi : float = 0.0
    old_psi: float = 0.0
    b1: np.ndarray = field(default_factory=lambda: np.array([1.0,0.0,0.0]).reshape(-1,1))
    b1_dot: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    b1_2dot: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    radii_to_target: np.floating = np.float32(0.0)

class MotionConstraints:
    def __init__(self, param):
        # params for obstacle avoidance/ vel and acc constraints 
        self.obstacle_center = np.array([0, 2.0, 1.0])
        self.obstacle_radius = np.diag([1.5, 1.5, 1.5])

        self.q: np.ndarray = np.zeros([(param.n_degree+1)*3,1])
        self.b_obs: np.ndarray = np.zeros((3 * param.horizon, 1))
        self.A_obs: np.ndarray = np.zeros((3 * param.horizon, (param.n_degree+1)*3))
        self.s: Optional[np.ndarray] = None
        self.lambda_obs: Optional[np.ndarray] = None
        self.lambda_ineq: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None

class DroneState:
    def __init__(self, param):
        self.position: np.ndarray = np.zeros((3,1))
        self.velocity: np.ndarray =  np.zeros((3,1))
        self.acceleration: np.ndarray =  np.zeros([3,1], dtype=np.float64)
        self.attitude: np.ndarray =  np.eye(3, dtype=np.float64).reshape(-1,1)
        self.ang_vel: np.ndarray =  np.zeros([3,1], dtype=np.float64)
        self.ei: np.ndarray =  np.zeros([3,1], dtype=np.float64)
        self.eI: np.ndarray =  np.zeros([3,1], dtype=np.float64)
        self.radii_to_target: np.floating = np.float32(0.0)

        self.estimated: EstimatedState = EstimatedState()
        self.desired: DesiredState = DesiredState()
        self.constraints: MotionConstraints = MotionConstraints(param)

        self.local_origin: np.ndarray = np.zeros((3,1))
        self.last_update: float = 0.0

class Param:
    def __init__(self, number_of_drones: int, traj:int =2):
        # ----------------------------
        # Simulation
        # ----------------------------
        self.Tend = 30
        self.dTi = 0.1

        self.nD = number_of_drones
        self.sphere_radius = 0.8        # (m)

        self.target_traj_radius = 1.5  # (m)
        self.default_speed = 0.5        # Default speed (m/s)
        self.default_height = 1.3       # Default takeoff altitude (m)
        self.deadline_time = 10

        # Fibonacci controller gains
        self.k_rad = 0.3
        self.k_geometric = 5
        self.k_apf = 1

        # constraints
        self.n_degree = 5
        self.horizon = 20       # horizon
        self.obs_horizon = 5       # horizon
        self.rho_obs = 100.0      # Obstacle avoidance penalty
        self.rho_ineq = 20.0      # max/min vel and acc penalty
        self.max_iter = 100     # max iterations per update
        self.w_goal = 1000.0    # gain to reduce dist between agent and goal
        self.w_smooth = 0.1     # gain to reduce high accelerations
        self.v_max = 10          # max velocity constraint 
        self.a_max = 50          # max acceleration constraint 

        # Trajectory
        self.traj = traj
        self.ref_target_traj = self._make_traj(traj)
        self.initial_target_pos = self.ref_target_traj["pos"](0)
        

    # ----------------------------
    # Trajectory generator
    # ----------------------------
    def _make_traj(self, traj):
        if traj == 1:
            return {
                "pos": lambda t: np.array([0, 0, 5]),
                "vel": lambda t: np.array([0, 0, 0]),
            }

        if traj == 2:
            return {
                "pos": lambda t: np.array([
                     [self.target_traj_radius * np.sin(self.default_speed * t)],
                     [self.target_traj_radius * np.cos(self.default_speed * t)],
                     [self.default_height]
                ])
            }

        if traj == 3:
            return {
                "pos": lambda t: np.array([
                    5 * np.sin(t),
                    5 * np.sin(2 * t),
                    5
                ]),
                "vel": lambda t: np.array([
                    5 * np.cos(t),
                    10 * np.cos(2 * t),
                    0
                ]),
            }

        if traj == 4:
            return {
                "pos": lambda t: np.array([
                    5 * np.sin(t),
                    5 * np.sin(2 * t),
                    5 + 0.2 * np.cos(2 * t),
                ]),
                "vel": lambda t: np.array([
                    5 * np.cos(t),
                    10 * np.cos(2 * t),
                    -0.4 * np.sin(2 * t),
                ]),
            }

        raise ValueError("Unknown trajectory")