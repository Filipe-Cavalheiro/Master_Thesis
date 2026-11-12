import numpy as np
from numpy.typing import NDArray
from typing import Optional
import math
from dataclasses import dataclass, field
import tomllib
from pathlib import Path

from plot_real_data import * 

# CONSTS
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

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
    
class SimulationConfig:
    def __init__(self, config_path=Path(__file__).resolve().parent / "drone_models.toml", drone_name="cf2x_P250"):
        # Simulation
        self.Tend = 30
        self.dTi = 0.01
        self.Nsim = round(self.Tend / self.dTi) + 1

        self.t = np.arange(
            0,
            self.Tend + self.dTi,
            self.dTi
        )

        self.gravity = 9.81

        self.compare_with_experimental = False
        self.load_experimental_state = True

        # Drones
        self.nD = 3 
        self.sphere_radius = 2.0

        # Drop parameters
        self.nD_drop = 0
        self.nD_drop_time = 500

        # if a specific test is desired define its path here else the newest one is used
        csv_path = None

        # Load the real world data
        self.real_trajectory_csv = Path(csv_path ) if csv_path else newest_log()

        # Load drone parameters from TOML
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        drone = config[drone_name]

        # Quadrotor parameters
        self.drone_mass = drone["mass"]
        self.J = np.array(drone["J"], dtype=float)

        self.thrust_min = drone["thrust_min"]
        self.thrust_max = drone["thrust_max"]

        self.acc_coef = drone["acc_coef"]
        self.thrust_dyn_coef = drone["thrust_dyn_coef"]

        self.drag_matrix = np.array(
            drone["drag_matrix"], dtype=float
        )

        self.rpy_coef = np.array(
            drone["rpy_coef"], dtype=float
        )

        self.rpy_rates_coef = np.array(
            drone["rpy_rates_coef"], dtype=float
        )


        # Measurements
        # 1: position
        # 2: position + velocity
        self.measurable_states = 1

        # Formation/distribution
        # options: fibonacci, dijkstra 
        self.distribution = "dijkstra"


class TrajectoryConfig:
    def __init__(
        self,
        traj=2,
        radius=2.0,
        speed=0.5,
        height=1.3
    ):

        self.target_traj_radius = radius
        self.default_speed = speed
        self.default_height = height

        traj = 1
        self.ref_target_traj = self.make_traj(traj)

        self.initial_target_pos = (
            self.ref_target_traj["pos"](0)
        )


    def make_traj(self, traj):
        if traj == 1:
            # Hover
            return {
                "pos": lambda t: np.array([
                    [0],
                    [0],
                    [5]
                ]),

                "vel": lambda t: np.zeros([3,1])
            }


        elif traj == 2:
            # circle 
            vel_gain = 0.25

            return {
                "pos": lambda t: np.array([
                    [-5 + 15*np.sin(vel_gain*t)],
                    [15*np.cos(vel_gain*t)],
                    [5]
                ]),

                "vel": lambda t: np.array([
                    [15*vel_gain*np.cos(vel_gain*t)],
                    [-15*vel_gain*np.sin(vel_gain*t)],
                    [0]
                ])
            }


        elif traj == 3:
            # Lissajous
            return {
                "pos": lambda t: np.array([
                    5*np.sin(t),
                    5*np.sin(2*t),
                    5
                ]),

                "vel": lambda t: np.array([
                    5*np.cos(t),
                    10*np.cos(2*t),
                    0
                ])
            }


        elif traj == 4:
            # 3D Lissajous
            return {
                "pos": lambda t: np.array([
                    5*np.sin(t),
                    5*np.sin(2*t),
                    5+0.2*np.cos(2*t)
                ]),

                "vel": lambda t: np.array([
                    5*np.cos(t),
                    10*np.cos(2*t),
                    -0.4*np.sin(2*t)
                ])
            }

        else:
            raise ValueError(
                f"Unknown trajectory {traj}"
            )

class GraphicalDisplay:
    def __init__(self, dTi):
        self.verbose = True
        self.composed = False

        self.radial_hist_plot = True
        self.radial_hist_animation = False

        self.consensus_plot = False
        self.consensus_animation = False

        self.tracking_error_plot = True 
        self.interagent_geodesic_distance_plot = True 

        self.trajectory_plot = True
        self.trajectory_animate = False

        self.store_data = True
        self.save_to_mp4 = False
        self.mp4_fps = 30

        self.save_location = str(Path(__file__).resolve().parent / "Results")
        self.trail_length = 100
        self.step = 5
        

@dataclass
class SimulationData:
    target_true_hist: list
    removed_initial_pos: bool
    colors: np.ndarray
    min_safe_dist: np.float64
    position_history: NDArray[np.float64]
    p_d_hist: NDArray[np.float64]
    radial_history: NDArray[np.float64]
    inter_agent_dist_hist: NDArray[np.float64]
    agent_pairs: list 

    def __post_init__(self):
        assert self.position_history.ndim == 3
        assert self.position_history.shape[1] == 3

        assert self.p_d_hist.ndim == 3
        assert self.p_d_hist.shape[1] == 3

class Param:
    def __init__(
        self
    ):
        self.sim = SimulationConfig()

        self.controller = ControllerConfig(self)

        self.traj = TrajectoryConfig()

        self.display = GraphicalDisplay(
            self.sim.dTi
        )


        # Disturbance
        self.x_delta = np.array([
            0.5,
            0.8,
            -1.0
        ])

        self.R_delta = np.array([
            0.2,
            1.0,
            -0.1
        ])

        self.k_rad = 0.3
        self.k_geometric = 5
        self.k_apf = 5

        # Target estimation
        self.ema_alpha = 0.2
        self.noise_std = 0.05

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

@dataclass
class MellingerGains:
    kp: np.ndarray
    kv: np.ndarray
    kR: np.ndarray
    kom: np.ndarray

class ControllerConfig:
    def __init__(self, param: Param):
        self.selected_controller = "Mellinger"
        m, J = param.sim.drone_mass, param.sim.J
        wn_p, zeta = 6.0, 0.9      # position loop bandwidth, rad/s
        wn_R = 40.0                # attitude loop, ~7x faster than position

        self.Mellinger = MellingerGains(
            kp=np.diag([m * wn_p**2] * 3),
            kv=np.diag([2 * zeta * wn_p * m] * 3),
            kR=J * wn_R**2,
            kom=J * 2 * zeta * wn_R,
        )

class Voronoi:
    def __init__(self, param: Param):
        self.mesh_pos = np.array(None)
        self.surf_direction = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)
        self.v_circle = np.array(None)
        self.v_cone = np.array(None)
        self.v_sphere = np.array(None)
        self.vertices = np.empty((0, 3))
        self.faces = np.empty((0, 3), dtype=int)
        self.sites = np.array((1, param.sim.nD))
        self.newSite = np.zeros((1, param.sim.nD))
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