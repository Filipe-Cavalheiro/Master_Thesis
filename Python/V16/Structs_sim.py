import numpy as np
from numpy.typing import NDArray
import math
from dataclasses import dataclass

# CONSTS
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

class SimulationConfig:
    def __init__(self):
        # Simulation
        self.Tend = 30 # [seconds]
        self.dTi = 0.01 # [seconds]
        self.Nsim = round(self.Tend / self.dTi) + 1

        self.t = np.arange(
            0,
            self.Tend + self.dTi,
            self.dTi
        )

        self.gravity = 9.81

        # Drones
        self.nD = 12
        self.sphere_radius = 5.0

        # Drop parameters
        self.nD_drop = 4
        self.nD_drop_time = 1000 # -1 to deactivate

        # Quadrotor
        self.drone_mass = 5
        self.J = np.diag([
            2e-2,
            2e-2,
            3e-2
        ])

        self.drag = 0.169
        self.ctf = 0.0135

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

        # Target estimation
        self.ema_alpha = 0.2
        self.noise_std = 0.05

        # constraints
        self.obstacle_avoidance = True 
        self.restricted_vel_acc = True
        self.obstacle_avoidance_dim = 1.00
        self.n_degree = 5
        self.horizon = 5           # horizon
        self.rho_obs = 100.0        # Obstacle avoidance penalty
        self.rho_ineq = 20.0        # max/min vel and acc penalty
        self.max_iter = 100         # max iterations per update
        self.w_goal = 1000.0        # gain to reduce dist between agent and goal
        self.w_smooth = 0.1         # gain to reduce high accelerations
        self.v_max = 3             # max velocity constraint 
        self.a_max = 5             # max acceleration constraint 

        # --- CVT parameters ---
        # Number of simulation iterations between CVT site updates.
        # The underlying mesh points are still updated every iteration.
        # Only the CVT sites (goal positions) are recomputed at this interval
        # using the current mesh.
        self.update_CVT_interval = 100
        self.weights_sigma = 1.2
        
        # --- Measurement model ---
        # options: "estimated" or "ground_truth"
        self.measurement_model = "estimated"
        # 1: position
        # 2: position + velocity
        self.measurement_dimension = 2

        # Formation/distribution
        # options: "fibonacci" or "dijkstra" 
        self.distribution = "dijkstra"

class ControllerConfig:
    def __init__(self):
        # Options: Mellinger or Lee (Lee is not implemented in Python only in matlab (sorry))
        self.selected_controller = "Mellinger"

        self.Mellinger = {
            "kp": np.diag([1, 1, 1]),
            "kv": np.diag([15, 15, 15]),
            "kR": np.diag([8, 8, 8]),
            "kom": np.diag([0.5, 0.5, 0.5]),
        }

        self.Lee = {
            "x": 10,
            "v": 8,
            "i": 10,
            "R": 1.5,
            "W": 0.35,
            "I": 10,
            "y": 0.8,
            "wy": 0.15,
            "yI": 2,
        }

        # Consensus parameters
        self.c1 = 1.5
        self.sigma = 10
        self.c2 = 2
        self.c3 = 2

class TrajectoryConfig:
    def __init__(
        self,
        traj=2,
        radius=2.2,
        speed=0.5,
        height=1.3
    ):

        self.target_traj_radius = radius
        self.default_speed = speed
        self.default_height = height

        self.ref_target_traj = self.make_traj(2)

        self.initial_target_pos = (
            self.ref_target_traj["pos"](0)
        )


    def make_traj(self, traj):
        if traj == 1:
            # Hover
            return {
                "pos": lambda t: np.array([
                    [10],
                    [0],
                    [5]
                ]),

                "vel": lambda t: np.zeros([3,1])
            }


        elif traj == 2:
            # Circle
            vel_gain = 0.25
            radius = 15

            return {
                "pos": lambda t: np.array([
                    [-5 + radius*np.sin(vel_gain*t)],
                    [radius*np.cos(vel_gain*t)],
                    [5]
                ]),

                "vel": lambda t: np.array([
                    [radius*vel_gain*np.cos(vel_gain*t)],
                    [-radius*vel_gain*np.sin(vel_gain*t)],
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
    def __init__(self):
        self.verbose = True

        self.composed = True

        self.radial_hist_plot = True
        self.radial_hist_animation = False

        self.inter_agent_dist  = True

        self.low_level_tracking_error = True

        self.consensus_plot = True 
        self.consensus_animation = False 

        self.velocity_plot = True
        self.velocity_animate = False

        self.acceleration_plot = True
        self.acceleration_animate = False

        self.trajectory_plot = True 
        self.trajectory_animate = False

        self.three_view_plot = True

        self.store_data = True
        self.save_to_mp4 = False
        self.mp4_fps = 30

        self.save_location = "/Results"
        self.trail_length = 100
        self.step = 5

@dataclass
class SimulationData:
    target_true_hist: list
    removed_initial_pos: bool
    colors: np.ndarray
    active_ids: list
    numAgents_history: list

    position_history: NDArray[np.float64]
    velocity_history: NDArray[np.float64]
    acceleration_history: NDArray[np.float64]
    perceived_target_hist: NDArray[np.float64]
    p_d_hist: NDArray[np.float64]
    radial_history: NDArray[np.float64]

class Param:
    def __init__(self):
        self.sim = SimulationConfig()

        self.controller = ControllerConfig()

        self.traj = TrajectoryConfig()

        self.display = GraphicalDisplay()

class Voronoi:
    def __init__(self, param: Param):
        self.mesh_pos = np.array(None)
        self.surf_direction = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)
        self.v_circle = np.array(None)
        self.v_cone = np.array(None)
        self.v_sphere = np.array(None)
        self.vertices = np.empty((0, 3))
        self.faces = np.empty((0, 3), dtype=int)
        self.sites: NDArray[np.int_] | None = None
        self.newSites: NDArray[np.int_] | None = None
        self.ran_u2_eq_1 = False
        self.weights = np.array(None)

class MeshVertex:
    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = np.array(pos, dtype=float)

        self.dist = np.inf
        self.origin_site = -1
        self.origin_site_weight = -1
        
        self.neigh = []
        self.neighW = []