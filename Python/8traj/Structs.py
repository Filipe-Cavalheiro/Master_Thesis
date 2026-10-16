import numpy as np
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
    b1: np.ndarray = field(default_factory=lambda: np.array([1.0,0.0,0.0]).reshape(-1,1))
    b1_dot: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    b1_2dot: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    radii_to_target: np.floating = np.float32(0.0)

class DroneState:
    def __init__(self, param):
        self.position: np.ndarray = np.zeros((3,1))
        self.velocity: np.ndarray =  np.zeros((3,1))
        self.acceleration: np.ndarray =  np.zeros([3,1], dtype=np.float64)
        self.attitude: np.ndarray =  np.eye(3, dtype=np.float64).reshape(-1,1)
        self.ang_vel: np.ndarray =  np.zeros([3,1], dtype=np.float64)

        self.desired: DesiredState = DesiredState()
    
class SimulationConfig:
    def __init__(self, config_path="8traj/drone_models.toml", drone_name="cf2x_P250"):

        # Simulation settings
        self.Tend = 30
        self.dTi = 0.01
        self.Nsim = round(self.Tend / self.dTi) + 1

        self.t = np.arange(
            0,
            self.Tend + self.dTi,
            self.dTi
        )

        self.gravity = 9.81

        # Number of drones
        self.nD = 1

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
        
    def print_parameters(self):
        """Print all simulation and drone parameters."""

        print("\n" + "=" * 50)
        print("          SIMULATION PARAMETERS")
        print("=" * 50)

        # Simulation settings
        print("\n--- Simulation Settings ---")
        print(f"Simulation end time (Tend):       {self.Tend:.3f} s")
        print(f"Time step (dTi):                  {self.dTi:.3f} s")
        print(f"Number of simulation steps (Nsim): {self.Nsim}")
        print(f"Number of drones (nD):            {self.nD}")
        print(f"Gravity:                          {self.gravity:.3f} m/s²")

        # Drone parameters
        print("\n--- Drone Parameters ---")
        print(f"Mass:                             {self.drone_mass}")
        print(f"Inertia matrix (J):\n{self.J}")

        print(f"\nMinimum thrust:                   {self.thrust_min}")
        print(f"Maximum thrust:                   {self.thrust_max}")

        print(f"\nAcceleration coefficient:         {self.acc_coef}")
        print(f"Thrust dynamic coefficient:       {self.thrust_dyn_coef}")

        print(f"\nDrag matrix:\n{self.drag_matrix}")
        print(f"\nRPY coefficients:\n{self.rpy_coef}")
        print(f"\nRPY rates coefficients:\n{self.rpy_rates_coef}")


class TrajectoryConfig:
    def __init__(
        self,
        traj=2,
        radius=2.0,
        speed=0.8,
        height=1.3
    ):

        self.target_traj_radius = radius
        self.default_speed = speed
        self.default_height = height

        traj = 3
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

            return {
                "pos": lambda t: np.array([
                    [-5 + 15*np.sin(self.default_speed*t)],
                    [15*np.cos(self.default_speed*t)],
                    [5]
                ]),

                "vel": lambda t: np.array([
                    [15*self.default_speed*np.cos(self.default_speed*t)],
                    [-15*self.default_speed*np.sin(self.default_speed*t)],
                    [0]
                ])
            }


        elif traj == 3:
            # Lissajous
            return {
                "pos": lambda t: np.array([
                    [1.5*np.sin(self.default_speed*t)],
                    [np.sin(2*self.default_speed*t)],
                    [1.0]
                ]),

                "vel": lambda t: np.array([
                    [1.5*self.default_speed*np.cos(self.default_speed*t)],
                    [2*self.default_speed*np.cos(2*self.default_speed*t)],
                    [0]
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

        self.real_trajectory_plot = False 
        
        self.real_sim_comparison_plot = True
        self.real_sim_comparison_animate = True

        self.trajectory_plot = False
        self.trajectory_animate = False

        self.store_data = True
        self.save_to_mp4 = False
        self.mp4_fps = 30

        self.save_location = "Results"
        self.trail_length = 100
        self.step = 5
        

@dataclass
class SimulationData:
    agent_true_hist: list
    removed_initial_pos: bool
    colors: np.ndarray

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