import numpy as np
import math
from dataclasses import dataclass, field
from typing import List

from cflib.crazyflie.log import LogConfig

# CONSTS
BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [
    BASE_URI + f"{i}"
    for i in range(5, 6)   # 01 → 05
]
GRAVITY = 9.81
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

@dataclass
class DroneState:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attitude: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # roll,pitch,yaw
    log_conf: LogConfig = field(init=False)

class Param:
    def __init__(self, number_of_drones: int, traj:int = 2):
        # ----------------------------
        # Simulation
        # ----------------------------
        self.Tend = 30
        self.dTi = 0.1

        self.nD = number_of_drones
        self.sphere_radius = 0.4        # (m)

        # ----------------------------
        # Trajectory
        # ----------------------------
        self.target_traj_radius = 2.0  # (m)
        self.default_speed = 0.5        # Default speed (m/s)
        self.default_height = 1.0       # Default takeoff altitude (m)
        self.deadline_time = 10

        self.ref_target_traj = self._make_traj(traj = 2)
        self.initial_target_pos = self.ref_target_traj["pos"](0)

        # ----------------------------
        # Drones
        # ----------------------------
        self.H = 40             # planning horizon
        self.n_degree = 10      # Bernstein polynomials degree
        
        self.rho_obs = 100                   # Obstacle avoidance penalty
        self.rho_ineq = 20.0                 # max/min vel and acc penalty
        self.max_iter = 1000                 # max iterations per update

        self.w_goal = 1000.0                 # gain to reduce dist between agent and goal
        self.w_smooth = 0.1                  # gain to reduce high accelerations

        self.v_max = 100                     # max velocity constraint 
        self.a_max = 100 + GRAVITY           # max acceleration constraint 

        # ----------------------------
        # Obstacle
        # ----------------------------
        self.obstacle_center = np.array([2,0.0,self.default_height])
        self.obstacle_radius = np.diag([0.5,0.5,1.5])

    # ----------------------------
    # Trajectory generator
    # ----------------------------
    def _make_traj(self, traj):
        if traj == 1:
            return {
                "pos": lambda t: np.array([1.0, 1.0, self.default_height])
            }

        if traj == 2:
            return {
                "pos": lambda t: np.array([
                     self.target_traj_radius * np.sin(self.default_speed * t),
                     self.target_traj_radius * np.cos(self.default_speed * t),
                     self.default_height
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