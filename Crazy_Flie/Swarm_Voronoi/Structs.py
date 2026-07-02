import numpy as np
import math
from dataclasses import dataclass, field
from typing import List

from cflib.crazyflie.log import LogConfig

# CONSTS
BASE_URI = "radio://0/10/2M/E7E7E7E70"
CANDIDATE_URIS = [
    BASE_URI + f"{i}"
    for i in range(1, 6)   # 01 → 05
]
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

@dataclass
class DroneState:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attitude: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # roll,pitch,yaw
    log_conf: LogConfig = field(init=False)

class Param:
    def __init__(self, number_of_drones: int, traj:int =2):
        # ----------------------------
        # Simulation
        # ----------------------------
        self.Tend = 30
        self.dTi = 0.1

        self.nD = number_of_drones
        self.sphere_radius = 0.6        # (m)

        self.target_traj_radius = 2.2  # (m)
        self.default_speed = 0.5        # Default speed (m/s)
        self.default_height = 1.3       # Default takeoff altitude (m)
        self.deadline_time = 10

        # ----------------------------
        # Trajectory
        # ----------------------------
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
        self.newSite = np.zeros((1, param.nD))
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