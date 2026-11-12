"""
Parameters, state and data of the simulation (as Structs.py of
../swarm_geometric_Fibonacci_sim_target/ and ../8traj/).
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class DesiredState:
    position: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    snap: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    psi: float = 0.0
    old_psi: float = 0.0


class DroneState:
    def __init__(self, param=None):
        self.position: np.ndarray = np.zeros((3, 1))
        self.velocity: np.ndarray = np.zeros((3, 1))
        self.acceleration: np.ndarray = np.zeros([3, 1], dtype=np.float64)
        self.attitude: np.ndarray = np.eye(3, dtype=np.float64).reshape(-1, 1)
        self.ang_vel: np.ndarray = np.zeros([3, 1], dtype=np.float64)
        self.desired: DesiredState = DesiredState()


class SimulationConfig:
    def __init__(self, config_path=Path(__file__).resolve().parent / "drone_models.toml", drone_name="cf2x_P250"):
        # Simulation (main.py without --IRL_log: the Main Controller, the hover)
        self.Tend = 10
        self.dTi = 0.01
        self.Nsim = round(self.Tend / self.dTi) + 1
        self.t = np.arange(0, self.Tend + self.dTi, self.dTi)
        self.gravity = 9.81

        self.nD = 3
        self.real_trajectory_csv = None     # flight log (main.py --IRL_log)

        # Drone parameters from TOML
        with open(config_path, "rb") as f:
            drone = tomllib.load(f)[drone_name]

        self.drone_mass = drone["mass"]
        self.J = np.array(drone["J"], dtype=float)
        self.thrust_min = drone["thrust_min"]
        self.thrust_max = drone["thrust_max"]
        self.acc_coef = drone["acc_coef"]
        self.thrust_dyn_coef = drone["thrust_dyn_coef"]
        self.drag_matrix = np.array(drone["drag_matrix"], dtype=float)
        self.rpy_coef = np.array(drone["rpy_coef"], dtype=float)
        self.rpy_rates_coef = np.array(drone["rpy_rates_coef"], dtype=float)


class HoverConfig:
    """Main Controller of the simulation (no --IRL_log): every drone holds a point above its home."""
    def __init__(self):
        self.height = 0.6           # m, hover height
        self.spacing = 0.5          # m, distance between the homes (a row along x, centered on 0)

    def points(self, nD):
        """(nD, 3) hover points."""
        return np.array([[(k - (nD - 1) / 2) * self.spacing, 0.0, self.height] for k in range(nD)])


class GraphicalDisplay:
    def __init__(self, dTi):
        self.verbose = True

        self.position_plot = True       # x, y, z of every drone + setpoint
        self.top_view_plot = True       # x-y paths
        self.trajectory_plot = True     # 3D paths
        self.trajectory_animate = False # 3D animation

        self.store_data = True          # Results/simulation_results.pkl (python Animation_display.py redraws from it)
        self.save_to_mp4 = False
        self.mp4_fps = 30

        self.save_location = str(Path(__file__).resolve().parent / "Results")
        self.trail_length = 100


@dataclass
class SimulationData:
    position_history: NDArray[np.float64]   # (nD, 3, Nsim)
    p_d_hist: NDArray[np.float64]           # (nD, 3, Nsim) setpoints (NaN: none)
    ids: list                               # rigid body ID of every drone
    phases: dict                            # phase start times [s]
    colors: np.ndarray

    def __post_init__(self):
        assert self.position_history.ndim == 3
        assert self.position_history.shape[1] == 3
        assert self.p_d_hist.shape == self.position_history.shape


@dataclass
class MellingerGains:
    kp: np.ndarray
    kv: np.ndarray
    kR: np.ndarray
    kom: np.ndarray


class ControllerConfig:
    def __init__(self, param):
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


class Param:
    def __init__(self):
        self.sim = SimulationConfig()
        self.controller = ControllerConfig(self)
        self.hover = HoverConfig()
        self.display = GraphicalDisplay(self.sim.dTi)
