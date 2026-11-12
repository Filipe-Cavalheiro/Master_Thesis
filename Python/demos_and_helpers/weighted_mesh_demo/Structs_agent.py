import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from Structs_sim import Param

@dataclass
class EstimatedState:        
    target_position: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    target_velocity: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))

@dataclass
class DesiredState:
    # final desired position
    goal: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))

    # next point in the horizon desired values
    position: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    snap: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))
    psi : float = 0.0
    old_psi: float = 0.0


    # radii control associated with the geometric controller of V8
    radii_to_target: np.floating = np.float32(0.0)

class MotionConstraints:
    def __init__(self, param: Param):
        # params for obstacle avoidance/ vel and acc constraints 
        self.obstacle_center = np.array([0, 2.0, 1.0])
        self.obstacle_radius = np.diag([0.02, 0.02, 0.02])
        self.q: np.ndarray = np.zeros([(param.sim.n_degree+1)*3,1])
        self.b_obs: np.ndarray = np.zeros((3 * param.sim.horizon, 1))
        self.A_obs: np.ndarray = np.zeros((3 * param.sim.horizon, (param.sim.n_degree+1)*3))
        self.s: Optional[np.ndarray] = None
        self.lambda_obs: np.ndarray = np.array(0)
        self.lambda_ineq: np.ndarray = np.array(0)
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