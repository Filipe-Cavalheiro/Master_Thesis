import numpy as np
import math
from dataclasses import dataclass, field
from typing import List

@dataclass
class DroneState():
    def __init__(self, horizon):

        self.pos: np.ndarray = np.array([0,0,0], dtype='float')
        self.position_horizon: np.ndarray = np.array(
            [0.0, 0.0, 0.0] for _ in range(horizon)
        )
        self.velocity: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.acceleration: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.attitude: List[float] = [0.0, 0.0, 0.0]  # roll, pitch, yaw
