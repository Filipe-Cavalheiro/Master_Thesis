import numpy as np

from Structs_agent import DroneState
from Structs_sim import Param

def update_target_measurement(
    target: DroneState,
    chaser: DroneState,
    ema_estimated,
    idx,
    connection_laplacian,
    param: Param,
):
    """
    Estimate target position using an exponential moving average and
    a consensus update.

    Depending on `param.sim.measurement_model`, the target state is:
      - copied directly from the ground truth, or
      - estimated from noisy measurements using EMA and consensus.

    Parameters
    ----------
    target : object
        Must have attribute `position` (3,1).
    ema_estimated : ndarray
        Array of shape (3, N) containing EMA estimates.
    idx : int
        Column index corresponding to this agent.
    connection_laplacian : ndarray
        Graph Laplacian matrix (N x N).
    param : object
        Must have attributes:
            noise_std
            ema_alpha
            dTi
    Updates
    -------
    estimated_target_position : ndarray
        Estimated 3D target position for agent idx.
    ema_estimated : ndarray
        Updated EMA estimates.
    """
    if param.sim.measurement_model == "ground_truth":
        chaser.estimated.target_position  = target.position
        chaser.estimated.target_velocity = target.velocity

    elif param.sim.measurement_model == "estimated":
        # Noisy target measurement
        noisy_target_pos = target.position + (param.sim.noise_std * np.random.randn(3)).reshape(-1,1)
        noisy_target_vel = target.velocity + (param.sim.noise_std * np.random.randn(3)).reshape(-1,1)
        
        # Exponential Moving Average (EMA)
        ema_estimated[:, idx:idx+1] = np.concatenate(
            [(1.0 - param.sim.ema_alpha) * ema_estimated[0:3, idx:idx+1]
            + param.sim.ema_alpha * noisy_target_pos,
            (1.0 - param.sim.ema_alpha) * ema_estimated[3:6, idx:idx+1]
            + param.sim.ema_alpha * noisy_target_vel]
        )

        # Consensus update
        estimated_target_variables = (
            ema_estimated[:, idx:idx+1]
            - param.sim.dTi
            * ema_estimated @ connection_laplacian[:, idx:idx+1]
        )

        chaser.estimated.target_position = estimated_target_variables[0:3]
        chaser.estimated.target_velocity = estimated_target_variables[3:6]
        
    else:
        raise ValueError(
            f"Unknown value for param.sim.measurement_model: "
            f"{param.sim.measurement_model!r}"
        )
    return ema_estimated