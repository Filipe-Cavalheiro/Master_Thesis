
import numpy as np
import pandas as pd
from pathlib import Path
from Structs import Param

# Import these from your own project:
# from your_module import Param, DroneState

def skew(x):
    """
    Obtain skew-symmetric matrix from vector.

    Supports:
      - n = 3: so(3)
      - n = 1: so(2)

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    X : ndarray
        Skew-symmetric matrix.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if n == 3:
        X = np.array([
            [0,    -x[2],  x[1]],
            [x[2],  0,   -x[0]],
            [-x[1], x[0], 0]
        ])

    elif n == 1:
        X = np.array([
            [0,    -x[0]],
            [x[0],  0]
        ])

    else:
        raise ValueError(
            "SKEW function not implemented for input dimensions other than "
            "1 or 3 (i.e., so(2) and so(3))."
        )

    return X

def unskew(M):
    """
    Extracts the vector from a 3x3 skew-symmetric matrix.
    """
    return np.array([
        [M[2, 1]],
        [M[0, 2]],
        [M[1, 0]]
    ])

def rot_integrate(R, om, dT):
    """
    Integrate rotation matrix using angular velocity.
    
    Parameters:
        R : ndarray, shape (3,3)
            Initial rotation matrix
        om : array-like, shape (3,)
            Angular velocity vector
        dT : float
            Time step
            
    Returns:
        Rp : ndarray, shape (3,3)
            Updated rotation matrix
    """
    # om is constant over the step, so the 10 sub-step exponentials collapse
    # into a single one: R @ expm(dT*skew(om)), evaluated in closed form
    # (Rodrigues formula) instead of the generic scipy expm.
    om = np.asarray(om, dtype=float).flatten()
    theta = np.linalg.norm(om) * dT

    if theta < 1e-12:
        return R.copy()

    K = skew(om / np.linalg.norm(om))
    Rot = (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )

    return R @ Rot

import numpy as np

def Euler2R(ang):
    """
    Converts an Euler angles vector [phi, theta, psi] into a rotation matrix
    of the Z-Y-X representation:
        R = rot_z(psi) * rot_y(theta) * rot_x(phi)

    Parameters:
        ang : array-like
            Euler angles [phi, theta, psi] in radians

    Returns:
        rot : numpy.ndarray
            3x3 rotation matrix
    """

    phi, theta, psi = ang

    rotx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])

    roty = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0,              1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])

    rot = rotz @ roty @ rotx

    return rot


EST_COLS = [
    "est_x", "est_y", "est_z",
    "est_vx", "est_vy", "est_vz",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
]

def get_initial_real_offset(param: "Param", phase="Main Controller", flip_pitch=True):
    """
    Return the real drone's initial state at the start of a flight phase.

    Parameters
    ----------
    param : Param
        Must provide param.sim.real_trajectory_csv.
    phase : str
        Value of the `flight_state` column that marks the interval of interest
        (the logger's mark_state("figure_8")).
    flip_pitch : bool
        The Crazyflie firmware logs stateEstimate.pitch with an inverted sign
        relative to the standard aerospace convention. True negates it before
        building the rotation matrix.

    Returns
    -------
    position : np.ndarray (3,1)
    velocity : np.ndarray (3,1)
    attitude : np.ndarray (9,1)
        Flattened row-major 3x3 rotation matrix, matching DroneState.attitude.
    ang_vel : np.ndarray (3,1)
        Body rates in rad/s.
    time0 : float
        `time_s` of the sample used for initialization.

    Usage
    -----
    agent = DroneState(param)
    (agent.position, agent.velocity, agent.attitude,
     agent.ang_vel, real_world_time_offset) = get_initial_real_offset(param)
    """
    csv_path = Path(param.sim.real_trajectory_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"Real trajectory CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["time_s", "flight_state", *EST_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    # Estimator columns are empty until the first log packets arrive.
    for col in ["time_s", *EST_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Every row carries its flight phase, so select the phase directly.
    states = df["flight_state"].fillna("").astype(str).str.strip()
    df_phase = df[states.eq(phase)]

    if df_phase.empty:
        available = sorted(s for s in states.unique() if s)
        raise ValueError(
            f"Flight state '{phase}' not found in CSV. Available: {available}"
        )

    # First sample in the phase with a complete estimator reading.
    df_phase = df_phase.dropna(subset=["time_s", *EST_COLS])
    if df_phase.empty:
        raise ValueError(
            f"No complete estimator samples found in phase '{phase}'."
        )

    row = df_phase.iloc[0]

    # =========================================================================
    # Initial values, shaped to match DroneState
    # =========================================================================
    position = row[["est_x", "est_y", "est_z"]].to_numpy(
        dtype=np.float64).reshape(3, 1)

    velocity = row[["est_vx", "est_vy", "est_vz"]].to_numpy(
        dtype=np.float64).reshape(3, 1)

    roll = np.deg2rad(row["roll_deg"])
    pitch = np.deg2rad(row["pitch_deg"])
    yaw = np.deg2rad(row["yaw_deg"])
    if flip_pitch:
        pitch = -pitch

    # DroneState.attitude is a flattened 3x3 rotation matrix (row-major),
    # matching np.eye(3).reshape(-1, 1).
    attitude = Euler2R([roll, pitch, yaw]).reshape(-1, 1)

    ang_vel = np.deg2rad(
        row[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy(
            dtype=np.float64)
    ).reshape(3, 1)

    time0 = float(row["time_s"])

    return position, velocity, attitude, ang_vel, time0

def _exp_so3(phi):
    """Rodrigues formula: rotation vector -> rotation matrix."""
    phi = np.asarray(phi).ravel()
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-9:
        return np.eye(3) + K + 0.5 * K @ K
    return (np.eye(3)
            + np.sin(theta) / theta * K
            + (1 - np.cos(theta)) / theta**2 * K @ K)

def rk4_step_rigid_body(deriv, p, v, R, om, dt):
    """
    One RK4 step for a rigid body.

    deriv(p, v, R, om) -> (dot_p, dot_v, dot_om)
    R is a 3x3 rotation matrix; om is the body-frame angular velocity (R_dot = R @ hat(om)).
    Position, velocity and angular velocity use standard RK4. The attitude is
    propagated on SO(3) using intermediate rotations and an RK4-weighted
    angular velocity, so R stays orthonormal.
    """
    # Stage 1
    k1p, k1v, k1o = deriv(p, v, R, om)
    om1 = om

    # Stage 2
    om2 = om + 0.5 * dt * k1o
    R2 = R @ _exp_so3(0.5 * dt * om1)
    k2p, k2v, k2o = deriv(p + 0.5 * dt * k1p, v + 0.5 * dt * k1v, R2, om2)

    # Stage 3
    om3 = om + 0.5 * dt * k2o
    R3 = R @ _exp_so3(0.5 * dt * om2)
    k3p, k3v, k3o = deriv(p + 0.5 * dt * k2p, v + 0.5 * dt * k2v, R3, om3)

    # Stage 4
    om4 = om + dt * k3o
    R4 = R @ _exp_so3(dt * om3)
    k4p, k4v, k4o = deriv(p + dt * k3p, v + dt * k3v, R4, om4)

    p_new  = p  + dt / 6 * (k1p + 2*k2p + 2*k3p + k4p)
    v_new  = v  + dt / 6 * (k1v + 2*k2v + 2*k3v + k4v)
    om_new = om + dt / 6 * (k1o + 2*k2o + 2*k3o + k4o)

    om_eff = (om1 + 2*om2 + 2*om3 + om4) / 6
    R_new = R @ _exp_so3(dt * om_eff)

    return p_new, v_new, R_new, om_new