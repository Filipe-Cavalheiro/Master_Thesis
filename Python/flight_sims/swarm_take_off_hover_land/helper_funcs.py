"""Rotations, RK4 on SO(3) and the initial state from a flight log (as helper_funcs.py of the thesis simulation)."""

from pathlib import Path

import numpy as np
import pandas as pd

from Structs import Param

EST_COLS = ["est_x", "est_y", "est_z"]


def skew(x):
    """Skew-symmetric matrix of a 3-vector (so(3)) or a scalar (so(2))."""
    x = np.asarray(x).flatten()
    if len(x) == 3:
        return np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0],
        ])
    if len(x) == 1:
        return np.array([[0, -x[0]], [x[0], 0]])
    raise ValueError("SKEW function not implemented for input dimensions other than 1 or 3 (i.e., so(2) and so(3)).")


def unskew(M):
    """Vector of a 3x3 skew-symmetric matrix."""
    return np.array([[M[2, 1]], [M[0, 2]], [M[1, 0]]])


def Euler2R(ang):
    """Z-Y-X Euler angles [phi, theta, psi] -> rotation matrix rot_z(psi) rot_y(theta) rot_x(phi)."""
    phi, theta, psi = ang
    rotx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    roty = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    rotz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return rotz @ roty @ rotx


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
    One RK4 step for a rigid body. deriv(p, v, R, om) -> (dot_p, dot_v, dot_om),
    R body -> world, om body-frame angular velocity (R_dot = R hat(om)). The
    attitude is propagated on SO(3) with intermediate rotations and an
    RK4-weighted angular velocity, so R stays orthonormal.
    """
    k1p, k1v, k1o = deriv(p, v, R, om)
    om1 = om

    om2 = om + 0.5 * dt * k1o
    R2 = R @ _exp_so3(0.5 * dt * om1)
    k2p, k2v, k2o = deriv(p + 0.5 * dt * k1p, v + 0.5 * dt * k1v, R2, om2)

    om3 = om + 0.5 * dt * k2o
    R3 = R @ _exp_so3(0.5 * dt * om2)
    k3p, k3v, k3o = deriv(p + 0.5 * dt * k2p, v + 0.5 * dt * k2v, R3, om3)

    om4 = om + dt * k3o
    R4 = R @ _exp_so3(dt * om3)
    k4p, k4v, k4o = deriv(p + dt * k3p, v + dt * k3v, R4, om4)

    p_new = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
    v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    om_new = om + dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)

    om_eff = (om1 + 2 * om2 + 2 * om3 + om4) / 6
    R_new = R @ _exp_so3(dt * om_eff)

    return p_new, v_new, R_new, om_new


def get_initial_real_offset(param: Param, iD: int | None = None, phase: str = "Main Controller",
                            flip_pitch: bool = True):
    """
    Initial state of the real drone(s) at the start of a flight phase of the
    log param.sim.real_trajectory_csv: each drone's first complete estimator
    sample in the phase ('flight_state' column).

    Returns (position (3,1), velocity (3,1), attitude (9,1) row-major R,
    ang_vel (3,1), time0) for one drone (iD given, or a log without an 'id'
    column), or lists of them ordered by drone ID (iD None).
    """
    csv_path = Path(param.sim.real_trajectory_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Real trajectory CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # a complete sample: every value the initial state is built from (the
    # attitude / gyro of the first samples of a flight can be missing)
    numeric_cols = list(dict.fromkeys(["time_s", *EST_COLS, "est_vx", "est_vy", "est_vz",
                                       "roll_deg", "pitch_deg", "yaw_deg",
                                       "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]))
    required = ["flight_state", *numeric_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    states = df["flight_state"].fillna("").astype(str).str.strip()
    df_phase = df.loc[states.eq(phase)].copy()
    if df_phase.empty:
        available = sorted(s for s in states.unique() if s)
        raise ValueError(f"Flight state '{phase}' not found in CSV. Available: {available}")

    df_phase = df_phase.dropna(subset=numeric_cols)
    if df_phase.empty:
        raise ValueError(f"No complete estimator samples found in phase '{phase}'.")

    has_id = "id" in df_phase.columns
    if has_id:
        df_phase = df_phase.dropna(subset=["id"])
        if df_phase.empty:
            raise ValueError(f"No valid drone IDs found in phase '{phase}'.")

        if iD is not None:
            ids = pd.to_numeric(df_phase["id"], errors="coerce")
            df_phase = df_phase.loc[ids.eq(iD)]
            if df_phase.empty:
                available = sorted(pd.to_numeric(df["id"], errors="coerce").dropna().unique().tolist())
                raise ValueError(f"Drone ID {iD} has no complete estimator samples "
                                 f"in phase '{phase}'. Available IDs: {available}")
            drone_ids = [iD]
        else:
            drone_ids = sorted(df_phase["id"].unique().tolist())
            if len(drone_ids) < param.sim.nD:
                raise ValueError(f"Not enough drones in the log: {len(drone_ids)} available, "
                                 f"{param.sim.nD} desired")
    else:
        drone_ids = [None]      # no ID column: one drone

    positions, velocities, attitudes, angular_velocities, timestamps = [], [], [], [], []

    for drone_id in drone_ids:
        drone_df = df_phase.loc[df_phase["id"].eq(drone_id)] if has_id else df_phase
        row = drone_df.loc[drone_df["time_s"].idxmin()]     # earliest sample, not the first row

        position = row[["est_x", "est_y", "est_z"]].to_numpy(dtype=np.float64).reshape(3, 1)
        velocity = row[["est_vx", "est_vy", "est_vz"]].to_numpy(dtype=np.float64).reshape(3, 1)

        roll = np.deg2rad(row["roll_deg"])
        pitch = np.deg2rad(row["pitch_deg"])
        yaw = np.deg2rad(row["yaw_deg"])
        if flip_pitch:
            pitch = -pitch
        attitude = np.asarray(Euler2R([roll, pitch, yaw]), dtype=np.float64).reshape(9, 1)

        ang_vel = np.deg2rad(
            row[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy(dtype=np.float64)
        ).reshape(3, 1)

        positions.append(position)
        velocities.append(velocity)
        attitudes.append(attitude)
        angular_velocities.append(ang_vel)
        timestamps.append(float(row["time_s"]))

    if len(drone_ids) == 1:
        return positions[0], velocities[0], attitudes[0], angular_velocities[0], timestamps[0]
    return positions, velocities, attitudes, angular_velocities, timestamps
