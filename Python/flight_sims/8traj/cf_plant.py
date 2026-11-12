"""
Crazyflie plant of the thesis simulation (V8), self-contained: the parts the
flight simulations (main.py of swarm_take_off_hover_land/ and 8traj/) use,
copied unchanged from flight_sims/swarm_geometric_Fibonacci_sim_target
(Structs.py, drone_motion.py, drone_3dfull_dyn.py, drone_mellinger_ctrl.py,
helper_funcs.py), without the Fibonacci / MPC / display parts:

    Param             drone model (drone_models.toml, cf2x_P250) + Mellinger gains
    DroneState        state + desired state of one drone
    drone_motion      one step: Mellinger controller + full nonlinear dynamics, RK4 on SO(3)
    get_initial_real_offset   initial state of the drone(s) from a flight log
    make_param, ground_contact, passive_motion   helpers of the flight simulations

The same file is in swarm_take_off_hover_land/ and 8traj/ (keep both copies equal).
"""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

EST_COLS = ["est_x", "est_y", "est_z"]

# =============================================================================
# Parameters and state (V8 Structs.py)
# =============================================================================

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
        # Simulation
        self.Tend = 30
        self.dTi = 0.01
        self.Nsim = round(self.Tend / self.dTi) + 1
        self.t = np.arange(0, self.Tend + self.dTi, self.dTi)
        self.gravity = 9.81

        self.nD = 1
        self.real_trajectory_csv = None     # flight log (set by make_param)

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

# =============================================================================
# Rotations and integration (V8 helper_funcs.py)
# =============================================================================

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

# =============================================================================
# Dynamics, controller, one step (V8 drone_3dfull_dyn.py, drone_mellinger_ctrl.py, drone_motion.py)
# =============================================================================

def drone_3dfull_dyn(v, R, om, thrust, tau, param: Param, f_external=None, tau_external=None):
    """
    Full 3D drone dynamics: v world velocity (3,1), R body -> world, om body
    angular velocity (3,1), thrust magnitude, tau body torque (3,1).
    Returns (dot_p, dot_v, dot_om).
    """
    zW = np.array([[0.0], [0.0], [1.0]])
    zB = R[:, 2:3]

    f_external = np.zeros((3, 1)) if f_external is None else np.asarray(f_external, dtype=float).reshape(3, 1)
    tau_external = np.zeros((3, 1)) if tau_external is None else np.asarray(tau_external, dtype=float).reshape(3, 1)

    tau = np.asarray(tau, dtype=float).reshape(3, 1)
    thrust = float(np.asarray(thrust).item())

    # Translational drag: assumes D is expressed in the body frame
    D = np.abs(param.sim.drag_matrix)
    drag_force_world = -R @ D @ R.T @ v

    d_p = v
    d_v = (
        -param.sim.gravity * zW
        + (thrust / param.sim.drone_mass) * zB
        + (f_external + drag_force_world) / param.sim.drone_mass
    )

    J = param.sim.J
    angular_momentum = J @ om
    gyroscopic = np.cross(om.ravel(), angular_momentum.ravel()).reshape(3, 1)
    d_om = np.linalg.solve(J, -gyroscopic + tau + tau_external)

    return d_p, d_v, d_om


def drone_mellinger_ctrl(agent: DroneState, dpsi_d, param: Param):
    """Mellinger controller: (thrust, body torque) for the desired state of agent."""
    zW = np.array([[0.0], [0.0], [1.0]])   # world frame zW is up
    R = agent.attitude.reshape(3, 3)
    zB = R[:, 2].reshape(3, 1)             # drone frame zB

    # Translation errors
    e_p = agent.position - agent.desired.position
    e_v = agent.velocity - agent.desired.velocity

    # Desired force vector
    f_d = (-param.controller.Mellinger.kp @ e_p
           - param.controller.Mellinger.kv @ e_v
           + param.sim.drone_mass * param.sim.gravity * zW
           + param.sim.drone_mass * agent.desired.acceleration)

    # Thrust, clipped to the motor limits
    thrust = np.squeeze(f_d.T @ zB)
    thrust = np.clip(thrust, 4 * param.sim.thrust_min, 4 * param.sim.thrust_max)

    # Desired rotation matrix
    norm_fd = np.linalg.norm(f_d)
    if norm_fd < 1e-8:
        norm_fd = 1e-8
    zB_d = f_d / norm_fd
    xC_d = np.array([np.cos(agent.desired.psi), np.sin(agent.desired.psi), 0.0]).reshape(3, 1)
    yB_d = (skew(zB_d) @ xC_d) / (np.linalg.norm(skew(zB_d) @ xC_d))
    xB_d = skew(yB_d) @ zB_d
    R_d = np.column_stack((xB_d, yB_d, zB_d))

    # Desired angular velocity
    hw_d = (param.sim.drone_mass / thrust) * (
        agent.desired.jerk - (zB_d.T @ agent.desired.jerk) * zB_d
    )
    ang_p_d = (-hw_d.T @ yB_d)[0]
    ang_q_d = (hw_d.T @ xB_d)[0]
    ang_r_d = (dpsi_d * zW.T @ zB_d)[0]     # minimum-snap style yaw rate projection
    om_d = np.array([ang_p_d, ang_q_d, ang_r_d])

    # Torques
    e_om = agent.ang_vel - om_d
    e_R = 0.5 * unskew((R_d.T @ R) - (R.T @ R_d))
    torque = (
        -param.controller.Mellinger.kR @ e_R
        - param.controller.Mellinger.kom @ e_om
    )

    return thrust, torque


def drone_motion(drone: DroneState, param: Param):
    """Propagates the drone one time step (param.sim.dTi): Mellinger controller + dynamics, RK4."""
    pos = drone.position
    vel = drone.velocity
    R = drone.attitude.reshape(3, 3)
    om = drone.ang_vel

    dpsi_d = (drone.desired.psi - drone.desired.old_psi) / param.sim.dTi
    drone.desired.old_psi = drone.desired.psi

    thrust, torque = drone_mellinger_ctrl(drone, dpsi_d, param)

    def deriv(p_, v_, R_, om_):
        return drone_3dfull_dyn(v_, R_, om_, thrust, torque, param)

    p_new, v_new, R_new, om_new = rk4_step_rigid_body(deriv, pos, vel, R, om, param.sim.dTi)

    drone.position = p_new
    drone.velocity = v_new
    drone.attitude = R_new.reshape(-1, 1)
    drone.ang_vel = om_new

# =============================================================================
# Initial state from a flight log (V8 helper_funcs.py)
# =============================================================================

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

# =============================================================================
# Helpers of the flight simulations
# =============================================================================

def make_param(log_path, n_drones, t_end, dt=0.01):
    """V8 parameters, with the log as the real trajectory."""
    param = Param()
    param.sim.nD = n_drones
    param.sim.dTi = dt
    param.sim.Tend = t_end
    param.sim.Nsim = round(t_end / dt) + 1
    param.sim.t = np.arange(0, t_end + dt, dt)
    param.sim.real_trajectory_csv = Path(log_path) if log_path else None
    return param


def ground_contact(agent: DroneState, z_floor):
    """Drone resting on the floor: no penetration, no sliding, level (yaw kept)."""
    if agent.position[2, 0] < z_floor or (agent.position[2, 0] <= z_floor + 1e-9 and agent.velocity[2, 0] < 0):
        agent.position[2, 0] = z_floor
        agent.velocity[:] = 0.0
        yaw = Rotation.from_matrix(agent.attitude.reshape(3, 3)).as_euler("ZYX")[0]
        agent.attitude = Rotation.from_euler("z", yaw).as_matrix().reshape(-1, 1)
        agent.ang_vel[:] = 0.0
        return True
    return False


def passive_motion(agent: DroneState, param: Param):
    """Motors stopped: the dynamics with no thrust and no torque."""
    def deriv(p_, v_, R_, om_):
        return drone_3dfull_dyn(v_, R_, om_, 0.0, np.zeros((3, 1)), param)
    p, v, R, om = rk4_step_rigid_body(deriv, agent.position, agent.velocity,
                                      agent.attitude.reshape(3, 3), agent.ang_vel, param.sim.dTi)
    agent.position, agent.velocity, agent.attitude, agent.ang_vel = p, v, R.reshape(-1, 1), om
