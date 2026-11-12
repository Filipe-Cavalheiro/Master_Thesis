"""
MPC of the Python simulation (Python/obstacle_avoidance_funcs.py, AdmmMPC)
for the Crazyflies.

Every drone plans a MPC_HORIZON_TIME horizon to its formation goal: a
Bernstein polynomial (position, velocity, acceleration), smoothness + goal +
goal velocity cost, starting at the drone's current state (equality
constraint). The other drones' horizons (sphere of radius MPC_RADIUS_XY,
or the legacy cylinder MPC_RADIUS_XY / MPC_RADIUS_Z, see MPC_ENVELOPE), the arena geofence and the velocity/acceleration norms are
projection constraints solved with ADMM. The KKT matrix is constant (the
number of drones does not change), so it is factorized once.

Differences from the simulation, for real drones that track their setpoints
tightly:
  - the plan starts from the previous plan (at the current time) as long as
    the drone is within MPC_REPLAN_RESET_DIST of it, else from the measured
    state (velocity clipped to 0.95 v_max, acceleration of the previous plan);
  - scaled ADMM projection, b = Proj(A x + lambda / rho), and the
    multipliers reset every replanning (MPC_WARM_MULTIPLIERS);
  - the first horizon sample (fixed by the equality constraint) is left out
    of the obstacle constraints;
  - MPC_ENVELOPE = "cylinder" (legacy, flown on 26 Oct 2026) replaces the
    sphere around the other drones by a downwash cylinder (see project_others);
  - the setpoints are sampled from the plan at the current time, every
    SETPOINT_PERIOD, with the feedforward clipped to 1.2 v_max / 1.5 a_max.
"""
import time
import threading
import numpy as np
from math import comb
from scipy.linalg import lu_factor, lu_solve

from Structs import *

def bernstein_basis_matrix(t, n_degree):
    t = np.asarray(t)

    M = np.zeros((len(t), n_degree + 1))

    for k in range(n_degree + 1):
        M[:, k] = comb(n_degree, k) * t**k * (1 - t)**(n_degree - k)

    return M

def bernstein_bases(t, n_degree, T):
    """
    Position, velocity and acceleration bases (W0, W1, W2) at the
    normalized times t in [0, 1] of a horizon of duration T.
    """
    W0 = bernstein_basis_matrix(t, n_degree)

    B = bernstein_basis_matrix(t, n_degree-1)
    D1 = np.zeros((n_degree, n_degree+1))
    for i in range(n_degree):
        D1[i,i]   = -1
        D1[i,i+1] = 1
    W1 = n_degree * (B @ D1) / T

    B = bernstein_basis_matrix(t, n_degree-2)
    D2 = np.zeros((n_degree-1, n_degree+1))
    for i in range(n_degree-1):
        D2[i,i]   = 1
        D2[i,i+1] = -2
        D2[i,i+2] = 1
    W2 = n_degree * (n_degree-1) * (B @ D2) / T**2

    return W0, W1, W2

def clip_norm(v, r):
    """Rows of v with their norm clipped to r."""
    v = np.atleast_2d(v)
    d = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, r / np.maximum(d, 1e-12))

def project_vel_acc(Gx, H):
    """
    Projection of the stacked samples [v(3H); a(3H)] onto the balls
    ||v[k]|| <= MPC_V_MAX and ||a[k]|| <= MPC_A_MAX.
    """
    va = Gx.reshape(2, 3, H)                                  # [vel/acc, axis, k]
    d = np.linalg.norm(va, axis=1, keepdims=True)             # (2, 1, H)
    r = np.array([MPC_V_MAX, MPC_A_MAX]).reshape(2, 1, 1)
    scale = np.minimum(1.0, r / np.maximum(d, 1e-12))
    return (va * scale).reshape(-1, 1)

def project_others(pts, others, side, r_xy=MPC_RADIUS_XY):
    """
    Points projected out of the safety envelopes around the other drones:
    spheres of radius MPC_RADIUS_XY (MPC_ENVELOPE = "sphere", default), or the
    legacy cylinders (radius MPC_RADIUS_XY, half-height MPC_RADIUS_Z) with
    which the 26 Oct 2026 flights were flown. Legacy note: an ellipsoid of
    vertical radius 1.2 m
    whose projection pushes two stacked drones vertically (to 1.2 m apart,
    which the formation cannot give, so the goal cost wins and they stay
    stacked); the cylinder projection takes the shortest way out, sideways.

    pts    : (n, m, 3) points, one set per other drone
    others : (n, m, 3) horizons of the other drones
    side   : (n,) +1/-1, direction along y when exactly above/below (opposite for the other drone)
    returns: (n*3*m, 1), per obstacle laid out as [x(m), y(m), z(m)]
    """
    if MPC_ENVELOPE == "sphere":
        # sphere of radius r_xy: radial projection (the method of the paper,
        # as Python/obstacle_avoidance_funcs.py build_b_obs)
        delta = pts - others
        d = np.linalg.norm(delta, axis=2, keepdims=True)
        fallback = np.zeros_like(delta)
        fallback[..., 1] = side[:, None]
        u = np.where(d > 1e-9, delta / np.maximum(d, 1e-9), fallback)
        b = others + r_xy * np.maximum(d / r_xy, 1.0) * u
        return b.transpose(0, 2, 1).reshape(-1, 1)

    delta = pts - others                                      # (n, m, 3)
    dh = np.linalg.norm(delta[..., :2], axis=2)               # (n, m)
    dz = delta[..., 2]
    inside = (dh < r_xy) & (np.abs(dz) < MPC_RADIUS_Z)
    horizontal = inside & (r_xy - dh <= MPC_RADIUS_Z - np.abs(dz))
    vertical = inside & ~horizontal

    fallback = np.zeros_like(delta[..., :2])
    fallback[..., 1] = side[:, None]
    hdir = np.where((dh > 1e-6)[..., None], delta[..., :2] / np.maximum(dh, 1e-6)[..., None], fallback)

    b = pts.copy()
    b[..., :2] = np.where(horizontal[..., None], others[..., :2] + r_xy * hdir, b[..., :2])
    b[..., 2] = np.where(vertical, others[..., 2] + np.where(dz >= 0, 1.0, -1.0) * MPC_RADIUS_Z, b[..., 2])
    return b.transpose(0, 2, 1).reshape(-1, 1)

def project_arena(pts):
    """Points (m, 3) kept inside the geofence (and in front of the wall, trajectory 5); returns (3*m, 1) laid out as [x(m), y(m), z(m)]."""
    return clip_to_arena(pts).T.reshape(-1, 1)


class SwarmMPC:
    """
    Plans of all the drones (drone i = row i of the positions given to
    plan). with_target: the target is a drone, avoided as the others (its
    horizon: constant velocity from its measured state).
    """

    def __init__(self, nD: int, with_target: bool = False):
        n_degree = MPC_N_DEGREE
        H = MPC_HORIZON
        T = MPC_HORIZON_TIME
        n_ctrl = n_degree + 1
        dim = 3 * n_ctrl
        Hc = H - 1                     # constrained samples (sample 0 fixed by A_eq)

        self.nD, self.H, self.Hc, self.T = nD, H, Hc, T
        self.n_target = int(with_target)
        self.target_horizon = None     # (Hc, 3) predicted target positions of the current replanning
        self.n_degree, self.n_ctrl, self.dim = n_degree, n_ctrl, dim
        self.t_h = np.linspace(0, 1, H)
        W0, W1, W2 = bernstein_bases(self.t_h, n_degree, T)
        self.W0, self.W1, self.W2 = W0, W1, W2

        # Cost: smoothness + goal + goal velocity (x = [c0x, c0y, c0z, c1x, ...])
        Q = np.zeros((dim, dim))
        Qa = MPC_W_SMOOTH * W2.T @ W2 + MPC_W_GOAL * W0.T @ W0 + MPC_W_VEL * W1.T @ W1
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            Q[np.ix_(idx, idx)] = Qa

        # [p(0), v(0), a(0)] = start
        A_eq = np.zeros((9, dim))
        for axis in range(3):
            cols = np.arange(axis, dim, 3)
            A_eq[axis, cols]   = W0[0]
            A_eq[axis+3, cols] = W1[0]
            A_eq[axis+6, cols] = W2[0]

        # Positions of samples 1..H-1, laid out as [x(Hc), y(Hc), z(Hc)]
        A_blk = np.zeros((3 * Hc, dim))
        for axis in range(3):
            cols = np.arange(axis, dim, 3)
            A_blk[Hc*axis:Hc*(axis+1), cols] = W0[1:]

        # Velocities and accelerations, laid out as [vx(H), vy(H), vz(H), ax(H), ay(H), az(H)]
        G = np.zeros((6 * H, dim))
        for axis in range(3):
            cols = np.arange(axis, dim, 3)
            G[np.ix_(np.arange(axis * H, (axis + 1) * H), cols)] = W1
            G[np.ix_(np.arange((3 + axis) * H, (4 + axis) * H), cols)] = W2

        # One block per other drone (+ the target drone) + one block for the arena
        A_full = np.tile(A_blk, (nD + self.n_target, 1))
        A_bar = Q + MPC_RHO_OBS * A_full.T @ A_full + MPC_RHO_INEQ * G.T @ G
        KKT = np.block([
            [A_bar, A_eq.T],
            [A_eq, np.zeros((9, 9))]
        ])

        self.A_eq, self.A_full, self.A_full_T, self.G, self.GT = A_eq, A_full, A_full.T.copy(), G, G.T.copy()
        self.lu = lu_factor(KKT)
        self.others = [np.array([j for j in range(nD) if j != i], dtype=int) for i in range(nD)]

        self.lambda_obs = np.zeros((nD, A_full.shape[0], 1))
        self.lambda_ineq = np.zeros((nD, G.shape[0], 1))

        self.lock = threading.Lock()
        self.C = None                  # (nD, n_ctrl, 3) plans
        self.t_plan = None             # time.monotonic() at which the plans start

    # ============================================================
    # Plan evaluation
    # ============================================================
    def _eval(self, C, tau):
        """Position, velocity, acceleration of the plans C (..., n_ctrl, 3) at the normalized times tau."""
        W0, W1, W2 = bernstein_bases(np.atleast_1d(tau), self.n_degree, self.T)
        return W0 @ C, W1 @ C, W2 @ C

    def sample(self, idx, now):
        """Setpoint (p, v, a) of drone idx at time `now` and the age of its plan, or None before the first plan."""
        with self.lock:
            if self.C is None:
                return None
            C, t_plan = self.C[idx], self.t_plan
        tau = np.clip((now - t_plan) / self.T, 0.0, 1.0)
        p, v, a = self._eval(C, tau)
        v = clip_norm(v[0], 1.2 * MPC_V_MAX)[0]
        a = clip_norm(a[0], 1.5 * MPC_A_MAX)[0]
        return p[0], v, a, now - t_plan

    def _initial_state(self, positions, velocities, now):
        p0 = positions.copy()
        v0 = clip_norm(velocities, 0.95 * MPC_V_MAX)
        a0 = np.zeros((self.nD, 3))
        if self.C is None:
            return p0, v0, a0

        tau = np.clip((now - self.t_plan) / self.T, 0.0, 1.0)
        for i in range(self.nD):
            p, v, a = (m[0] for m in self._eval(self.C[i], tau))
            a0[i] = clip_norm(a, 0.95 * MPC_A_MAX)[0]
            if np.linalg.norm(p - positions[i]) < MPC_REPLAN_RESET_DIST:
                p0[i] = p
                v0[i] = clip_norm(v, 0.95 * MPC_V_MAX)[0]
        return p0, v0, a0

    def _goal_cost(self, p0, goal, goal_vel):
        """
        Linear cost term (Python set_goal_cost): exponential approach to the
        (moving) goal, never asking for more than v_max.
        """
        t = (self.t_h * self.T).reshape(-1, 1)
        err0 = (p0 - goal).reshape(1, 3)
        tau = max(MPC_GOAL_REF_TAU, np.linalg.norm(err0) / MPC_V_MAX)
        decay = np.exp(-t / tau)
        ref_pos = goal.reshape(1, 3) + t * goal_vel.reshape(1, 3) + decay * err0
        ref_vel = goal_vel.reshape(1, 3) - decay / tau * err0

        q = np.zeros((self.dim, 1))
        for axis in range(3):
            idx = np.arange(axis, self.dim, 3)
            q[idx] = (-MPC_W_GOAL * self.W0.T @ ref_pos[:, axis:axis+1]
                      - MPC_W_VEL * self.W1.T @ ref_vel[:, axis:axis+1])
        return q

    # ============================================================
    # ADMM
    # ============================================================
    def plan(self, positions, velocities, goals, goal_vel, now, target=None):
        """
        One replanning of all the drones from their state at `now`.
        positions, velocities, goals: (nD, 3); goal_vel: (3,) shared goal velocity;
        target: (position, velocity) of the target drone (with_target).
        """
        if self.n_target:
            tp, tv = (np.asarray(x, dtype=float).ravel() for x in target)
            self.target_horizon = tp + (self.t_h[1:, None] * self.T) * tv
        nD, H, Hc = self.nD, self.H, self.Hc
        A_full, A_full_T, G, GT = self.A_full, self.A_full_T, self.G, self.GT
        rho_o, rho_i = MPC_RHO_OBS, MPC_RHO_INEQ

        positions = np.asarray(positions, dtype=float)
        p0, v0, a0 = self._initial_state(positions, np.asarray(velocities, dtype=float), now)
        goals = clip_to_arena(goals)
        goal_vel = clip_norm(goal_vel, MPC_V_MAX)[0]

        b_eq = [np.concatenate([p0[i], v0[i], a0[i]]).reshape(-1, 1) for i in range(nD)]
        q = [self._goal_cost(p0[i], goals[i], goal_vel) for i in range(nD)]

        # Initial guess: previous plans shifted to `now` (or hovering at p0)
        if self.C is None:
            horizons = np.repeat(p0[:, None, :], H, axis=1)
            vel = np.zeros((nD, H, 3))
            acc = np.zeros((nD, H, 3))
        else:
            tau = np.clip((now - self.t_plan) / self.T + self.t_h, 0.0, 1.0)
            horizons, vel, acc = self._eval(self.C, tau)
        horizons = horizons.copy()

        if not MPC_WARM_MULTIPLIERS:
            self.lambda_obs[:] = 0.0
            self.lambda_ineq[:] = 0.0
        s = [project_vel_acc(np.concatenate([vel[i].T.reshape(-1), acc[i].T.reshape(-1)]), H)
             for i in range(nD)]
        b_obs = [self._project(i, self._stack(horizons, i), horizons) for i in range(nD)]

        C = np.zeros((nD, self.n_ctrl, 3))
        for _ in range(MPC_ITERS):
            for i in range(nD):
                lo, li = self.lambda_obs[i], self.lambda_ineq[i]
                rhs = (
                    - q[i]
                    - A_full_T @ lo
                    + rho_o * A_full_T @ b_obs[i]
                    - GT @ li
                    + rho_i * GT @ s[i]
                )
                x = lu_solve(self.lu, np.concatenate([rhs, b_eq[i]]))[:self.dim]
                C[i] = x.reshape(self.n_ctrl, 3)
                horizons[i] = self.W0 @ C[i]

                Ax = A_full @ x
                Gx = G @ x
                b_obs[i] = self._project(i, Ax + lo / rho_o, horizons)
                s[i] = project_vel_acc(Gx + li / rho_i, H)

                lo += rho_o * (Ax - b_obs[i])
                li += rho_i * (Gx - s[i])

        with self.lock:
            self.C = C
            self.t_plan = now

    def _stack(self, horizons, i):
        """A_full x of drone i's horizon: samples 1..H-1 once per block."""
        blk = horizons[i][1:].T.reshape(-1, 1)
        return np.tile(blk, (self.nD + self.n_target, 1))

    def _project(self, i, v, horizons):
        """b_obs of drone i: v (= A x + lambda / rho) projected per block [other drones ..., target, arena]."""
        Hc = self.Hc
        n_blocks = self.nD + self.n_target
        blocks = v.reshape(n_blocks, 3, Hc).transpose(0, 2, 1)      # (n_blocks, Hc, 3)
        parts = []
        others = self.others[i]
        if len(others):
            parts.append(project_others(blocks[:len(others)], horizons[others][:, 1:, :],
                                        np.where(others > i, 1.0, -1.0)))
        if self.n_target:
            parts.append(project_others(blocks[-2:-1], self.target_horizon[None], np.ones(1), TARGET_RADIUS_XY))
        parts.append(project_arena(blocks[-1]))
        return np.vstack(parts)

    def settle(self, positions, goals, duration=SLOT_SETTLE_TIME, target=None):
        """
        Positions the MPC holds around the (still) formation goals: ideal
        drones following the plans from `positions` for `duration`. The
        Voronoi goals can be stacked; the drones fly to these slots instead.
        target: position of the (still) target drone (with_target).
        """
        p = np.array(positions, dtype=float)
        v = np.zeros_like(p)
        t = 0.0
        target_state = None if target is None else (np.asarray(target, dtype=float), np.zeros(3))
        while t < duration:
            self.plan(p, v, goals, np.zeros(3), t, target_state)
            t += MPC_PERIOD
            samples = [self.sample(i, t) for i in range(self.nD)]
            p = np.array([sp[0] for sp in samples])
            v = np.array([sp[1] for sp in samples])

        with self.lock:
            self.C = None
            self.t_plan = None
        self.lambda_obs[:] = 0.0
        self.lambda_ineq[:] = 0.0
        return p

    # ============================================================
    # Planner thread
    # ============================================================
    def run(self, states: list[DroneState], formation, position_lock: threading.Lock,
            stop_event: threading.Event, abort_event: threading.Event):
        """Replans every MPC_PERIOD (states[i] is drone i) until stop_event."""
        next_time = time.monotonic()
        while not stop_event.is_set():
            try:
                self.plan_from(states, formation, position_lock)
            except Exception as e:
                print(f"!!! MPC ERROR: {e} !!!")
                abort_event.set()
                return

            next_time += MPC_PERIOD
            delay = next_time - time.monotonic()
            if delay > 0:
                stop_event.wait(delay)
            else:
                print(f"MPC overran its period by {-delay * 1000:.0f} ms")
                next_time = time.monotonic()

    def plan_from(self, states: list[DroneState], formation, position_lock: threading.Lock):
        """One replanning from the latest OptiTrack states and formation goals."""
        with position_lock:
            now = time.monotonic()
            positions = np.array([np.ravel(s.position)[:3] for s in states], dtype=float)
            velocities = np.array([np.ravel(s.velocity)[:3] for s in states], dtype=float)
        goals, goal_vel = formation.get_goals()
        self.plan(positions, velocities, goals, goal_vel, now,
                  formation.get_target_state() if self.n_target else None)

    def start_thread(self, states, formation, position_lock, stop_event, abort_event):
        t = threading.Thread(target=self.run, args=(states, formation, position_lock, stop_event, abort_event),
                             daemon=True)
        t.start()
        return t


def min_stacked_distance(points):
    """Min horizontal distance between two points less than MPC_RADIUS_Z - SLOT_STACK_TOL apart vertically (inf if none)."""
    p = np.asarray(points, dtype=float)
    iu = np.triu_indices(len(p), k=1)
    dh = np.linalg.norm(p[:, None, :2] - p[None, :, :2], axis=2)[iu]
    dz = np.abs(p[:, None, 2] - p[None, :, 2])[iu]
    close = dz < MPC_RADIUS_Z - SLOT_STACK_TOL
    return float(dh[close].min()) if np.any(close) else np.inf
