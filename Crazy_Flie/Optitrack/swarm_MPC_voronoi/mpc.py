"""
MPC of the formation: the distributed Bernstein-polynomial MPC of the thesis
(Chapter 6, "Distributed Bernstein-Polynomial MPC"), the same code as the
Python simulation V17 (Python/flight_sims/V17_Lisbon_aeroport/Python/
obstacle_avoidance_funcs.py: init_obstacle_avoidance, set_goal_cost,
AdmmMPC). This module only feeds it the OptiTrack states and the formation
goals, and runs it in a thread for the Crazyflies.

As in the thesis, at every control step (MPC_PERIOD) every chaser plans a
1 s horizon (Bernstein degree MPC_N_DEGREE, MPC_HORIZON samples) from its
current state to its formation goal:
  - cost: tracking of the exponential approach to the goal moving with the
    goal velocity (w_g, w_v, tau_r0) + smoothness (w_s);
  - constraints, all projections solved with scaled ADMM (rho_obs, rho_ineq):
    |v| <= v_max, |a| <= a_max, a sphere of radius MPC_SAFETY_RADIUS around
    the other chasers' horizons (and around the constant-velocity prediction
    of a target drone), no airspace (free space: the arena has no geofence
    in the MPC);
  - multipliers warm-started from the previous control step, up to MPC_ITERS
    iterations with the 1e-2 residual stop, drones updated in turn
    (Gauss-Seidel);
  - setpoint: the plan at the second horizon sample t_1 (position, velocity,
    acceleration), sent as a full-state setpoint until the next control step.

Initial state of every plan: the OptiTrack position and velocity, zero
acceleration, as the V17 code (its agent.acceleration stays zero). Feeding
back the acceleration of the setpoint sent last instead (the Crazyflies do
not measure it) made the closed loop diverge in the dry run: the plan's
acceleration at t_1 becomes the next plan's initial acceleration.
"""
import time
import threading
from types import SimpleNamespace

import numpy as np

import Structs as S                                    # also puts the V17 simulation on sys.path
from Structs_sim import SimulationConfig               # V17
from Structs_agent import DroneState as MpcAgent       # V17
from obstacle_avoidance_funcs import (                 # V17
    init_obstacle_avoidance, solve_unconstrained_motion, set_goal_cost, AdmmMPC, bernstein_eval)

if S.MPC_HORIZON_TIME != 1.0:
    raise ValueError("The V17 MPC plans a 1 s horizon (MPC_HORIZON_TIME must be 1.0)")


class FreeSpace:
    """Airspace without restrictions: the airspace block of the MPC leaves the horizon unchanged."""
    enabled = False

    def project(self, pts):
        return np.asarray(pts, dtype=float)


def mpc_param(nD: int, with_target: bool):
    """param of the V17 MPC with the MPC_* settings of Structs.py (read now, so a replay can override them)."""
    sim = SimulationConfig()
    sim.nD = nD
    sim.dTi = S.MPC_PERIOD
    sim.n_degree = S.MPC_N_DEGREE
    sim.horizon = S.MPC_HORIZON
    sim.mpc_iters = S.MPC_ITERS
    sim.rho_obs = S.MPC_RHO_OBS
    sim.rho_ineq = S.MPC_RHO_INEQ
    sim.w_goal = S.MPC_W_GOAL
    sim.w_vel = S.MPC_W_VEL
    sim.w_smooth = S.MPC_W_SMOOTH
    sim.goal_ref_tau = S.MPC_GOAL_REF_TAU
    sim.v_max = S.MPC_V_MAX
    sim.a_max = S.MPC_A_MAX
    sim.obstacle_avoidance_dim = S.MPC_SAFETY_RADIUS
    sim.target_is_drone = with_target
    sim.mpc_tight_tracking = S.MPC_START == "plan"   # V17: start from the previous plan (Crazyflies)
    sim.replan_reset_dist = S.MPC_REPLAN_RESET_DIST
    sim.admm_warm_multipliers = True        # thesis: warm start
    sim.env_clearance = 0.0
    return SimpleNamespace(sim=sim, airspace=FreeSpace())


def mpc_settings():
    """The MPC settings in use (saved in <log>_meta.json)."""
    return {k: getattr(S, k) for k in (
        "MPC_PERIOD", "MPC_HORIZON_TIME", "MPC_HORIZON", "MPC_N_DEGREE", "MPC_ITERS",
        "MPC_RHO_OBS", "MPC_RHO_INEQ", "MPC_W_GOAL", "MPC_W_VEL", "MPC_W_SMOOTH",
        "MPC_GOAL_REF_TAU", "MPC_V_MAX", "MPC_A_MAX", "MPC_SAFETY_RADIUS", "MPC_SETPOINT", "MPC_START",
        "MPC_REPLAN_RESET_DIST")}


def col(v):
    return np.asarray(v, dtype=float).reshape(3, 1)


class SwarmMPC:
    """
    Plans of all the chasers (chaser i = row i of the positions given to
    plan). with_target: the target is a drone, avoided as the others (its
    horizon: constant velocity from its measured state).
    """

    def __init__(self, nD: int, with_target: bool = False):
        self.nD = nD
        self.param = mpc_param(nD, with_target)
        self.agents = [MpcAgent(self.param) for _ in range(nD)]
        self.obs_vars = init_obstacle_avoidance(self.agents, self.param)
        self.W = (self.obs_vars["W0"], self.obs_vars["W1"], self.obs_vars["W2"])
        self.active = np.arange(nD)
        self.mpc = None                 # AdmmMPC, built at the first plan (from the drones' states)
        self.lock = threading.Lock()
        self.setpoints = None           # (nD, 3, 3): p, v, a at t_1 of the latest plans
        self.t_plan = None              # time at which the latest plans start

    # ============================================================
    # Planning
    # ============================================================
    def _start(self, target_state):
        """First plan: unconstrained plans from the current states, then the ADMM state (as V17 main)."""
        G = self.obs_vars["G"]
        for a in self.agents:
            set_goal_cost(a, self.obs_vars, self.param)
            a.constraints.s = np.zeros((G.shape[0], 1))
            a.constraints.lambda_obs = np.zeros((3 * self.param.sim.horizon, 1))
            a.constraints.lambda_ineq = np.zeros((G.shape[0], 1))
            a.constraints.A_obs = self.obs_vars["A_obs_blk"]
            solve_unconstrained_motion(self.obs_vars, a, self.param)
        self.mpc = AdmmMPC(self.obs_vars, self.agents, self.param)
        self.mpc.target_state = target_state or (np.zeros((3, 1)), np.zeros((3, 1)))
        self.mpc.rebuild(self.active)

    def plan(self, positions, velocities, goals, goal_vel, now, target=None, accelerations=None):
        """
        One control step of all the chasers from their state at `now`.
        positions, velocities, goals: (nD, 3); goal_vel: (3,) goal velocity
        (shared); target: (position, velocity) of the target drone;
        accelerations: (nD, 3), default zero (as V17).
        """
        if accelerations is None:
            accelerations = np.zeros((self.nD, 3))
        for i, a in enumerate(self.agents):
            a.position = col(positions[i])
            a.velocity = col(velocities[i])
            a.acceleration = col(accelerations[i])
            a.desired.goal = col(goals[i])
            a.desired.goal_velocity = col(goal_vel)
        if target is not None:
            target_state = (col(target[0]), col(target[1]))
        else:
            target_state = None
        if self.mpc is None:
            self._start(target_state)
        if target_state is not None:
            self.mpc.target_state = target_state
        for a in self.agents:
            set_goal_cost(a, self.obs_vars, self.param)

        self.mpc.solve(self.active)

        if S.MPC_SETPOINT == "t1":            # thesis: second horizon sample
            W0, W1, W2 = self.W
            sp = np.array([[W0[1] @ a.constraints.C, W1[1] @ a.constraints.C, W2[1] @ a.constraints.C]
                           for a in self.agents])
        else:                                 # "next_step": the plan one control period ahead (V17 mpc_setpoint)
            tau = S.MPC_PERIOD / S.MPC_HORIZON_TIME
            sp = np.array([bernstein_eval(a.constraints.C, tau, self.param.sim.n_degree) for a in self.agents])
        with self.lock:
            self.setpoints = sp
            self.t_plan = now

    def sample(self, idx, now):
        """Setpoint (p, v, a) of chaser idx (the plan at t_1) and the age of its plan, or None before the first plan."""
        with self.lock:
            if self.setpoints is None:
                return None
            p, v, a = (x.copy() for x in self.setpoints[idx])
            return p, v, a, now - self.t_plan

    def horizon(self, idx):
        """Planned positions (H, 3) of chaser idx."""
        return self.W[0] @ self.agents[idx].constraints.C

    def settle(self, positions, goals, duration=S.SLOT_SETTLE_TIME, target=None):
        """
        Positions the MPC holds around the (still) formation goals: ideal
        drones following their plans from `positions` for `duration` (the
        Voronoi goals can be closer than the safety sphere). The drones fly
        to these slots before the Formation phase. target: position of the
        (still) target drone. The MPC then starts again from scratch.
        """
        p = np.array(positions, dtype=float)
        v = np.zeros_like(p)
        t = 0.0
        target_state = None if target is None else (np.asarray(target, dtype=float), np.zeros(3))
        n = self.param.sim.n_degree
        tau = S.MPC_PERIOD / S.MPC_HORIZON_TIME
        while t < duration:
            self.plan(p, v, goals, np.zeros(3), t, target_state)
            for i, ag in enumerate(self.agents):
                p[i], v[i], _ = bernstein_eval(ag.constraints.C, tau, n)
            t += S.MPC_PERIOD

        self.mpc = None
        with self.lock:
            self.setpoints = None
            self.t_plan = None
        return p

    # ============================================================
    # Planner thread
    # ============================================================
    def run(self, states: list, formation, position_lock: threading.Lock,
            stop_event: threading.Event, abort_event: threading.Event):
        """Replans every MPC_PERIOD (states[i] is chaser i) until stop_event."""
        next_time = time.monotonic()
        while not stop_event.is_set():
            try:
                self.plan_from(states, formation, position_lock)
            except Exception as e:
                print(f"!!! MPC ERROR: {e} !!!")
                abort_event.set()
                return

            next_time += S.MPC_PERIOD
            delay = next_time - time.monotonic()
            if delay > 0:
                stop_event.wait(delay)
            else:
                print(f"MPC overran its period by {-delay * 1000:.0f} ms")
                next_time = time.monotonic()

    def plan_from(self, states: list, formation, position_lock: threading.Lock):
        """One control step from the latest OptiTrack states and formation goals."""
        with position_lock:
            now = time.monotonic()
            positions = np.array([np.ravel(s.position)[:3] for s in states], dtype=float)
            velocities = np.array([np.ravel(s.velocity)[:3] for s in states], dtype=float)
        goals, goal_vel = formation.get_goals()
        self.plan(positions, velocities, goals, goal_vel, now,
                  formation.get_target_state() if self.param.sim.target_is_drone else None)

    def start_thread(self, states, formation, position_lock, stop_event, abort_event):
        t = threading.Thread(target=self.run, args=(states, formation, position_lock, stop_event, abort_event),
                             daemon=True)
        t.start()
        return t


def min_distance(points):
    """Min distance between two of the points (inf for fewer than two)."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return np.inf
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return float(d[np.triu_indices(len(p), k=1)].min())
