"""
Real-world simulation of the Crazyflie flights (OptiTrack): runs the flight
code of a swarm folder of Crazy_Flie/Optitrack (its Formation: target and
Voronoi formation; its SwarmMPC, for swarm_MPC_voronoi the V17 MPC of this
folder; its setpoint loop) on a simulated clock, with the drones simulated by
the Crazyflie plant of this simulation (plant.py, CrazyfliePlant: onboard
Mellinger with the firmware gains, experiment_arena.py CRAZYFLIE_ONBOARD).

Two runs (main.py --IRL_log / --arena, or Crazy_Flie/Optitrack/flight_sim.py),
as the geometric Fibonacci simulation (swarm_geometric_Fibonacci_sim_target):
  - replay of a flight log (--IRL_log): the whole flight, from the drones at
    rest on the floor at their logged takeoff points, with the commands of
    the flight code (simulate_flight): take off, approach to the formation
    slots, the Formation phase (formation, MPC, controllers), return to the
    takeoff points, landing, motors off. Each phase starts when it did in the
    log, with the flight setup of <log>_meta.json when it exists (scene,
    drone, target mode and trajectory, formation goals and approach slots,
    formation and MPC settings, Mellinger position gains read back from the
    firmware) and the periods measured in the log. Writes <log>_sim.csv
    next to the log (same columns) and compares: errors printed (whole
    flight and per phase), figures (PDF and PNG) in <run folder>/comparison/;
  - dry run (--arena, no log): only the Formation phase of the flight code
    with its current settings, the drones starting still on the slots it
    flies them to before it. Saved to Results/<code>_dry_run_traj<k>_<n>drones_<date>/.

A flight code with a simulated / real target switch (swarm_MPC_voronoi:
REAL_TARGET_DRONE_ID in its Structs.py, -1 = simulated target) is named
<folder>_<sim|real>_target, as its runs: the mode comes from
<log>_meta.json, the run folder of the log, --target, or that name (else
REAL_TARGET_DRONE_ID). --traj sets TARGET_TRAJ (default: from
<log>_meta.json, else Structs.py).
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parent                    # this simulation (plant, arena, MPC)
REPO = next(p for p in PYTHON_DIR.parents if (p / ".git").exists())
LOGS_DIR = REPO / "Results"                                     # repository Results folder, one subfolder per run
FLIGHT_DIR = REPO / "Crazy_Flie" / "Optitrack"                  # the flight codes

POS_COLS = ["opti_x", "opti_y", "opti_z"]
SP_COLS = ["sp_x", "sp_y", "sp_z"]
TARGET_COLS = ["target_x", "target_y", "target_z"]
PHASE = "Formation"
FLIGHT_PHASES = ("Take off", "Approach", "Formation", "Return", "Land")   # logger.mark_state of the flight code
OPTI_PERIOD = 0.01      # [s] OptiTrack frames (velocity EMA of natnet_funcs)
PLANT_PERIOD = 0.01     # [s] simulation step of the plant (substeps inside CrazyfliePlant)


# ============================================================
# LOG
# ============================================================

def load_flight(csv_path, meta=None):
    """
    The whole flight, from the start of the take off: drone ids (sorted:
    drone i of the flight code), takeoff points (homes, on the floor: from
    <log>_meta.json, else the first logged position), the phase start times,
    per drone the logged track (time from the take off start, position,
    setpoint, target, phase) and the setpoint / formation update periods
    measured in the Formation phase.
    """
    df = pd.read_csv(csv_path, dtype={"id": str, "flight_state": str})
    df["flight_state"] = df["flight_state"].fillna("").str.strip()
    ids = sorted(df["id"].dropna().unique(), key=lambda s: int(s) if s.isdigit() else s)

    takeoff = df[df["flight_state"] == "Take off"]
    if takeoff.empty:
        raise ValueError(f"No 'Take off' phase in {csv_path}")
    t0 = takeoff["time_s"].min()
    phases = {}
    for name in (*FLIGHT_PHASES, "EOF"):
        rows = df[df["flight_state"] == name]
        if not rows.empty:
            phases[name] = float(rows["time_s"].min() - t0)
    phases.setdefault("EOF", float(df["time_s"].max() - t0))

    md = (meta or {}).get("drones", {})
    homes = np.array([md[i]["home"] if i in md and "home" in md[i]
                      else takeoff[takeoff["id"] == i][POS_COLS].iloc[0].to_numpy(float) for i in ids], dtype=float)

    flight = df[df["time_s"] >= t0]
    drones = {}
    for i in ids:
        g = flight[flight["id"] == i].sort_values("time_s")
        drones[i] = {
            "t": g["time_s"].to_numpy(float) - t0,
            "p": g[POS_COLS].to_numpy(float),
            "sp": g[SP_COLS].to_numpy(float),
            "target": g[TARGET_COLS].to_numpy(float),
            "phase": g["flight_state"].to_numpy(str),
        }

    def period(t, x):
        # mean interval between changes of x (logged at 50 Hz), without the
        # gaps where x did not change (setpoint already on its goal)
        changed = np.r_[False, np.any(np.diff(x, axis=0) != 0, axis=1)]
        dt = np.diff(t[changed])
        dt = dt[np.isfinite(dt)]
        return float(np.mean(dt[dt < 1.8 * np.median(dt)])) if len(dt) > 2 else None

    d0 = drones[ids[0]]
    in_formation = d0["phase"] == PHASE
    periods = {"setpoint": None, "formation": None}
    if in_formation.sum() > 3:
        periods = {"setpoint": period(d0["t"][in_formation], d0["sp"][in_formation]),
                   "formation": period(d0["t"][in_formation], d0["target"][in_formation])}
    return ids, homes, phases, drones, periods


# ============================================================
# SIMULATION
# ============================================================

_FLIGHT_CODE = {}
_SAFETY = argparse.Namespace(radius=None)    # MPC safety radius of the flight code (dry-run figure)


def import_flight_code(folder):
    """
    The flight modules of `folder` (Structs, formation, voronoi_mesh_funcs,
    mpc), imported once. The flight folder and this simulation both have a
    voronoi_mesh_funcs.py: the flight modules get the flight one, and this
    simulation keeps its own in sys.modules.
    """
    folder = Path(folder).resolve()
    if folder in _FLIGHT_CODE:
        return _FLIGHT_CODE[folder]
    flight_names = {p.stem for p in folder.glob("*.py")}
    shared = {n for n in flight_names if (PYTHON_DIR / f"{n}.py").exists()}
    saved = {n: sys.modules.pop(n) for n in shared if n in sys.modules}
    if str(PYTHON_DIR) not in sys.path:
        sys.path.append(str(PYTHON_DIR))
    sys.path.insert(0, str(folder))
    try:
        import Structs
        from formation import Formation
        from voronoi_mesh_funcs import runCVT, pos_based_mesh_transform
        try:
            from mpc import SwarmMPC
        except ImportError:
            SwarmMPC = None
    finally:
        sys.path.remove(str(folder))
        for n in shared:                      # back to this simulation's modules
            sys.modules.pop(n, None)
            if n in saved:
                sys.modules[n] = saved[n]
    _FLIGHT_CODE[folder] = (Structs, Formation, SwarmMPC, runCVT, pos_based_mesh_transform)
    return _FLIGHT_CODE[folder]


def make_plant(nD, setpoint_mode, gains=None, drone_model=None):
    from Structs_sim import Param as SimParam
    from Structs_agent import DroneState as SimDrone
    from experiment_arena import DRONE_MODEL, configure_onboard_gains
    from plant import CrazyfliePlant

    param = SimParam()
    param.sim.load_drone_model(drone_model or DRONE_MODEL)
    param.sim.dTi = PLANT_PERIOD
    configure_onboard_gains(param)
    if gains:   # Mellinger values read back from the firmware (<log>_meta.json, firmware_gains)
        mel = param.controller.Mellinger
        mel["kp"] = np.diag([gains["ctrlMel.kp_xy"], gains["ctrlMel.kp_xy"], gains["ctrlMel.kp_z"]])
        mel["kv"] = np.diag([gains["ctrlMel.kd_xy"], gains["ctrlMel.kd_xy"], gains["ctrlMel.kd_z"]])
        if "ctrlMel.mass" in gains:
            mel["mass"] = gains["ctrlMel.mass"]
        if "ctrlMel.ki_xy" in gains and "ctrlMel.ki_z" in gains:
            mel["ki"] = np.diag([gains["ctrlMel.ki_xy"], gains["ctrlMel.ki_xy"], gains["ctrlMel.ki_z"]])
        if "ctrlMel.i_range_xy" in gains and "ctrlMel.i_range_z" in gains:
            mel["i_range"] = np.array([gains["ctrlMel.i_range_xy"], gains["ctrlMel.i_range_xy"],
                                       gains["ctrlMel.i_range_z"]])
    plant = CrazyfliePlant(param, setpoint_mode=setpoint_mode)
    return plant, [SimDrone(param) for _ in range(nD)]


def ground_contact(drone, z_floor):
    """Drone resting on the floor: no penetration, no sliding, level (yaw kept)."""
    if drone.position[2, 0] < z_floor or (drone.position[2, 0] <= z_floor + 1e-9 and drone.velocity[2, 0] < 0):
        drone.position[2, 0] = z_floor
        drone.velocity[:] = 0.0
        R = drone.attitude.reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s_ = np.cos(yaw), np.sin(yaw)
        drone.attitude = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]).reshape(-1, 1)
        drone.ang_vel[:] = 0.0


class FlightCode:
    """
    The flight code of a swarm folder (imported by import_flight_code) for
    the drones `ids` (target drone included, if the folder has one): its
    Formation (target thread and Voronoi worker), its SwarmMPC thread and the
    setpoint loop of its controllers, stepped on the simulated clock.
    """

    def __init__(self, folder, ids, periods):
        S, Formation, SwarmMPC, runCVT, pos_based_mesh_transform = import_flight_code(folder)
        self.S, self.Formation = S, Formation
        self.runCVT, self.pos_based_mesh_transform = runCVT, pos_based_mesh_transform
        self.use_mpc = SwarmMPC is not None
        self.setpoint_mode = getattr(S, "SETPOINT_MODE", "position") if self.use_mpc else "position"
        self.has_target = getattr(S, "TARGET_URI", None) is not None
        self.target_id = str(S.TARGET_DRONE_ID) if self.has_target else None
        self.ch = [k for k, i in enumerate(ids) if i != self.target_id]     # chasers (drone k of the flight code)
        self.ti = ids.index(self.target_id) if self.has_target else None
        self.nD, self.n_all = len(self.ch), len(ids)
        self.mpc = (SwarmMPC(self.nD, with_target=True) if self.has_target else SwarmMPC(self.nD)) \
            if self.use_mpc else None
        self.periods = {"setpoint": periods.get("setpoint") or S.SETPOINT_PERIOD,
                        "formation": periods.get("formation") or S.VORONOI_PERIOD}
        self.formation = None
        if self.has_target:
            from formation import target_reference
            self.target_reference = target_reference

    def new_formation(self, sites_near=None):
        """
        The formation as the flight code builds it before arming (converged).
        sites_near: start the CVT from the mesh vertices nearest to these
        points (the formation goals of <log>_meta.json); the flight code
        starts from random sites.
        """
        S = self.S
        self.formation = self.Formation(S.Param(number_of_drones=self.nD, traj=S.TARGET_TRAJ))
        if sites_near is not None:
            v = self.formation.voronoi
            d = np.linalg.norm(v.vertices[None, :, :] - np.asarray(sites_near)[:, None, :], axis=2)
            v.sites = np.argmin(d, axis=1)
        self.formation.converge()
        return self.formation

    def target_start(self):
        if self.has_target:
            return self.target_reference(self.formation.param, 0.0)[0]
        return np.asarray(self.formation.param.ref_target_traj["pos"](0), dtype=float).ravel()

    def start(self, t0):
        """Formation phase from time t0: the threads start, the first plan, then the controllers."""
        self.t0 = t0
        self.t_next = {"formation": t0, "mpc": t0, "setpoint": t0}
        self.k_vor = 0

    def step(self, t, meas_p, meas_v, sp, sp_v, sp_a):
        """Everything of the Formation phase due at time t (updates the setpoints sp, sp_v, sp_a in place)."""
        S, f, ch, ti = self.S, self.formation, self.ch, self.ti
        tf, eps = t - self.t0, 1e-9

        # Formation threads: target (virtual, or the measured target drone),
        # then the Voronoi worker (same period)
        if t + eps >= self.t_next["formation"]:
            with f.lock:
                if self.has_target:
                    f.target, f.target_velocity = meas_p[ti].copy(), meas_v[ti].copy()
                else:
                    f.target = np.array(f.param.ref_target_traj["pos"](tf), dtype=float)
                    if hasattr(f, "target_velocity"):
                        f.target_velocity = np.array(f.param.ref_target_traj["vel"](tf), dtype=float)
            v = f.voronoi
            v.vertices, v.faces, v.mesh_pos = self.pos_based_mesh_transform(f.get_target(), v, f.param)
            if self.k_vor == 1 or self.k_vor % 5 == 0:
                v.sites = self.runCVT(v, v.sites)
            with f.lock:
                goals = v.vertices[v.sites, :]
                f.desired_positions = (S.clip_to_arena(goals) if hasattr(S, "clip_to_arena")
                                       else np.clip(goals, S.GOAL_MIN, S.GOAL_MAX))
            self.k_vor += 1
            self.t_next["formation"] += self.periods["formation"]

        if self.use_mpc and t + eps >= self.t_next["mpc"]:
            goals, goal_vel = f.get_goals()
            if self.has_target:
                self.mpc.plan(meas_p[ch], meas_v[ch], goals, goal_vel, tf, (meas_p[ti], meas_v[ti]))
            else:
                self.mpc.plan(meas_p[ch], meas_v[ch], goals, goal_vel, tf)
            self.t_next["mpc"] += S.MPC_PERIOD

        # controllers
        if t + eps >= self.t_next["setpoint"]:
            for j, k in enumerate(ch):
                if self.use_mpc:
                    p, v_, a_, _ = self.mpc.sample(j, tf)
                    sp[k], sp_v[k], sp_a[k] = p, v_, a_
                else:
                    step = f.goal(j) - sp[k]
                    dist = np.linalg.norm(step)
                    max_step = S.MAX_SETPOINT_SPEED * S.SETPOINT_PERIOD
                    if dist > max_step:
                        step *= max_step / dist
                    sp[k] = sp[k] + step
            if self.has_target:
                sp[ti], sp_v[ti] = self.target_reference(f.param, tf)
                sp_a[ti] = 0.0
            self.t_next["setpoint"] += self.periods["setpoint"]


class Measurement:
    """OptiTrack of the flight code: position every OPTI_PERIOD, finite-difference velocity, EMA filtered."""

    def __init__(self, drones, alpha):
        self.p = np.array([d.position.ravel() for d in drones], dtype=float)
        self.v = np.array([d.velocity.ravel() for d in drones], dtype=float)
        self.alpha, self.t_next = alpha, 0.0

    def update(self, t, drones):
        if t + 1e-9 >= self.t_next:
            pos = np.array([d.position.ravel() for d in drones])
            self.v = self.alpha * (pos - self.p) / OPTI_PERIOD + (1 - self.alpha) * self.v
            self.p = pos
            self.t_next += OPTI_PERIOD


def set_desired(drones, sp, sp_v, sp_a):
    for k, d in enumerate(drones):
        d.desired.position = sp[k].reshape(3, 1).copy()
        d.desired.velocity = sp_v[k].reshape(3, 1).copy()
        d.desired.acceleration = sp_a[k].reshape(3, 1).copy()


def simulate(folder, ids, periods, T):
    """
    Dry run (no log): only the Formation phase of the flight code of
    `folder`, for T seconds, with the simulated drones `ids`, starting still
    on the slots the flight flies them to before it (converged formation,
    then MPC-settled slots). With a target drone (TARGET_DRONE_ID of the
    folder, one of ids) it flies the target path and the formation follows
    its simulated OptiTrack state.
    """
    fc = FlightCode(folder, ids, periods)
    S, ch, ti, n_all = fc.S, fc.ch, fc.ti, fc.n_all
    formation = fc.new_formation()
    goals = formation.desired_positions.copy()
    goals = formation.assign_slots(np.column_stack([goals[:, :2], np.full(fc.nD, S.TAKEOFF_HEIGHT)]))
    start = np.zeros((n_all, 3))
    if fc.has_target:
        start[ti] = fc.target_start()
        start[ch] = fc.mpc.settle(goals, goals, target=start[ti])
    else:
        start[ch] = fc.mpc.settle(goals, goals) if fc.use_mpc else goals
    if fc.use_mpc:
        from mpc import min_distance
        print(f"Slots: min distance between slots {min_distance(start):.2f} m")

    plant, drones = make_plant(n_all, fc.setpoint_mode)
    for k in range(n_all):
        drones[k].position = start[k].reshape(3, 1).copy()
        drones[k].desired.position = drones[k].position.copy()
    meas = Measurement(drones, getattr(S, "VEL_FILTER_ALPHA", 0.3))
    sp, sp_v, sp_a = np.array(start, dtype=float), np.zeros((n_all, 3)), np.zeros((n_all, 3))
    fc.start(0.0)

    log = []
    for n in range(int(round(T / PLANT_PERIOD)) + 1):
        t = n * PLANT_PERIOD
        meas.update(t, drones)
        fc.step(t, meas.p, meas.v, sp, sp_v, sp_a)
        set_desired(drones, sp, sp_v, sp_a)
        target = formation.get_target()
        for k, i in enumerate(ids):
            log.append([t, PHASE, i, *drones[k].position.ravel(), *sp[k], *target])
        for k in range(n_all):
            plant.actuate(drones[k])

    cols = ["time_s", "flight_state", "id", *POS_COLS, *SP_COLS, *TARGET_COLS]
    return pd.DataFrame(log, columns=cols)


def smoothstep(f):
    return f * f * (3.0 - 2.0 * f)


def simulate_flight(folder, ids, homes, phases=None, periods=None, meta=None, gains=None, drone_model=None,
                    history=None):
    """
    The whole flight of the flight code of `folder` (swarm_MPC_voronoi:
    main.py), from the drones at rest on the floor at their takeoff points
    `homes` (rows in the order of ids), with the commands of the flight code:

      Take off   ramp_to above the takeoff point (TAKEOFF_HEIGHT, TAKEOFF_TIME)
      Approach   ramp_to the formation slots (target drone: the start of its
                 path), duration from the distance (APPROACH_SPEED)
      Formation  the formation threads, the MPC and the controllers (FlightCode)
      Return     ramp_to the takeoff points (Hungarian, RETURN_SPEED)
      Land       ramp_to LAND_HEIGHT in place (LAND_TIME)
      EOF        stop_flight: motors off

    Every ramp_to starts from the measured position, with smoothstep
    setpoints. phases: the start times of the phases in the log (from the
    take off start), so each phase starts when it did in the flight and its
    setpoints are spread over its logged duration (the setpoint loop of the
    flight runs slower than SETPOINT_PERIOD); a phase missing from the log
    (abort) is skipped. None: the nominal durations of the flight code.
    meta: <log>_meta.json (the formation goals and the approach slots flown).
    history: a dict, filled with the state of every step for the results
    .pkl (t, position, velocity, goal, target position / velocity, chasers,
    target drone).
    """
    fc = FlightCode(folder, ids, periods or {})
    S, ch, ti, n_all = fc.S, fc.ch, fc.ti, fc.n_all
    homes = np.asarray(homes, dtype=float)
    above = np.column_stack([homes[:, :2], np.full(n_all, S.TAKEOFF_HEIGHT)])   # above the takeoff points

    # Formation and slots as the flight code before arming (or as flown: <log>_meta.json)
    md = (meta or {}).get("drones", {})
    by_index = {d["chaser_index"]: d for d in md.values() if d.get("role") == "chaser" and "chaser_index" in d}
    logged = len(by_index) == fc.nD and all("formation_goal" in d and "slot" in d for d in by_index.values())
    goals_flown = np.array([by_index[j]["formation_goal"] for j in range(fc.nD)]) if logged else None
    formation = fc.new_formation(goals_flown)
    goals = formation.assign_slots(above[ch])
    target_start = fc.target_start()
    if logged:
        slots = np.array([by_index[j]["slot"] for j in range(fc.nD)], dtype=float)
    elif fc.use_mpc:
        slots = fc.mpc.settle(goals, goals, target=target_start if fc.has_target else None)
    else:
        slots = goals
    print("Approach slots: " + ("<log>_meta.json" if logged else "computed as the flight code"))

    # Phases
    if phases:
        seq = sorted(((name, t) for name, t in phases.items() if name in FLIGHT_PHASES), key=lambda p: p[1])
        t_eof = phases["EOF"]
    else:
        seq, t_eof = [(name, None) for name in FLIGHT_PHASES], None

    plant, drones = make_plant(n_all, fc.setpoint_mode, gains, drone_model)
    plant.start(drones, None)
    for k in range(n_all):
        drones[k].position = homes[k].reshape(3, 1).copy()
        drones[k].desired.position = drones[k].position.copy()
    meas = Measurement(drones, getattr(S, "VEL_FILTER_ALPHA", 0.3))
    sp, sp_v, sp_a = homes.copy(), np.zeros((n_all, 3)), np.zeros((n_all, 3))
    has_sp = False

    def begin(name, t):
        """Phase `name` starts at t: its ramp (start, goal, setpoint count) or the formation."""
        p = meas.p.copy()
        if name == "Take off":
            return p, above.copy(), S.TAKEOFF_TIME
        if name == "Approach":
            dest = np.zeros((n_all, 3))
            dest[ch] = slots
            if fc.has_target:
                dest[ti] = target_start
            dist = np.max(np.linalg.norm(dest - p, axis=1))
            return p, dest, max(1.5 * dist / S.APPROACH_SPEED, 2.0)
        if name == "Return":        # assign_land_spots
            from scipy.optimize import linear_sum_assignment
            cost = np.sum((p[:, None, :] - above[None, :, :]) ** 2, axis=2)
            _, home_of = linear_sum_assignment(cost)
            dest = above[home_of]
            dist = np.max(np.linalg.norm(p - dest, axis=1))
            return p, dest, max(1.5 * dist / S.RETURN_SPEED, 2.0)
        if name == "Land":
            return p, np.column_stack([p[:, :2], np.full(n_all, S.LAND_HEIGHT)]), S.LAND_TIME
        fc.start(t)                 # Formation
        return None

    log = []
    phase_i, name, ramp, t_phase, motors_off = -1, "", None, 0.0, False
    t, n = 0.0, 0
    while True:
        t = n * PLANT_PERIOD
        if t_eof is not None and t > t_eof + 1e-9:
            break
        meas.update(t, drones)

        # next phase: at its logged start, else when the current one is over
        if not motors_off:
            nxt = seq[phase_i + 1] if phase_i + 1 < len(seq) else None
            if nxt is None:
                # motors off at the logged end, else 1 s after the landing ramp (time.sleep(1.0))
                if t_eof is not None:
                    due = t + 1e-9 >= t_eof
                else:
                    due = ramp is None or (ramp["j"] >= ramp["n"] and t + 1e-9 >= t_phase + ramp["n"] * ramp["dt"] + 1.0)
                if due:
                    name, motors_off = "EOF", True
                    if t_eof is None:
                        t_eof = t
            else:
                if nxt[1] is not None:
                    due = t + 1e-9 >= nxt[1]
                elif phase_i < 0:
                    due = True
                elif ramp is not None:
                    due = ramp["j"] >= ramp["n"]
                else:
                    due = t + 1e-9 >= t_phase + S.FORMATION_TIME
                if due:
                    phase_i += 1
                    name, t_phase = nxt[0], t
                    r = begin(name, t)
                    ramp = None
                    if r is not None:
                        start, goal, duration = r
                        steps = max(1, int(duration / S.SETPOINT_PERIOD))
                        t_end = seq[phase_i + 1][1] if phase_i + 1 < len(seq) else t_eof
                        dt = (t_end - t) / steps if t_end is not None else S.SETPOINT_PERIOD
                        ramp = {"start": start, "goal": goal, "n": steps, "dt": dt, "j": 0}
                    sp_v[:] = 0.0
                    sp_a[:] = 0.0

        # setpoints of the current phase
        if not motors_off and name:
            if ramp is not None:
                while ramp["j"] < ramp["n"] and t + 1e-9 >= t_phase + ramp["j"] * ramp["dt"]:
                    f = smoothstep((ramp["j"] + 1) / ramp["n"])
                    sp[:] = ramp["start"] + (ramp["goal"] - ramp["start"]) * f
                    ramp["j"] += 1
            elif name == PHASE:
                fc.step(t, meas.p, meas.v, sp, sp_v, sp_a)
            has_sp = True
        set_desired(drones, sp, sp_v, sp_a)

        target = formation.get_target()
        for k, i in enumerate(ids):
            log.append([t, name, i, *drones[k].position.ravel(), *sp[k], *target])
        if history is not None:
            goal = sp.copy()                     # desired position: the setpoint, the formation goal in its phase
            if name == PHASE:
                goal[ch] = formation.get_goals()[0]
            tp, tv = formation.get_target_state()
            for key, v in (("t", t), ("position", [d.position.ravel().copy() for d in drones]),
                           ("velocity", [d.velocity.ravel().copy() for d in drones]), ("goal", goal),
                           ("target_position", tp), ("target_velocity", tv)):
                history.setdefault(key, []).append(np.array(v, dtype=float))

        for k in range(n_all):
            if motors_off:
                plant.actuate_passive(drones[k])
            elif has_sp:
                plant.actuate(drones[k])
            ground_contact(drones[k], homes[k, 2])
        n += 1
        if t_eof is None and n * PLANT_PERIOD > 600:
            raise RuntimeError("simulate_flight: no end of the flight after 600 s")

    if history is not None:
        for key in list(history):
            history[key] = np.array(history[key])
        history.update(chasers=list(ch), target=ti, sphere_radius=float(S.SPHERE_RADIUS))
    cols = ["time_s", "flight_state", "id", *POS_COLS, *SP_COLS, *TARGET_COLS]
    return pd.DataFrame(log, columns=cols)


def load_setpoint_stream(log_path, ids, real, setpoint_mode="full_state"):
    """
    The setpoints the flight sent, per drone: (t from the take off, p, v, a),
    and where they come from:
      - <log>_setpoints.csv (flights from 13 Nov): every setpoint sent, with
        the velocity and acceleration of the full-state ones;
      - else the setpoint sp_* of the trajectory log (the last one sent,
        sampled at 50 Hz): a new setpoint where it changes; in the Formation
        phase of a full-state flight the velocity / acceleration are the
        derivatives of that stream (the MPC plan's are not logged).
    """
    sp_path = log_path.with_name(log_path.stem + "_setpoints.csv")
    stream = {}
    if sp_path.exists():
        df = pd.read_csv(log_path, usecols=["time_s", "flight_state"], dtype={"flight_state": str})
        t0 = df[df["flight_state"].fillna("").str.strip() == "Take off"]["time_s"].min()
        sp = pd.read_csv(sp_path, dtype={"id": str})
        for i in ids:
            g = sp[sp["id"] == i].sort_values("time_s")
            stream[i] = (g["time_s"].to_numpy(float) - t0, g[SP_COLS].to_numpy(float),
                         g[["sp_vx", "sp_vy", "sp_vz"]].to_numpy(float), g[["sp_ax", "sp_ay", "sp_az"]].to_numpy(float))
        return stream, "<log>_setpoints.csv (every setpoint sent)"
    for i in ids:
        r = real[i]
        ok = np.all(np.isfinite(r["sp"]), axis=1)
        t, p, ph = r["t"][ok], r["sp"][ok], r["phase"][ok]
        new = np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]
        t, p, ph = t[new], p[new], ph[new]
        v, a = np.zeros_like(p), np.zeros_like(p)
        m = ph == PHASE
        if setpoint_mode == "full_state" and m.sum() > 2:
            v[m] = np.gradient(p[m], t[m], axis=0)
            a[m] = np.gradient(v[m], t[m], axis=0)
        stream[i] = (t, p, v, a)
    return stream, "the setpoints sampled in the trajectory log (no <log>_setpoints.csv)"


def complete_lost_drones(stream, ids, homes, phases, S):
    """
    A drone that the real flight lost (OptiTrack lost it, e.g. after a crash)
    gets no setpoints after the loss: the flight code stops commanding it
    (tracking timeout). So that its simulated flight does not hover at its
    last setpoint until the end, its setpoint stream is completed as the flight
    code would have flown it:
      - until Return: its last setpoint (hover);
      - Return: the rest of the smoothstep ramp to the takeoff point the other
        drones left free (assign_land_spots: one drone per takeoff point), at
        TAKEOFF_HEIGHT, continuous with its logged ramp, ending when the
        other drones' Return ramps end;
      - Land: the landing ramp to LAND_HEIGHT in place, timed as the other
        drones' (or the rest of it, if it was lost during the landing).
    The setpoints are position setpoints (zero velocity / acceleration), as
    those of the flight code in these phases. Returns the completed stream and
    {id: (loss time, takeoff point)} of the completed drones.
    """
    from scipy.optimize import linear_sum_assignment
    last = {i: stream[i][0][-1] if len(stream[i][0]) else -np.inf for i in ids}
    t_end = max(last.values())
    lost = [i for i in ids if last[i] < t_end - 1.0]
    t_ret, t_land = phases.get("Return"), phases.get("Land")
    if not lost or t_ret is None or t_land is None:
        if lost:
            print("Lost drones " + ", ".join(lost) + ": no Return / Land in the log, not completed")
        return stream, {}
    ok = [i for i in ids if i not in lost]
    homes = np.asarray(homes, dtype=float)
    period = float(getattr(S, "SETPOINT_PERIOD", 0.05))

    def ramp_end(t0, t1):
        """End of the ramps of the drones still flying in [t0, t1): their last setpoint there."""
        ends = [stream[i][0][(stream[i][0] >= t0) & (stream[i][0] < t1)] for i in ok]
        ends = [e[-1] for e in ends if len(e)]
        return max(ends) if ends else t1

    t_ret_end, t_land_end = ramp_end(t_ret, t_land), ramp_end(t_land, np.inf)

    # takeoff points: those reached by the other drones at the end of Return are taken
    def sp_before(i, t):
        ts, P = stream[i][0], stream[i][1]
        k = np.searchsorted(ts, t, side="right") - 1
        return P[max(k, 0)]
    taken = {int(np.argmin(np.linalg.norm(homes[:, :2] - sp_before(i, t_land)[:2], axis=1))) for i in ok}
    free = [h for h in range(len(homes)) if h not in taken]
    lost_sp = np.array([sp_before(i, np.inf) for i in lost])
    cost = np.linalg.norm(lost_sp[:, None, :2] - homes[None, free, :2], axis=2)
    rows, cols = linear_sum_assignment(cost)
    home_of = {lost[r]: free[c] for r, c in zip(rows, cols)}

    def smooth(t, t0, t1):
        return smoothstep(np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))

    done = {}
    for i in lost:
        if i not in home_of:
            print(f"CF{i}: lost by the flight, no free takeoff point: not completed")
            continue
        ts, P, V, A = stream[i]
        t_l, p_l = last[i], sp_before(i, np.inf).astype(float)
        above = np.array([*homes[home_of[i], :2], S.TAKEOFF_HEIGHT])
        new_t, new_p = [], []
        if t_l < t_ret_end:
            # hover until Return, then the rest of the Return ramp (continuous with the logged one)
            f_l = float(smooth(t_l, t_ret, t_ret_end))
            for t in np.arange(max(t_l + period, t_ret), t_ret_end + 1e-9, period):
                f = (smooth(t, t_ret, t_ret_end) - f_l) / max(1.0 - f_l, 1e-9)
                new_t.append(t)
                new_p.append(p_l + (above - p_l) * f)
            land_start, f_l = above, 0.0
        else:
            land_start = p_l                      # lost during the landing: the rest of it
            f_l = float(smooth(t_l, t_land, t_land_end))
        land_goal = np.array([*land_start[:2], S.LAND_HEIGHT])
        for t in np.arange(max(t_l + period, t_land), t_land_end + 1e-9, period):
            f = (smooth(t, t_land, t_land_end) - f_l) / max(1.0 - f_l, 1e-9)
            new_t.append(t)
            new_p.append(land_start + (land_goal - land_start) * f)
        if not new_t:
            continue
        new_p = np.array(new_p)
        z = np.zeros_like(new_p)
        stream[i] = (np.r_[ts, new_t], np.vstack([P, new_p]), np.vstack([V, z]), np.vstack([A, z]))
        done[i] = (t_l, homes[home_of[i]])
        print(f"CF{i}: no setpoints after {t_l:.2f} s (lost by the flight): completed in the simulation, "
              f"Return to its takeoff point ({above[0]:.2f}, {above[1]:.2f}) and landing")
    return stream, done


def simulate_logged(ids, homes, stream, phases, real, setpoint_mode="full_state", target_index=None, gains=None,
                    drone_model=None, sphere_radius=0.8, history=None):
    """
    The real flight replayed on the simulated Crazyflies: each drone starts
    at rest on the floor at its takeoff point and gets the setpoints the
    flight sent (stream, load_setpoint_stream), at the times they were sent,
    until the motors are stopped (EOF). The decisions of the flight code
    (formation, MPC, landing spots) are the flown ones, so the difference
    to the real flight is the drone model. Same outputs as simulate_flight.
    """
    n_all = len(ids)
    homes = np.asarray(homes, dtype=float)
    plant, drones = make_plant(n_all, setpoint_mode, gains, drone_model)
    plant.start(drones, None)
    for k in range(n_all):
        drones[k].position = homes[k].reshape(3, 1).copy()
        drones[k].desired.position = drones[k].position.copy()
    sp = homes.copy()
    ptr, has_sp = [0] * n_all, [False] * n_all
    order = sorted(((name, t) for name, t in phases.items() if name in FLIGHT_PHASES), key=lambda p: p[1])
    t_eof = phases["EOF"]
    r0 = real[ids[0]]
    ok = np.all(np.isfinite(r0["target"]), axis=1)
    target_at = (lambda t: np.array([np.interp(t, r0["t"][ok], r0["target"][ok, a]) for a in range(3)])) \
        if ok.any() else (lambda t: np.full(3, np.nan))
    log = []
    for n in range(int(round(t_eof / PLANT_PERIOD)) + 1):
        t = n * PLANT_PERIOD
        name = "EOF" if t >= t_eof else max((p for p in order if p[1] <= t + 1e-9), key=lambda p: p[1],
                                             default=("", 0.0))[0]
        for k, i in enumerate(ids):
            ts, P, V, A = stream[i]
            while ptr[k] < len(ts) and ts[ptr[k]] <= t + 1e-9:
                j = ptr[k]
                d = drones[k].desired
                d.position, d.velocity, d.acceleration = (P[j].reshape(3, 1).copy(), V[j].reshape(3, 1).copy(),
                                                          A[j].reshape(3, 1).copy())
                sp[k] = P[j]
                has_sp[k] = True
                ptr[k] += 1
        target = target_at(t)
        for k, i in enumerate(ids):
            log.append([t, name, i, *drones[k].position.ravel(), *sp[k], *target])
        if history is not None:
            for key, v in (("t", t), ("position", [d.position.ravel().copy() for d in drones]),
                           ("velocity", [d.velocity.ravel().copy() for d in drones]), ("goal", sp.copy()),
                           ("target_position", target), ("target_velocity", np.zeros(3))):
                history.setdefault(key, []).append(np.array(v, dtype=float))
        for k in range(n_all):
            if t >= t_eof:
                plant.actuate_passive(drones[k])         # stop_flight: motors off
            elif has_sp[k]:
                plant.actuate(drones[k])
            ground_contact(drones[k], homes[k, 2])
    if history is not None:
        for key in list(history):
            history[key] = np.array(history[key])
        tp = history["target_position"]
        history["target_velocity"] = np.gradient(tp, PLANT_PERIOD, axis=0) if len(tp) > 1 else tp * 0
        history.update(chasers=[k for k in range(n_all) if k != target_index], target=target_index,
                       sphere_radius=float(sphere_radius))
    cols = ["time_s", "flight_state", "id", *POS_COLS, *SP_COLS, *TARGET_COLS]
    return pd.DataFrame(log, columns=cols)


def save_results_pkl(path, history, ids, phases, real, flight):
    """
    The simulated flight as a results .pkl of this simulation (the chasers;
    the target: virtual, or the target drone), so animation_display.py can
    redraw / rewatch it: SimulationData, plus "sim" (scene arena, the
    flight's trajectory, dTi, ...), "phases", "flight" (log, run code),
    "drone_ids" and "real_position_history" (the logged real chasers on the
    same times).
    """
    import pickle
    import matplotlib.pyplot as plt
    ch, ti = history["chasers"], history["target"]
    t, P, V = history["t"], history["position"], history["velocity"]            # (N,), (N, n_all, 3)
    dt, N, nD = PLANT_PERIOD, len(t), len(ch)
    pos = P[:, ch].transpose(1, 2, 0)                                           # (nD, 3, N)
    vel = V[:, ch].transpose(1, 2, 0)
    acc = np.gradient(vel, dt, axis=2)
    tp, tv = history["target_position"], history["target_velocity"]
    if ti is not None:                                                         # target drone: its own state
        tp, tv = P[:, ti], V[:, ti]
    target_true = np.zeros((24, N))
    target_true[0:3], target_true[3:6] = tp.T, tv.T
    target_true[6:15] = np.eye(3).reshape(9, 1)
    real_pos = np.full((nD, 3, N), np.nan)
    for j, k in enumerate(ch):
        r = real[ids[k]]
        ok = np.all(np.isfinite(r["p"]), axis=1)
        for a in range(3):
            real_pos[j, a] = np.interp(t, r["t"][ok], r["p"][ok, a], left=np.nan, right=np.nan)
    data = {
        "position_history": pos,
        "velocity_history": vel,
        "acceleration_history": acc,
        "target_true_hist": target_true,
        "radial_history": np.linalg.norm(pos - tp.T[None], axis=1),
        "removed_initial_pos": False,
        "p_d_hist": history["goal"][:, ch].transpose(1, 2, 0),
        "perceived_target_hist": np.repeat(np.vstack([tp.T, tv.T])[None], nD, axis=0),
        "colors": plt.cm.tab10(np.linspace(0, 1, nD)),
        "active_ids": list(range(nD)),
        "numAgents_history": [nD] * N,
        "sim": {"scene": "arena", "traj": int(flight.get("target_traj") or 2), "nD": nD, "Tend": float(t[-1]),
                "dTi": dt, "Nsim": N, "nD_drop_time": -1, "sphere_radius": history["sphere_radius"],
                "arena_wall_x": 0.0},
        "phases": phases,
        "flight": flight,
        "drone_ids": [ids[k] for k in ch],
        "real_position_history": real_pos,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Simulation results (.pkl, animation_display.py) saved to {path}")


# ============================================================
# COMPARISON
# ============================================================

def sim_tracks(sim_df, ids, t):
    """Simulated positions / setpoints of every drone at the log times t[i]."""
    out = {}
    for i in ids:
        g = sim_df[sim_df["id"] == i]
        ts = g["time_s"].to_numpy(float)
        out[i] = {c: np.column_stack([np.interp(t[i], ts, g[k].to_numpy(float)) for k in cols])
                  for c, cols in (("p", POS_COLS), ("sp", SP_COLS), ("target", TARGET_COLS))}
    return out


def metrics(tracks, ids):
    """Per drone: distance to the target; pair: inter-drone distance (horizontal and vertical)."""
    m = {i: np.linalg.norm(tracks[i]["p"] - tracks[i]["target"], axis=1) for i in ids}
    pairs = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pa, pb = tracks[ids[a]]["p"], tracks[ids[b]]["p"]
            n = min(len(pa), len(pb))
            pairs[(ids[a], ids[b])] = np.linalg.norm(pa[:n] - pb[:n], axis=1)
    return m, pairs


def plot_dry_run(ids, sim, title, save_dir, show=True, target_id=None):
    """Simulated flight only: top view, positions, distance to the target, inter-drone distances."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    t = sim[ids[0]]["t"]
    tgt = sim[ids[0]]["target"]
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(tgt[:, 0], tgt[:, 1], "k", lw=1.2, label="target")
    for k, i in enumerate(ids):
        ax.plot(sim[i]["p"][:, 0], sim[i]["p"][:, 1], color=colors(k), lw=1.4,
                label=f"drone {i}" + (" (target)" if i == target_id else ""))
        ax.plot(*sim[i]["p"][0, :2], "o", color=colors(k))
    ax.add_patch(plt.Rectangle((-2.5, -2.5), 5, 5, fill=False, ls="--", color="gray"))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("top view (o: start)")

    r, _ = metrics(sim, ids)
    axes = [fig.add_subplot(3, 2, 2 * j + 2) for j in range(3)]
    for k, i in enumerate(ids):
        axes[0].plot(t, sim[i]["p"][:, 2], color=colors(k))
        axes[1].plot(t, r[i], color=colors(k))
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            d = np.linalg.norm(sim[ids[a]]["p"] - sim[ids[b]]["p"], axis=1)
            axes[2].plot(t, d, lw=1.2, label=f"{ids[a]}-{ids[b]}")
    axes[0].set_ylabel("z [m]")
    axes[1].set_ylabel("distance to\ntarget [m]")
    axes[2].set_ylabel("inter-drone\ndistance [m]")
    r_safe = getattr(_SAFETY, "radius", None)
    if r_safe:
        axes[2].axhline(r_safe, color="r", lw=0.8, ls="--", label="safety radius")
    axes[2].legend(fontsize=7, ncol=3)
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)

    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(save_dir / f"dry_run.{ext}", bbox_inches="tight", dpi=150)
    print(f"Figures saved to {save_dir}")
    if show:
        plt.show()


def print_dry_run(ids, sim, target_id=None):
    r, d = metrics(sim, ids)
    print("\nDry run (Formation phase):")
    for i in ids:
        if i == target_id:
            e = np.linalg.norm(sim[i]["p"] - sim[i]["sp"], axis=1)
            print(f"  drone {i} (target): path tracking error mean {e.mean():.3f} m, max {e.max():.3f} m")
            continue
        print(f"  drone {i}: distance to target mean {r[i].mean():.2f} m (last 5 s {r[i][-250:].mean():.2f} m), "
              f"max speed {np.linalg.norm(np.diff(sim[i]['p'], axis=0), axis=1).max() / PLANT_PERIOD:.2f} m/s")
    for (a, b), dist in d.items():
        print(f"  pair {a}-{b}: min distance {dist.min():.2f} m")


def newest_log(flight_code):
    """
    Flight log of the newest run of `flight_code`
    (Results/<flight_code>[_<sim|real>_target]_trajectory_<date>/trajectory_<date>.csv).
    """
    runs = [r for r in LOGS_DIR.glob(f"{flight_code}*_trajectory_*/")
            if re.fullmatch(rf"{re.escape(flight_code)}(_(sim|real)_target)?_trajectory_\d{{8}}_\d{{6}}", r.name)]
    for run in sorted(runs, key=lambda r: r.name[-15:], reverse=True):
        log = run / f"trajectory_{run.name[-15:]}.csv"
        if log.exists():
            return log
    raise SystemExit(f"No run of {flight_code} in {LOGS_DIR}")


DRY_RUN_TARGET_ID = 4      # target drone of a --target real dry run when the flight code has REAL_TARGET_DRONE_ID = -1


def logged_target_id(meta):
    """Rigid-body ID of the target drone of a flight (<log>_meta.json), None if not logged."""
    if not meta:
        return None
    if meta.get("target_drone_id") is not None:
        return int(meta["target_drone_id"])
    ids = [i for i, d in meta.get("drones", {}).items() if d.get("role") == "target"]
    return int(ids[0]) if ids else None


def resolve_flight_code(name, target=None, log_path=None, meta=None):
    """
    (folder, run code) of a flight code name: a folder of Crazy_Flie/Optitrack,
    or <folder>_<sim|real>_target for a folder with the target switch
    (swarm_MPC_voronoi: REAL_TARGET_DRONE_ID in its Structs.py, -1 = simulated
    target). The target mode comes from <log>_meta.json, the run folder of
    log_path, `target`, or the name; it is set in CF_TARGET (and, for a real
    target, the logged target drone in CF_TARGET_DRONE_ID) before the flight
    code is imported. With none of them: REAL_TARGET_DRONE_ID of the flight code.
    """
    folder, mode = FLIGHT_DIR / name, None
    if not folder.is_dir():
        m = re.fullmatch(r"(.+)_(sim|real)_target", name)
        if not (m and (FLIGHT_DIR / m.group(1)).is_dir()):
            raise SystemExit(f"No flight code {name} in {FLIGHT_DIR}")
        folder, mode = FLIGHT_DIR / m.group(1), m.group(2)
    if 'CF_TARGET"' not in (folder / "Structs.py").read_text():
        return folder.resolve(), folder.name            # no switch: the folder name
    from_log = None
    if meta and meta.get("target_mode") in ("sim", "real"):
        from_log = meta["target_mode"]
    elif log_path is not None:
        m = re.match(rf"{re.escape(folder.name)}_(sim|real)_target_trajectory_", Path(log_path).parent.name)
        from_log = m.group(1) if m else None
    if from_log and target and target != from_log:
        raise SystemExit(f"--target {target}, but the run was flown with the {from_log} target")
    mode = from_log or target or mode
    os.environ.pop("CF_TARGET", None)
    os.environ.pop("CF_TARGET_DRONE_ID", None)
    if mode is not None:
        os.environ["CF_TARGET"] = mode
        tid = logged_target_id(meta)
        if mode == "real" and tid is None:
            # no log (dry run): REAL_TARGET_DRONE_ID, else drone DRY_RUN_TARGET_ID (no real drone needed)
            m = re.search(r"^REAL_TARGET_DRONE_ID\s*=\s*(-?\d+)", (folder / "Structs.py").read_text(), re.M)
            tid = int(m.group(1)) if m else -1
            if tid == -1:
                tid = DRY_RUN_TARGET_ID
                print(f"Note: REAL_TARGET_DRONE_ID = -1 in the flight code: the simulated target drone is {tid}")
        if mode == "real":
            os.environ["CF_TARGET_DRONE_ID"] = str(tid)
    S, *_ = import_flight_code(folder)              # REAL_TARGET_DRONE_ID when no mode was given
    return folder.resolve(), f"{folder.name}_{S.TARGET_MODE}_target"


# ============================================================
# FLIGHT SETUP (<log>_meta.json)
# ============================================================

def load_meta(log_path):
    """<log>_meta.json of a run (None for runs from before it was logged)."""
    path = Path(log_path).with_name(Path(log_path).stem + "_meta.json")
    return json.loads(path.read_text()) if path.exists() else None


def apply_meta(S, meta):
    """The settings the drones flew with (<log>_meta.json) in the flight code module S (Structs)."""
    for k, v in meta.get("mpc", {}).items():
        setattr(S, k, v)
    f = meta.get("formation", {})
    for key, name in (("sphere_radius", "SPHERE_RADIUS"), ("voronoi_period", "VORONOI_PERIOD"),
                      ("formation_time", "FORMATION_TIME"), ("cvt_preconverge_iters", "CVT_PRECONVERGE_ITERS")):
        if key in f:
            setattr(S, name, f[key])
    if "mesh_points" in f and f["mesh_points"] != S.MESH_POINTS:
        print(f"Note: flown with {f['mesh_points']} mesh points, the flight code now has {S.MESH_POINTS}")
    tg = meta.get("target", {})
    for key, name in (("radius", "TARGET_TRAJ_RADIUS"), ("angular_speed", "TARGET_ANGULAR_SPEED"),
                      ("height", "TARGET_HEIGHT")):
        if key in tg:
            setattr(S, name, tg[key])
    for k, v in {**meta.get("timing_s", {}), **meta.get("flight", {})}.items():
        if hasattr(S, k):
            setattr(S, k, v)


MEL_GAIN_KEYS = ("ctrlMel.kp_xy", "ctrlMel.kp_z", "ctrlMel.kd_xy", "ctrlMel.kd_z")
# also used by the plant when logged: feedforward mass and the integral term
MEL_EXTRA_KEYS = ("ctrlMel.mass", "ctrlMel.ki_xy", "ctrlMel.ki_z", "ctrlMel.i_range_xy", "ctrlMel.i_range_z")


def firmware_gains(meta):
    """
    Mellinger values of the plant: those read back from the firmware (first
    drone that has the position gains), else the ones the flight code set
    (mellinger_params_set, as the Fibonacci simulation); None if neither is
    logged. Position gains, and the feedforward mass and integral term
    (MEL_EXTRA_KEYS) when they are logged (else the plant keeps
    experiment_arena.CRAZYFLIE_ONBOARD).
    """
    keys = MEL_GAIN_KEYS
    fws = [d.get("firmware", {}) for d in (meta or {}).get("drones", {}).values()]
    fws = [f for f in fws if all(f.get(k) is not None for k in keys)]
    if not fws:
        set_ = (meta or {}).get("mellinger_params_set") or {}
        if not all(set_.get(k) is not None for k in keys):
            return None
        print("Note: no firmware read-back in <log>_meta.json, plant gains = mellinger_params_set")
        fws = [{k: float(v) for k, v in set_.items() if v is not None}]
    g = {k: fws[0][k] for k in (*keys, *MEL_EXTRA_KEYS) if fws[0].get(k) is not None}
    if any(any(f.get(k) is not None and abs(f[k] - g[k]) > 1e-9 for k in g) for f in fws[1:]):
        print("Note: the drones flew with different Mellinger values; the plant uses those of the first one")
    return g


# ============================================================
# RUNS
# ============================================================

def dry_run(folder, run_code, args):
    S, *_ = import_flight_code(folder)
    _SAFETY.radius = getattr(S, "MPC_SAFETY_RADIUS", None)
    target_id = getattr(S, "TARGET_DRONE_ID", None) if getattr(S, "TARGET_URI", None) is not None else None
    ids = [str(i + 1) for i in range(args.drones + 4) if target_id is None or i + 1 != target_id][:args.drones]
    if target_id is not None:
        ids.append(str(target_id))
    periods = {"setpoint": S.SETPOINT_PERIOD, "formation": S.VORONOI_PERIOD}
    T = S.FORMATION_TIME
    print(f"Dry run: trajectory {S.TARGET_TRAJ}, {args.drones} drones"
          f"{f' + target drone {target_id}' if target_id is not None else ''}, {T:.1f} s, "
          f"setpoint {S.SETPOINT_PERIOD * 1e3:.0f} ms, formation {S.VORONOI_PERIOD * 1e3:.0f} ms")
    sim_df = simulate(folder, ids, periods, T)
    save_dir = LOGS_DIR / (f"{run_code}_dry_run_traj{S.TARGET_TRAJ}_{args.drones}drones"
                           + ("_target_drone" if target_id is not None and run_code == folder.name else "")
                           + datetime.now().strftime("_%Y%m%d_%H%M%S"))
    save_dir.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(save_dir / "sim.csv", index=False)
    t = np.arange(0.0, T, 0.02)
    sim = sim_tracks(sim_df, ids, {i: t for i in ids})
    for i in ids:
        sim[i]["t"] = t
    tid = None if target_id is None else str(target_id)
    print_dry_run(ids, sim, tid)
    if not args.no_plot:
        plot_dry_run(ids, sim, f"{run_code}: dry run, trajectory {S.TARGET_TRAJ}, {args.drones} drones",
                     save_dir, show=not args.no_show, target_id=tid)


def describe_meta(meta):
    """Scene, drone and trajectory of the flight (<log>_meta.json); the drone model for the plant."""
    from experiment_arena import DRONE_MODEL
    scene = meta.get("scene", "arena")
    drone = meta.get("drone", {})
    model = drone.get("model", DRONE_MODEL)
    traj = meta.get("target_traj")
    name = meta.get("target_traj_name")
    print(f"Scene {scene}, drone {drone.get('type', 'Crazyflie')} ({model}), "
          f"trajectory {traj}{f' ({name})' if name else ''}, target {meta.get('target_mode', '?')}")
    if scene != "arena":
        print(f"Note: scene {scene!r}; the replay simulates the OptiTrack arena")
    return model


def replay(folder, run_code, log_path, meta, args):
    S, *_ = import_flight_code(folder)
    gains = drone_model = None
    if meta:
        apply_meta(S, meta)
        drone_model = describe_meta(meta)
        gains = firmware_gains(meta)
        print("Flight setup: <log>_meta.json" + (", plant gains " + ", ".join(
            f"{k[8:]} {v:g}" for k, v in gains.items()) if gains else ""))
    else:
        print("No <log>_meta.json (flight from before 13 Nov): the settings of the flight code now")
    print(f"Flight code: {run_code}\nLog: {log_path}")

    ids, homes, phases, real, periods = load_flight(log_path, meta)
    print("Drones " + ", ".join(ids) + "; phases: " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))
    if periods["setpoint"]:
        print(f"Measured in the {PHASE} phase: setpoint {periods['setpoint'] * 1e3:.0f} ms, "
              f"formation {periods['formation'] * 1e3:.0f} ms")

    history = {}
    if args.rerun:
        # the flight code re-run (formation, MPC, landing spots): the algorithm against the flight
        print("Replay: the flight code re-run from the takeoff points (--rerun)")
        sim_df = simulate_flight(folder, ids, homes, phases, periods, meta, gains, drone_model, history)
    else:
        # the logged setpoints (as the geometric Fibonacci simulation): the drone model against the flight
        setpoint_mode = getattr(S, "SETPOINT_MODE", "position")
        stream, source = load_setpoint_stream(log_path, ids, real, setpoint_mode)
        print(f"Replay: the setpoints of the flight, from {source}")
        if not getattr(args, "keep_lost", False):
            stream, _ = complete_lost_drones(stream, ids, homes, phases, S)
        target_id = str(S.TARGET_DRONE_ID) if getattr(S, "TARGET_URI", None) is not None else None
        sim_df = simulate_logged(ids, homes, stream, phases, real, setpoint_mode,
                                 ids.index(target_id) if target_id in ids else None, gains, drone_model,
                                 S.SPHERE_RADIUS, history)
    out = log_path.with_name(log_path.stem + "_sim.csv")
    sim_df.to_csv(out, index=False)
    print(f"Simulated flight saved to {out}")
    # the simulated flight to rewatch later (python animation_display.py: the newest .pkl)
    save_results_pkl(log_path.parent / "simulation" / "simulation_results.pkl", history, ids, phases, real,
                     {"log": str(log_path), "run_code": run_code,
                      "target_traj": getattr(S, "TARGET_TRAJ", None), "target_mode": getattr(S, "TARGET_MODE", None)})

    # real vs sim (compare_real.py, as the geometric Fibonacci simulation)
    from compare_real import compare
    target_id = str(S.TARGET_DRONE_ID) if getattr(S, "TARGET_URI", None) is not None else None
    compare(ids, real, sim_df, phases, log_path.parent / "comparison", f"{run_code} {log_path.stem}",
            S.SPHERE_RADIUS, target_id, meta, show=not args.no_show, plot=not args.no_plot)


def run(args):
    """
    args: folder (flight code), target, log (None: newest run), dry_run,
    drones, traj, no_show, no_plot.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    log_path = meta = None
    if not args.dry_run:
        code = args.folder
        if args.target and not re.search(r"_(sim|real)_target$", code) and (FLIGHT_DIR / code).is_dir() \
                and 'CF_TARGET"' in (FLIGHT_DIR / code / "Structs.py").read_text():
            code = f"{code}_{args.target}_target"
        log_path = Path(args.log).resolve() if args.log else newest_log(code)
        meta = load_meta(log_path)
    if args.traj is not None:
        os.environ["CF_TARGET_TRAJ"] = str(args.traj)
    elif meta and meta.get("target_traj") is not None:
        os.environ["CF_TARGET_TRAJ"] = str(meta["target_traj"])
    folder, run_code = resolve_flight_code(args.folder, args.target, log_path, meta)
    if args.dry_run:
        dry_run(folder, run_code, args)
    else:
        replay(folder, run_code, log_path, meta, args)


def add_arguments(parser):
    parser.add_argument("--target", choices=("sim", "real"),
                        help="target mode of a flight code with the switch (default: the one of the log, "
                             "else REAL_TARGET_DRONE_ID of the flight code, -1 = sim)")
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    parser.add_argument("--rerun", action="store_true",
                        help="replay: re-run the flight code (formation, MPC, landing spots) from the takeoff points "
                             "instead of feeding the drones the setpoints of the flight")
    parser.add_argument("--no-plot", action="store_true", help="no figures")
    parser.add_argument("--keep-lost", action="store_true",
                        help="replay: a drone the flight lost (no setpoints after the loss) keeps its last setpoint; "
                             "by default its Return and landing are completed as the flight code would fly them")
    parser.add_argument("--drones", type=int, default=3, help="dry run: number of chasers (plus the target drone, if any)")
    parser.add_argument("--traj", type=int, help="trajectory: of the flight code with --IRL_log / --arena "
                        "(TARGET_TRAJ, default: the one of the log, else Structs.py), else of the simulation "
                        "(1-6, in the scene of --scene)")


def main():
    """Crazy_Flie/Optitrack/flight_sim.py: any flight code folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="flight code: swarm folder (e.g. swarm_voronoi_sim_target), or "
                                       "<folder>_<sim|real>_target for a folder with the target switch")
    parser.add_argument("--log", help="flight log (default: newest run of <folder> in the repository Results folder)")
    parser.add_argument("--dry-run", action="store_true", help="no log: simulate the flight code from its slots")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
