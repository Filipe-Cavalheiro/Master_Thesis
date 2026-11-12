"""
The flight code Crazy_Flie/Optitrack/swarm_geometric_Fibonacci_sim_target, for
its simulation (main.py): its constants and target trajectory, its flight
log (+ the flight setup it logs, <log>_meta.json / <log>_setpoints.csv), the
setpoint schedule of its phases and its ramps, and the initial guidance state.
"""

import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())   # repository root
FLIGHT_CODE = HERE.name         # same name as its flight code, Crazy_Flie/Optitrack/<name>
FLIGHT_DIR = REPO / "Crazy_Flie" / "Optitrack" / FLIGHT_CODE
LOGS_DIR = REPO / "Results"     # one folder per run: Results/<flight code>_trajectory_<date>/
OLD_PREFIX = "swarm_Fibonacci_sim_target"       # runs from before the flight code was renamed

MAIN = "Main Controller"
RETURN = "Return to (local) origin"
PHASES = ("Take off", MAIN, RETURN, "Land")
PAUSE_BEFORE_RETURN = 0.3   # s, time.sleep(0.3) in the finally block of main.py, before the return
PAUSE_AFTER_LAND = 1.0      # s, time.sleep(1.0) between the landing and "EOF"
LAND_Z = 0.0                # m, cf_funcs.land: ramp to (hx, hy, 0.0)
POS_COLS = ["opti_x", "opti_y", "opti_z"]
PHASE_COLORS = {"Take off": "#cfe8ff", MAIN: "#d9f2d0", RETURN: "#f3e1f7", "Land": "#ffe0c7"}
SHORT = {"Take off": "Take off", MAIN: "Main", RETURN: "Return", "Land": "Land"}


# =============================================================================
# Constants and target trajectory of the flight code
# =============================================================================

def flight_constants(folder=FLIGHT_DIR):
    """
    Constants of the flight code, without importing it (it needs cflib / natnet):
    module-level ones of Structs.py, the self.<name> = <literal> of Param.__init__,
    and the target trajectory (traj=...) of the Param(...) call in main.py.
    """
    mod = {"TAKEOFF_TIME": 3.0, "RETURN_TIME": 3.0, "LAND_TIME": 4.0, "SETPOINT_PERIOD": 0.05}
    par = {"Tend": 30.0, "dTi": 0.1, "sphere_radius": 0.8, "target_traj_radius": 1.5,
           "default_speed": 0.5, "default_height": 1.3, "k_rad": 0.3, "k_geometric": 5.0, "k_apf": 1.0}
    found = {}
    try:
        tree = ast.parse((folder / "Structs.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in mod:
                found[node.targets[0].id] = _literal(node.value)
            if isinstance(node, ast.ClassDef) and node.name == "Param":
                for f in node.body:
                    if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                        for a in ast.walk(f):
                            if isinstance(a, ast.Assign) and len(a.targets) == 1 \
                                    and isinstance(a.targets[0], ast.Attribute) and a.targets[0].attr in par:
                                found[a.targets[0].attr] = _literal(a.value)
    except OSError:
        print(f"Note: {folder / 'Structs.py'} not found, nominal constants used")
    found = {k: v for k, v in found.items() if v is not None}
    traj = 2
    try:
        for node in ast.walk(ast.parse((folder / "main.py").read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Param":
                for kw in node.keywords:
                    if kw.arg == "traj":
                        traj = int(ast.literal_eval(kw.value))
    except (OSError, ValueError):
        pass
    want = {**mod, **par}
    missing = [k for k in want if k not in found]
    if missing:
        print("Note: not in the flight code, defaults used: " + ", ".join(f"{k}={want[k]}" for k in missing))
    return SimpleNamespace(**{k: found.get(k, v) for k, v in want.items()}, traj=traj)


def _literal(node):
    try:
        return float(ast.literal_eval(node))
    except (ValueError, TypeError):
        return None


def flight_target_traj(c):
    """Structs.Param._make_traj of the flight code, as param.traj.ref_target_traj ({"pos", "vel"}: t -> (3, 1))."""
    if c.traj == 1:
        return {"pos": lambda t: np.array([[0.0], [0.0], [5.0]]), "vel": lambda t: np.zeros((3, 1))}
    if c.traj == 2:
        w, R, h = c.default_speed, c.target_traj_radius, c.default_height
        return {"pos": lambda t: np.array([[R * np.sin(w * t)], [R * np.cos(w * t)], [h]]),
                "vel": lambda t: np.array([[R * w * np.cos(w * t)], [-R * w * np.sin(w * t)], [0.0]])}
    if c.traj == 3:
        return {"pos": lambda t: np.array([[5 * np.sin(t)], [5 * np.sin(2 * t)], [5.0]]),
                "vel": lambda t: np.array([[5 * np.cos(t)], [10 * np.cos(2 * t)], [0.0]])}
    if c.traj == 4:
        return {"pos": lambda t: np.array([[5 * np.sin(t)], [5 * np.sin(2 * t)], [5 + 0.2 * np.cos(2 * t)]]),
                "vel": lambda t: np.array([[5 * np.cos(t)], [10 * np.cos(2 * t)], [-0.4 * np.sin(2 * t)]])}
    raise ValueError(f"Unknown trajectory {c.traj}")


# =============================================================================
# Param of this folder (Structs.py) set up for the flight
# =============================================================================

def configure_param(param, c, n_drones, t_end, dt, log_path=None):
    """
    param (Structs.Param) for the flight: the plant steps dt, the guidance
    gains, sphere radius and target trajectory of the flight code.
    """
    param.sim.nD = n_drones
    param.sim.dTi = dt
    param.sim.Nsim = int(round(t_end / dt)) + 1
    param.sim.Tend = (param.sim.Nsim - 1) * dt
    param.sim.t = np.arange(param.sim.Nsim) * dt
    param.sim.sphere_radius = c.sphere_radius
    param.sim.real_trajectory_csv = Path(log_path) if log_path else None
    param.k_rad, param.k_geometric, param.k_apf = c.k_rad, c.k_geometric, c.k_apf
    param.traj.ref_target_traj = flight_target_traj(c)
    param.traj.initial_target_pos = param.traj.ref_target_traj["pos"](0)
    return param


def guidance_param(param, c):
    """
    The guidance functions of helper_funcs integrate with param.sim.dTi: the
    flight code integrates once per controller tick with its dTi (0.1 s), not
    with the plant step. Same param, with sim.dTi = the flight's dTi.
    """
    gparam = copy.copy(param)
    gparam.sim = copy.copy(param.sim)
    gparam.sim.dTi = c.dTi
    return gparam


# =============================================================================
# Real flight log, and the flight setup logged next to it (from 12 Nov)
# =============================================================================

def load_log(path):
    """Per drone: time, OptiTrack position; phase start times of the log."""
    df = pd.read_csv(path, dtype={"id": str, "flight_state": str})
    df["flight_state"] = df["flight_state"].fillna("").str.strip()
    df["id"] = df["id"].str.replace(r"\.0$", "", regex=True)
    ids = sorted(df["id"].dropna().unique(), key=lambda s: (len(s), s))
    phases = {}
    for name in (*PHASES, "EOF"):
        rows = df[df["flight_state"] == name]
        if len(rows):
            phases[name] = float(rows["time_s"].min())
    if "Take off" not in phases:
        raise ValueError(f"No 'Take off' phase in {path}")
    t_end = float(df["time_s"].max())
    phases.setdefault("EOF", t_end)
    drones = {}
    for i in ids:
        g = df[df["id"] == i].sort_values("time_s")
        g = g[np.all(np.isfinite(g[POS_COLS].to_numpy(float)), axis=1)]
        drones[i] = {"t": g["time_s"].to_numpy(float), "p": g[POS_COLS].to_numpy(float)}
    return ids, drones, phases, t_end


def load_extras(log_path):
    """The flight setup and the setpoint log of a run, None for runs from before they were logged."""
    log_path = Path(log_path)
    meta_path = log_path.with_name(log_path.stem + "_meta.json")
    sp_path = log_path.with_name(log_path.stem + "_setpoints.csv")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    sp_log = None
    if sp_path.exists():
        sp_log = pd.read_csv(sp_path, dtype={"id": str, "flight_state": str})
        sp_log["id"] = sp_log["id"].str.replace(r"\.0$", "", regex=True)
        sp_log["flight_state"] = sp_log["flight_state"].fillna("").str.strip()
        sp_log = sp_log.sort_values("time_s")
    return meta, sp_log


def apply_meta(c, meta):
    """Constants the drones flew with (meta) instead of those of the flight code now."""
    tm, gd, tg = meta.get("timing_s", {}), meta.get("guidance", {}), meta.get("target", {})
    for k in ("SETPOINT_PERIOD", "TAKEOFF_TIME", "RETURN_TIME", "LAND_TIME", "Tend"):
        if k in tm:
            setattr(c, k, float(tm[k]))
    for k in ("dTi", "sphere_radius", "k_geometric", "k_rad", "k_apf"):
        if k in gd:
            setattr(c, k, float(gd[k]))
    for k in ("target_traj_radius", "default_speed", "default_height"):
        if k in tg:
            setattr(c, k, float(tg[k]))
    if "traj" in tg:
        c.traj = int(tg["traj"])
    return c


def measured_ctrl_period(meta):
    """Mean controller period measured in the flight (all drones), None if not logged."""
    per = (meta or {}).get("measured_setpoint_periods", {}).get(MAIN, {})
    vals = [d["period_mean_s"] for d in per.values() if "period_mean_s" in d]
    return float(np.mean(vals)) if vals else None


def apply_firmware_gains(param, meta, ids):
    """Position gains of the onboard Mellinger controller (read back from the firmware) in the plant."""
    fw = [meta["drones"][i].get("firmware", {}) for i in ids if i in meta.get("drones", {})]
    keys = ("ctrlMel.kp_xy", "ctrlMel.kp_z", "ctrlMel.kd_xy", "ctrlMel.kd_z")
    fw = [f for f in fw if all(f.get(k) is not None for k in keys)]
    if not fw:
        return None
    g = {k: fw[0][k] for k in keys}
    if any(any(abs(f[k] - g[k]) > 1e-9 for k in keys) for f in fw[1:]):
        print("Note: the drones flew with different Mellinger gains; the plant uses those of the first one")
    param.controller.Mellinger.kp = np.diag([g["ctrlMel.kp_xy"], g["ctrlMel.kp_xy"], g["ctrlMel.kp_z"]])
    param.controller.Mellinger.kv = np.diag([g["ctrlMel.kd_xy"], g["ctrlMel.kd_xy"], g["ctrlMel.kd_z"]])
    if fw[0].get("stabilizer.controller") not in (None, 2.0):
        print(f"Note: onboard controller {fw[0]['stabilizer.controller']} (not Mellinger) in the flight")
    return g


def logged_main_stream(sp_log, ids):
    """Main Controller setpoints of the flight, {id: (times, values (n, 3))}, None if not logged."""
    if sp_log is None or not (sp_log["flight_state"] == MAIN).any():
        return None
    out = {}
    for i in ids:
        g = sp_log[(sp_log["flight_state"] == MAIN) & (sp_log["id"] == i)]
        out[i] = (g["time_s"].to_numpy(float), g[["sp_x", "sp_y", "sp_z"]].to_numpy(float))
    return out


def newest_log():
    """Flight log of the newest run: Results/<flight code>_trajectory_<date>/trajectory_<date>.csv."""
    runs = [*LOGS_DIR.glob(f"{FLIGHT_CODE}_trajectory_*/"), *LOGS_DIR.glob(f"{OLD_PREFIX}_trajectory_*/")]
    for run in sorted(runs, key=lambda r: r.name.split("_trajectory_")[-1], reverse=True):
        log = run / f"trajectory_{run.name.split('_trajectory_')[-1]}.csv"
        if log.exists():
            return log
    raise SystemExit(f"No {FLIGHT_CODE} run in {LOGS_DIR} (pass the log: --IRL_log LOG)")


def position_at(drone, t):
    return np.array([np.interp(t, drone["t"], drone["p"][:, a]) for a in range(3)])


def homes_of(drones):
    """capture_home: the measured position on the floor before arming."""
    return {i: d["p"][0].copy() for i, d in drones.items()}


# =============================================================================
# Setpoints of the flight code
# =============================================================================

def schedule(c, phases, ctrl_period):
    """
    (phase, t_start, n, period) of every phase: n setpoints `period` apart.
    Ramps: n from the nominal duration, period measured from the logged phase
    times (minus the sleeps of main.py). Main Controller: one setpoint every
    ctrl_period for Tend (or until an abort).
    """
    order = [p for p in PHASES if p in phases]
    sched = []
    for k, name in enumerate(order):
        t0 = phases[name]
        nxt = order[k + 1] if k + 1 < len(order) else "EOF"
        if name == "Land":
            pause = PAUSE_AFTER_LAND
        elif nxt in (RETURN, "Land"):           # the finally block: sleep(0.3), then return / land
            pause = PAUSE_BEFORE_RETURN if name in ("Take off", MAIN) else 0.0
        else:
            pause = 0.0
        busy = phases[nxt] - t0 - pause
        if name == MAIN:
            n = max(0, math.ceil(min(c.Tend, busy) / ctrl_period - 1e-6))
            sched.append((name, t0, n, ctrl_period))
            continue
        duration = {"Take off": c.TAKEOFF_TIME, RETURN: c.RETURN_TIME, "Land": c.LAND_TIME}[name]
        n = max(1, int(duration / c.SETPOINT_PERIOD))
        period = busy / n if busy > 0 else c.SETPOINT_PERIOD
        if period < 0.9 * c.SETPOINT_PERIOD:    # an aborted phase: nominal period, fewer steps
            period, n = c.SETPOINT_PERIOD, max(1, int(busy / c.SETPOINT_PERIOD))
        sched.append((name, t0, n, period))
    return sched


def setpoint_events(sched, logged_stream=None):
    """
    (time, phase, step, n, id) of every setpoint, in time order. Ramps and the
    rebuilt Main Controller ticks: id None (all drones; the controller updates
    the drones in guidance order). Logged Main Controller setpoints: one drone each.
    """
    ev = []
    for name, t0, n, period in sched:
        if name == MAIN and logged_stream is not None:
            continue
        ev += [(t0 + k * period, name, k, n, None) for k in range(n)]
    if logged_stream is not None:
        ev += [(float(t), MAIN, k, None, i) for i, (ts, _) in logged_stream.items() for k, t in enumerate(ts)]
    return sorted(ev, key=lambda e: e[0])


def ramp_goal(c, name, home):
    if name in ("Take off", RETURN):
        return np.array([home[0], home[1], c.default_height])
    return np.array([home[0], home[1], LAND_Z])


def ramp_setpoint(start, goal, i, n):
    """i-th setpoint of cf_funcs.ramp_to (smoothstep from start to goal in n steps)."""
    f = (i + 1) / n
    f = f * f * (3.0 - 2.0 * f)
    return start + (goal - start) * f


def guidance_init(c, ids, homes, meta=None, legacy=False):
    """
    Initial guidance state (q_ref, radius) per drone, and the drone order of
    the guidance (active_uris: rigid body IDs in increasing order):
      * meta: the values logged by the flight code
      * legacy (logs from before 12 Nov, no meta): main.py computed q_ref before
        the OptiTrack stream started, from zero positions -> the fallback
        (1, 0, 0) for every drone, radius 0
      * otherwise (cf_funcs.init_guidance): from the target start to the takeoff
        point (home x, home y, default_height), radius = that distance
    """
    order = list(ids)
    if meta and all(i in meta.get("drones", {}) for i in ids):
        order = sorted(ids, key=lambda i: meta["drones"][i]["guidance_index"])
        return order, {i: np.asarray(meta["drones"][i]["q_ref0"], float).reshape(3, 1) for i in ids}, \
            {i: float(meta["drones"][i]["radius0"]) for i in ids}
    if legacy:
        return order, {i: np.array([[1.0], [0.0], [0.0]]) for i in ids}, {i: 0.0 for i in ids}
    t0 = flight_target_traj(c)["pos"](0.0).ravel()
    q0, r0 = {}, {}
    for i in ids:
        v = np.array([homes[i][0], homes[i][1], c.default_height]) - t0
        n = float(np.linalg.norm(v))
        q0[i] = (v / n).reshape(3, 1) if n > 1e-6 else np.array([[1.0], [0.0], [0.0]])
        r0[i] = n
    return order, q0, r0
