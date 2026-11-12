"""
The flight code Crazy_Flie/Optitrack/swarm_take_off_hover_land, for its
simulation (main.py): its constants (read from its Structs.py / main.py), its
flight log, and the setpoints it sends (cf_funcs.ramp_to: smoothstep ramps
sent with send_position_setpoint):

    Take off  ramp from the measured position to (home x, home y, TAKEOFF_HEIGHT), TAKEOFF_TIME
    Hover     ramp from the measured position to (home x, home y, TAKEOFF_HEIGHT), HOVER_TIME
    Return    ramp to the assigned takeoff point (Hungarian, at TAKEOFF_HEIGHT),
              max(1.5 * max distance / RETURN_SPEED, 2 s)
    Land      ramp from the measured position to (x, y, LAND_HEIGHT), LAND_TIME, motors off at EOF

Each phase starts at the time logged in the flight log, with the setpoint
period measured in it (ramp_to sleeps SETPOINT_PERIOD after every setpoint,
plus the loop overhead).
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())   # repository root
FLIGHT_CODE = HERE.name         # same name as its flight code, Crazy_Flie/Optitrack/<name>
FLIGHT_DIR = REPO / "Crazy_Flie" / "Optitrack" / FLIGHT_CODE
LOGS_DIR = REPO / "Results"     # one folder per run: Results/<flight code>_trajectory_<date>/


def flight_constants(folder=FLIGHT_DIR):
    """Module-level numeric constants of the flight code (Structs.py, main.py), without importing it (cflib / natnet)."""
    want = {"TAKEOFF_HEIGHT": 0.6, "TAKEOFF_TIME": 3.0, "HOVER_TIME": 8.0, "RETURN_SPEED": 0.3,
            "LAND_TIME": 4.0, "LAND_HEIGHT": 0.05, "SETPOINT_PERIOD": 0.05}
    found = {}
    for name in ("Structs.py", "main.py"):
        try:
            for node in ast.parse((folder / name).read_text()).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id in want:
                    try:
                        found[node.targets[0].id] = float(ast.literal_eval(node.value))
                    except ValueError:
                        pass
        except OSError:
            pass
    missing = [k for k in want if k not in found]
    if missing:
        print("Note: not in the flight code, defaults used: " + ", ".join(f"{k}={want[k]}" for k in missing))
    return {k: found.get(k, v) for k, v in want.items()}


# Flight code constants (swarm_take_off_hover_land/Structs.py, main.py)
_C = flight_constants()
TAKEOFF_HEIGHT = _C["TAKEOFF_HEIGHT"]   # m
TAKEOFF_TIME = _C["TAKEOFF_TIME"]       # s
HOVER_TIME = _C["HOVER_TIME"]           # s
RETURN_SPEED = _C["RETURN_SPEED"]       # m/s, peak speed back to the takeoff points
LAND_TIME = _C["LAND_TIME"]             # s
LAND_HEIGHT = _C["LAND_HEIGHT"]         # m
SETPOINT_PERIOD = _C["SETPOINT_PERIOD"] # s (nominal; the measured one is used)
PAUSE_BEFORE_RETURN = 0.3  # s, time.sleep(0.3) before "Return to takeoff points" (main.py)
PAUSE_AFTER_LAND = 1.0    # s, time.sleep(1.0) between the landing and "EOF" (main.py)

RETURN = "Return to takeoff points"
PHASES = ("Take off", "Hover", RETURN, "Land")
PHASE_COLORS = {"Take off": "#cfe8ff", "Hover": "#d9f2d0", RETURN: "#f3e1f7", "Land": "#ffe0c7"}
SHORT = {"Take off": "Take off", "Hover": "Hover", RETURN: "Return", "Land": "Land"}
POS_COLS = ["opti_x", "opti_y", "opti_z"]


def configure_param(param, n_drones, t_end, dt, log_path=None):
    """param (Structs.Param) for the flight: the plant steps dt over the whole flight, the log as real trajectory."""
    param.sim.nD = n_drones
    param.sim.dTi = dt
    param.sim.Nsim = int(round(t_end / dt)) + 1
    param.sim.Tend = (param.sim.Nsim - 1) * dt
    param.sim.t = np.arange(param.sim.Nsim) * dt
    param.sim.real_trajectory_csv = Path(log_path) if log_path else None
    return param


def load_log(path):
    """Per drone: time, OptiTrack position, yaw; phase start times of the log."""
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
        drones[i] = {"t": g["time_s"].to_numpy(float), "p": g[POS_COLS].to_numpy(float),
                     "yaw_deg": g["yaw_deg"].to_numpy(float) if "yaw_deg" in g else np.zeros(len(g))}
    return ids, drones, phases, t_end


def position_at(drone, t):
    return np.array([np.interp(t, drone["t"], drone["p"][:, a]) for a in range(3)])


def homes_of(drones):
    """capture_home: the measured position on the floor before arming."""
    return {i: d["p"][0].copy() for i, d in drones.items()}


def return_spots(homes, starts):
    """
    assign_land_spots: every drone to one of the takeoff points (at takeoff
    height), minimizing the sum of squared distances; common return duration.
    """
    ids = list(starts)
    P = np.array([starts[i] for i in ids])
    H = np.array([[*homes[i][:2], TAKEOFF_HEIGHT] for i in ids])
    _, home_of = linear_sum_assignment(np.sum((P[:, None, :] - H[None, :, :]) ** 2, axis=2))
    spots = {i: H[h] for i, h in zip(ids, home_of)}
    duration = max(1.5 * float(np.max(np.linalg.norm(P - H[home_of], axis=1))) / RETURN_SPEED, 2.0)
    return spots, duration


def phase_goals(name, homes, starts):
    """Goal of the ramp of every drone, from the measured positions `starts` at the phase start."""
    if name in ("Take off", "Hover"):
        return {i: np.array([homes[i][0], homes[i][1], TAKEOFF_HEIGHT]) for i in starts}
    if name == RETURN:
        return return_spots(homes, starts)[0]
    return {i: np.array([s[0], s[1], LAND_HEIGHT]) for i, s in starts.items()}


def pause_after(name, next_name):
    """Sleep of the flight code between the end of the ramp `name` and the next mark_state."""
    if name == "Land":
        return PAUSE_AFTER_LAND
    if name == RETURN:                      # land right after the return
        return 0.0
    if next_name in (RETURN, "Land"):       # the finally block: sleep(0.3), then return / land
        return PAUSE_BEFORE_RETURN
    return 0.0


def setpoint_schedule(phases, drones):
    """
    (phase, t_start, n_steps, period) of every ramp_to call, from the logged
    phase times: the phase lasts its n setpoints plus the pause after it
    (pause_after). The return duration is computed as in the flight code,
    from the measured positions at the return start.
    """
    homes = homes_of(drones)
    order = [p for p in PHASES if p in phases]
    sched = []
    for k, name in enumerate(order):
        t0 = phases[name]
        nxt = order[k + 1] if k + 1 < len(order) else "EOF"
        t1 = phases[nxt]
        if name == RETURN:
            duration = return_spots(homes, {i: position_at(d, t0) for i, d in drones.items()})[1]
        else:
            duration = {"Take off": TAKEOFF_TIME, "Hover": HOVER_TIME, "Land": LAND_TIME}[name]
        n = max(1, int(duration / SETPOINT_PERIOD))
        busy = (t1 - t0) - pause_after(name, nxt)
        period = busy / n if busy > 0 else SETPOINT_PERIOD
        # an aborted phase (shorter than nominal): keep the nominal period, fewer steps
        if period < 0.9 * SETPOINT_PERIOD:
            period, n = SETPOINT_PERIOD, max(1, int(busy / SETPOINT_PERIOD))
        sched.append((name, t0, n, period))
    return sched


def ramp_setpoint(start, goal, i, n):
    """i-th setpoint of cf_funcs.ramp_to (smoothstep from start to goal in n steps)."""
    f = (i + 1) / n
    f = f * f * (3.0 - 2.0 * f)
    return start + (goal - start) * f


def real_setpoints(drones, sched, times):
    """
    Setpoints the flight code sent to the real drones (from their measured
    positions at each phase start), at the times `times[id]`.
    """
    homes = homes_of(drones)
    sp = {i: np.full((len(times[i]), 3), np.nan) for i in drones}
    for k, (name, t0, n, period) in enumerate(sched):
        starts = {i: position_at(d, t0) for i, d in drones.items()}
        goals = phase_goals(name, homes, starts)
        last_end = t0 + n * period
        nxt = sched[k + 1][0] if k + 1 < len(sched) else "EOF"
        # after the last setpoint the onboard controller keeps it until the next phase / stop
        hold_end = sched[k + 1][1] if k + 1 < len(sched) else last_end + pause_after(name, nxt)
        for i in drones:
            t = times[i]
            for j in range(n):
                sp[i][(t >= t0 + j * period) & (t < t0 + (j + 1) * period)] = \
                    ramp_setpoint(starts[i], goals[i], j, n)
            sp[i][(t >= last_end) & (t < hold_end)] = goals[i]
    return sp


def newest_log():
    """Flight log of the newest run: Results/swarm_take_off_hover_land_trajectory_<date>/trajectory_<date>.csv."""
    prefix = f"{FLIGHT_CODE}_"
    for run in sorted(LOGS_DIR.glob(f"{FLIGHT_CODE}_trajectory_*/"), reverse=True):
        log = run / f"{run.name[len(prefix):]}.csv"
        if log.exists():
            return log
    raise SystemExit(f"No {FLIGHT_CODE} run in {LOGS_DIR}")
