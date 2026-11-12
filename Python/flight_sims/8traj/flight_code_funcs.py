"""
The flight code Crazy_Flie/Optitrack/8traj.py, for its simulation (main.py):
its constants (read from 8traj.py), its flight log, and the setpoint stream it
sends (send_position_setpoint, sample and hold between setpoints):

    Take off          (0, 0, H (k+1)/n), n = TAKEOFF_TIME / SETPOINT_PERIOD
    Main Controller   (A sin(phi), B sin(2 phi), H), phi = figure8_phase(t since the phase start):
                      w t with the rate eased in / out over TRAJ_EASE_TIME (0 for the flights
                      before 10 Nov), MAIN_CONTROL_TIME
    Return to origin  smoothstep ramp from the position at the phase start to (0, 0, H),
                      max(1.5 dist / RETURN_SPEED, LAND_TIME), then (0, 0, 0) once
                      (flights before 10 Nov: a step, (0, 0, H) x LAND_TIME / SETPOINT_PERIOD)
    Land              (0, 0, H (1 - (k+1)/n)), n = LAND_TIME / SETPOINT_PERIOD

Each phase starts at its logged time, with the setpoint period measured in the log.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())
FLIGHT_SCRIPT = REPO / "Crazy_Flie" / "Optitrack" / "8traj.py"
FLIGHT_CODE = FLIGHT_SCRIPT.stem
LOGS_DIR = REPO / "Results"

POS_COLS = ["opti_x", "opti_y", "opti_z"]
PHASES = ("Take off", "Main Controller", "Return to origin", "Land")


def configure_param(param, t_end, dt, log_path=None):
    """param (Structs.Param) for the flight: the plant steps dt over the whole flight, the log as real trajectory."""
    param.sim.nD = 1
    param.sim.dTi = dt
    param.sim.Nsim = int(round(t_end / dt)) + 1
    param.sim.Tend = (param.sim.Nsim - 1) * dt
    param.sim.t = np.arange(param.sim.Nsim) * dt
    param.sim.real_trajectory_csv = Path(log_path) if log_path else None
    return param


def flight_constants(path=FLIGHT_SCRIPT):
    """Module-level numeric constants of the flight script (no import: it needs cflib / natnet)."""
    want = {"TARGET_HEIGHT": 1.0, "DEFAULT_SPEED": 0.8, "TRAJ_A": 1.5, "TRAJ_B": 1.0,
            "MAIN_CONTROL_TIME": 30.0, "TAKEOFF_TIME": 3.0, "LAND_TIME": 2.0, "SETPOINT_PERIOD": 0.05,
            "RIGID_BODY_ID": 4, "RETURN_SPEED": 0.5, "TRAJ_EASE_TIME": 0.0}
    found = {}
    try:
        for node in ast.parse(Path(path).read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in want:
                    try:
                        found[name] = float(ast.literal_eval(node.value))
                    except ValueError:
                        pass
    except OSError:
        print(f"Note: {path} not found, nominal constants used")
    missing = [k for k in want if k not in found]
    if missing:
        print(f"Note: not in {Path(path).name}, defaults used: " + ", ".join(f"{k}={want[k]}" for k in missing))
    return {k: found.get(k, v) for k, v in want.items()}


def load_log(path):
    """(drone id, {t, p, yaw_deg}, phase start times, t_end)."""
    df = pd.read_csv(path, dtype={"flight_state": str})
    df["flight_state"] = df["flight_state"].fillna("").str.strip()
    if "id" in df.columns:
        ids = df["id"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).unique()
        if len(ids) != 1:
            raise SystemExit(f"8traj flies one drone, {path} has ids {list(ids)}")
        did = ids[0]
    else:
        did = "1"
    df = df.sort_values("time_s")
    phases = {}
    for name in (*PHASES, "EOF"):
        r = df[df["flight_state"] == name]
        if len(r):
            phases[name] = float(r["time_s"].min())
    if "Take off" not in phases:
        raise SystemExit(f"No 'Take off' phase in {path}")
    t_end = float(df["time_s"].max())
    g = df[np.all(np.isfinite(df[POS_COLS].to_numpy(float)), axis=1)]
    drone = {"t": g["time_s"].to_numpy(float), "p": g[POS_COLS].to_numpy(float)}
    return did, drone, phases, t_end


def figure8_phase(c, t):
    """figure8_phase of 8traj.py (soft start / stop of the figure 8)."""
    w, T = c["DEFAULT_SPEED"], c["MAIN_CONTROL_TIME"]
    E = min(c["TRAJ_EASE_TIME"], T / 2)
    if E <= 0:
        return w * t
    if t < E:
        return w * t * t / (2 * E)
    if t < T - E:
        return w * (t - E / 2)
    tau = min(t, T) - (T - E)
    return w * (T - 1.5 * E + tau - tau * tau / (2 * E))


def return_steps(c, mode, p0):
    """Number of ramp / hold setpoints of return_origin (before the final (0, 0, 0))."""
    if mode == "step":
        return int(c["LAND_TIME"] / c["SETPOINT_PERIOD"])
    d = float(np.linalg.norm(np.array([0.0, 0.0, c["TARGET_HEIGHT"]]) - p0))
    return int(max(1.5 * d / c["RETURN_SPEED"], c["LAND_TIME"]) / c["SETPOINT_PERIOD"])


def return_setpoint(c, mode, p0, k, n):
    """k-th setpoint of return_origin from the position p0 at its start (k == n: the final (0, 0, 0))."""
    if k >= n:
        return np.zeros(3)
    goal = np.array([0.0, 0.0, c["TARGET_HEIGHT"]])
    if mode == "step":
        return goal
    f = (k + 1) / n
    f = f * f * (3.0 - 2.0 * f)
    return p0 + (goal - p0) * f


def return_mode(c, phases, t_end, p0, period):
    """'step' or 'ramp': the return_origin whose duration matches the logged phase."""
    if "Return to origin" not in phases:
        return "ramp"
    nxt = phases.get("Land", phases.get("EOF", t_end))
    dur = nxt - phases["Return to origin"]
    err = {m: abs(dur - (return_steps(c, m, p0) + 1) * period) for m in ("ramp", "step")}
    return min(err, key=err.get) if err["step"] + 1e-3 < err["ramp"] else "ramp"


def setpoint_events(c, phases, t_end, mode, p0_return):
    """
    [(t, phase, setpoint)] sent by 8traj.py, the period of every phase measured
    from the logged phase durations (nominal when a phase is not logged). The
    return setpoints are placeholders ("ret", k, n), resolved from the position
    at the start of the return (return_setpoint); n from p0_return, the logged one.
    """
    H, w, A, B, sp = c["TARGET_HEIGHT"], c["DEFAULT_SPEED"], c["TRAJ_A"], c["TRAJ_B"], c["SETPOINT_PERIOD"]
    n_to, n_land = int(c["TAKEOFF_TIME"] / sp), int(c["LAND_TIME"] / sp)
    order = [p for p in PHASES if p in phases]
    end = {p: (phases[order[k + 1]] if k + 1 < len(order) else phases.get("EOF", t_end)) for k, p in enumerate(order)}
    ev = []
    period_to = sp
    if "Take off" in phases:
        t0 = phases["Take off"]
        period_to = (end["Take off"] - t0) / n_to
        if not 0.9 * sp <= period_to <= 1.5 * sp:       # aborted / odd: nominal period
            period_to = sp
        for k in range(n_to):
            ev.append((t0 + k * period_to, "Take off", np.array([0.0, 0.0, H * (k + 1) / n_to])))
    if "Main Controller" in phases:
        t0 = phases["Main Controller"]
        k = 0
        while True:
            tr = k * period_to                          # time.monotonic() - start_time at the send
            if tr >= c["MAIN_CONTROL_TIME"] or t0 + tr >= end["Main Controller"]:
                break
            phi = figure8_phase(c, tr)
            ev.append((t0 + tr, "Main Controller", np.array([A * np.sin(phi), B * np.sin(2 * phi), H])))
            k += 1
    if "Return to origin" in phases:
        t0 = phases["Return to origin"]
        n_ret = return_steps(c, mode, p0_return)
        per = (end["Return to origin"] - t0) / (n_ret + 1)
        per = per if 0.9 * sp <= per <= 1.5 * sp else sp
        for k in range(n_ret + 1):                                    # k == n_ret: the (0, 0, 0) setpoint
            if t0 + k * per < end["Return to origin"] + 1e-6:
                ev.append((t0 + k * per, "Return to origin", ("ret", k, n_ret)))
    if "Land" in phases:
        t0 = phases["Land"]
        per = (end["Land"] - t0) / n_land
        per = per if 0.9 * sp <= per <= 1.5 * sp else sp
        for k in range(n_land):
            ev.append((t0 + k * per, "Land", np.array([0.0, 0.0, H * (1.0 - (k + 1) / n_land)])))
    return sorted(ev, key=lambda e: e[0])


def resolve(events, c, mode, p0):
    """Events with the return placeholders replaced, the return ramp starting at p0."""
    return [(t, ph, return_setpoint(c, mode, p0, v[1], v[2]) if isinstance(v, tuple) else v)
            for t, ph, v in events]


def detect_ease(c, phases, t_end, drone):
    """TRAJ_EASE_TIME of the flight: 8traj.py's value or 0 (old flights), whichever the real drone tracked best."""
    if "Main Controller" not in phases or c["TRAJ_EASE_TIME"] <= 0:
        return c["TRAJ_EASE_TIME"]
    t0 = phases["Main Controller"]
    t1 = phases.get("Return to origin", phases.get("EOF", t_end))
    m = (drone["t"] >= t0) & (drone["t"] < t1)
    rms = {}
    for e in (c["TRAJ_EASE_TIME"], 0.0):
        ce = dict(c, TRAJ_EASE_TIME=e)
        ev = [x for x in setpoint_events(ce, phases, t_end, "step", np.zeros(3)) if x[1] == "Main Controller"]
        et, ep = np.array([x[0] for x in ev]), np.array([x[2] for x in ev])
        k = np.clip(np.searchsorted(et, drone["t"][m], side="right") - 1, 0, len(et) - 1)
        rms[e] = float(np.sqrt(np.mean(np.sum((drone["p"][m] - ep[k]) ** 2, 1))))
    return min(rms, key=rms.get)


def newest_log():
    for run in sorted(LOGS_DIR.glob(f"{FLIGHT_CODE}_trajectory_*/"), reverse=True):
        log = run / f"{run.name[len(FLIGHT_CODE) + 1:]}.csv"
        if log.exists():
            return log
    raise SystemExit(f"No {FLIGHT_CODE} run in {LOGS_DIR}")


def flight_from_log(log_path, c, ret="auto", ease=None, verbose=True):
    """
    Everything the replay of a logged flight needs, from the log:
    (did, drone, phases, t_end, mode, p0_return). Sets c["TRAJ_EASE_TIME"]
    (ease, or detected from the log) and picks the return mode (ret, or
    detected from the logged return duration).
    """
    did, drone, phases, t_end = load_log(log_path)
    if verbose:
        print("Phases: " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))
    if ease is None:
        c["TRAJ_EASE_TIME"] = detect_ease(c, phases, t_end, drone)
        if verbose:
            print(f"Figure 8 soft start / stop: TRAJ_EASE_TIME = {c['TRAJ_EASE_TIME']:g} s (detected from the log)")
    else:
        c["TRAJ_EASE_TIME"] = ease
    p0_ret = (np.array([np.interp(phases["Return to origin"], drone["t"], drone["p"][:, a]) for a in range(3)])
              if "Return to origin" in phases else np.zeros(3))
    tk = phases.get("Main Controller", t_end) - phases["Take off"]
    period = tk / int(c["TAKEOFF_TIME"] / c["SETPOINT_PERIOD"]) if "Main Controller" in phases else c["SETPOINT_PERIOD"]
    mode = return_mode(c, phases, t_end, p0_ret, period) if ret == "auto" else ret
    if verbose:
        print(f"Return to origin: {mode}" + (" (detected from the logged duration)" if ret == "auto" else ""))
    return did, drone, phases, t_end, mode, p0_ret


def real_setpoints(c, phases, t_end, mode, p0_return):
    """The setpoints 8traj.py sent to the real drone: [(t, phase, setpoint)]."""
    return resolve(setpoint_events(c, phases, t_end, mode, p0_return), c, mode, p0_return)
