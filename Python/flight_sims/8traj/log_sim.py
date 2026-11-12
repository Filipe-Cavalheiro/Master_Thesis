"""
Simulation of a figure-8 flight (Crazy_Flie/Optitrack/8traj.py), to compare
with its log and to overlay on the phone video (video_overlay/flight_overlay.py).

Same simulation as swarm_take_off_hover_land/main.py: the V8 Crazyflie plant
(cf_plant.py in this folder, self-contained: Param, drone cf2x_P250, Mellinger
controller with the gains 8traj.py writes to the firmware, kp = 1.1448,
kv = 0.34344, full nonlinear dynamics, RK4 on SO(3)), the drone starting at
rest from its logged position and yaw. The outer loop is the setpoint stream
of 8traj.py (send_position_setpoint, sample and hold between setpoints):

    Take off          (0, 0, H (k+1)/n), n = TAKEOFF_TIME / SETPOINT_PERIOD
    Main Controller   (A sin(phi), B sin(2 phi), H), phi = figure8_phase(t since the phase start):
                      w t with the rate eased in / out over TRAJ_EASE_TIME (0 for the flights
                      before 10 Nov: --ease 0), MAIN_CONTROL_TIME
    Return to origin  smoothstep ramp from the position at the phase start to (0, 0, H),
                      max(1.5 dist / RETURN_SPEED, LAND_TIME), then (0, 0, 0) once
                      (flights before 10 Nov: a step, (0, 0, H) x LAND_TIME / SETPOINT_PERIOD;
                      --return auto picks the version whose duration matches the log)
    Land              (0, 0, H (1 - (k+1)/n)), n = LAND_TIME / SETPOINT_PERIOD

Apart from the start of the return ramp (each drone's own position), the
setpoints do not depend on the measured state: the real and the simulated
drone get the same stream. Each phase starts at its logged time,
with the setpoint period measured in the log (SETPOINT_PERIOD + loop overhead).
H, w = DEFAULT_SPEED, A, B and the times are read from 8traj.py itself (so the
simulation follows edits of the flight code).

    python 8traj/log_sim.py                    # newest 8traj run in Results/
    python 8traj/log_sim.py --log ../../Results/8traj_trajectory_X/trajectory_X.csv --no-show
    python 8traj/log_sim.py --sim              # no log: the nominal flight

Writes <log>_sim.csv next to the flight log (same columns as the log, plus the
setpoints sp_x, sp_y, sp_z) and the comparison in <run folder>/comparison/:
position.pdf (x, y, z real / sim / setpoint), top_view.pdf, difference.pdf
(sim - real per axis and |e|), metrics.json (RMS per phase, max, tracking error).
--sim writes Results/8traj_<date>/sim.csv.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())
FLIGHT_SCRIPT = REPO / "Crazy_Flie" / "Optitrack" / "8traj.py"
FLIGHT_CODE = FLIGHT_SCRIPT.stem
LOGS_DIR = REPO / "Results"

# The V8 Crazyflie plant, copied into this folder (cf_plant.py; drone_models.toml is the same).
# This folder also has an older simulation (Structs.py, helper_funcs.py, ...): not used here.
sys.path.insert(0, str(HERE))
from cf_plant import (DroneState, drone_motion, get_initial_real_offset,  # noqa: E402
                      ground_contact, make_param, passive_motion)

DT = 0.01
POS_COLS = ["opti_x", "opti_y", "opti_z"]
PHASES = ("Take off", "Main Controller", "Return to origin", "Land")
PHASE_COLORS = {"Take off": "#cfe8ff", "Main Controller": "#d9f2d0", "Return to origin": "#f3e1f7",
                "Land": "#ffe0c7"}


# =============================================================================
# Flight code constants (read from 8traj.py)
# =============================================================================

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


# =============================================================================
# Log
# =============================================================================

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


# =============================================================================
# Setpoints of 8traj.py
# =============================================================================

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


def nominal_phases(c, mode):
    """Phase times without a log, and the position at the return start (the figure-8 setpoint)."""
    sp = c["SETPOINT_PERIOD"]
    ph = {"Take off": 0.0}
    ph["Main Controller"] = c["TAKEOFF_TIME"]
    ph["Return to origin"] = ph["Main Controller"] + c["MAIN_CONTROL_TIME"]
    phi = figure8_phase(c, c["MAIN_CONTROL_TIME"])
    p0 = np.array([c["TRAJ_A"] * np.sin(phi), c["TRAJ_B"] * np.sin(2 * phi), c["TARGET_HEIGHT"]])
    ph["Land"] = ph["Return to origin"] + (return_steps(c, mode, p0) + 1) * sp
    ph["EOF"] = ph["Land"] + c["LAND_TIME"]
    return ph, p0


# =============================================================================
# Simulation
# =============================================================================

def simulate(log_path, did, home, yaw0, phases, t_end, c, mode, p0_return):
    param = make_param(log_path, 1, t_end)
    events = setpoint_events(c, phases, t_end, mode, p0_return)
    t_takeoff = phases["Take off"]

    a = DroneState(param)
    a.position = np.asarray(home, float).reshape(3, 1).copy()
    a.velocity = np.zeros((3, 1))
    a.attitude = Rotation.from_euler("z", yaw0).as_matrix().reshape(-1, 1)     # level on the floor
    a.ang_vel = np.zeros((3, 1))
    a.desired.position = a.position.copy()
    a.desired.psi = a.desired.old_psi = 0.0                                     # yaw setpoint 0
    floor = float(home[2])

    log, sp, ev, t, ret_p0 = [], None, 0, min(0.0, t_takeoff), None
    while t <= t_end + 1e-9:
        while ev < len(events) and events[ev][0] <= t + 1e-9:
            v = events[ev][2]
            if isinstance(v, tuple):                    # return_origin: get_optitrack_position() at its start
                if ret_p0 is None:
                    ret_p0 = a.position.ravel().copy()
                v = return_setpoint(c, mode, ret_p0, v[1], v[2])
            sp = v
            ev += 1
        phase = max((p for p in phases.items() if p[1] <= t + 1e-9 and p[0] != "EOF"),
                    key=lambda p: p[1], default=("", 0))[0]
        if sp is not None and t >= t_takeoff:
            a.desired.position = sp.reshape(3, 1).copy()
            drone_motion(a, param)
        elif t >= t_takeoff:
            passive_motion(a, param)
        ground_contact(a, floor)
        roll, pitch, yaw = Rotation.from_matrix(a.attitude.reshape(3, 3)).as_euler("xyz", degrees=True)
        s = sp if sp is not None else np.full(3, np.nan)
        log.append([t + DT, phase, did, *a.position.ravel(), *a.velocity.ravel(), roll, pitch, yaw, *s])
        t += DT
    cols = ["time_s", "flight_state", "id", *POS_COLS, "est_vx", "est_vy", "est_vz",
            "roll_deg", "pitch_deg", "yaw_deg", "sp_x", "sp_y", "sp_z"]
    return pd.DataFrame(log, columns=cols), events


# =============================================================================
# Comparison
# =============================================================================

def compare(log_path, sim_df, events, phases, drone, out_dir, show):
    """events: the setpoints of the real drone (resolve() with its logged return start)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t = drone["t"]
    P = drone["p"]
    S = np.column_stack([np.interp(t, sim_df["time_s"], sim_df[c], left=np.nan, right=np.nan) for c in POS_COLS])
    et = np.array([e[0] for e in events])
    ep = np.array([e[2] for e in events])
    k = np.searchsorted(et, t, side="right") - 1
    SP = np.where((k >= 0)[:, None], ep[np.clip(k, 0, None)], np.nan)
    E = S - P

    order = [p for p in PHASES if p in phases]
    ends = {p: (phases[order[i + 1]] if i + 1 < len(order) else t[-1]) for i, p in enumerate(order)}

    def stats(m):
        e = E[m]
        e = e[np.all(np.isfinite(e), 1)]
        if not len(e):
            return None
        n = np.linalg.norm(e, axis=1)
        tr_r = np.linalg.norm((P - SP)[m], axis=1)
        tr_s = np.linalg.norm((S - SP)[m], axis=1)
        return {"rms_xyz_m": np.sqrt(np.mean(e ** 2, 0)).round(4).tolist(), "rms_3d_m": round(float(np.sqrt(np.mean(n ** 2))), 4),
                "mean_3d_m": round(float(n.mean()), 4), "p95_3d_m": round(float(np.percentile(n, 95)), 4),
                "max_3d_m": round(float(n.max()), 4),
                "tracking_rms_real_m": round(float(np.sqrt(np.nanmean(tr_r ** 2))), 4),
                "tracking_rms_sim_m": round(float(np.sqrt(np.nanmean(tr_s ** 2))), 4)}

    metrics = {"log": str(log_path), "sim_minus_real": {"whole flight": stats(t >= phases["Take off"])}}
    for p in order:
        s = stats((t >= phases[p]) & (t < ends[p]))
        if s:
            metrics["sim_minus_real"][p] = s
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def shade(ax):
        for p in order:
            ax.axvspan(phases[p], ends[p], color=PHASE_COLORS[p], lw=0, zorder=0)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=150)

    C_R, C_S = (60 / 255, 160 / 255, 80 / 255), (255 / 255, 140 / 255, 30 / 255)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for a, ax in enumerate(axs):
        shade(ax)
        ax.plot(t, SP[:, a], "--", color="0.4", lw=1.2, label="setpoint")
        ax.plot(t, P[:, a], color=C_R, lw=1.8, label="real")
        ax.plot(t, S[:, a], color=C_S, lw=1.5, label="sim")
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(True, alpha=0.4)
    axs[0].legend(loc="upper right", ncol=3)
    axs[-1].set_xlabel("Time [s]")
    save(fig, "position")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    m = (t >= phases.get("Main Controller", phases["Take off"]))
    ax.plot(SP[m, 0], SP[m, 1], "--", color="0.4", lw=1.2, label="setpoint")
    ax.plot(P[m, 0], P[m, 1], color=C_R, lw=1.6, label="real")
    ax.plot(S[m, 0], S[m, 1], color=C_S, lw=1.4, label="sim")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    save(fig, "top_view")

    fig, ax = plt.subplots(figsize=(10, 4))
    shade(ax)
    for a, col in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        ax.plot(t, 100 * E[:, a], color=col, lw=1.2, label=f"$e_{'xyz'[a]}$")
    ax.plot(t, 100 * np.linalg.norm(E, axis=1), "k", lw=1.8, label=r"$\|e\|$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("sim - real [cm]")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", ncol=4)
    save(fig, "difference")

    w = metrics["sim_minus_real"]
    print("sim - real (3D RMS / max):")
    for p, s in w.items():
        print(f"  {p:18s} {100 * s['rms_3d_m']:6.1f} / {100 * s['max_3d_m']:6.1f} cm   "
              f"tracking RMS real {100 * s['tracking_rms_real_m']:5.1f} cm, sim {100 * s['tracking_rms_sim_m']:5.1f} cm")
    print(f"Comparison saved to {out_dir}")
    if show:
        plt.show()
    plt.close("all")


# =============================================================================
# Main
# =============================================================================

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help=f"flight log (default: newest {FLIGHT_CODE} run in Results/)")
    ap.add_argument("--sim", action="store_true", help="no log: simulate the nominal flight")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    ap.add_argument("--return", dest="ret", choices=("auto", "ramp", "step"), default="auto",
                    help="return_origin of the flight: smooth ramp (from 10 Nov) or the old step; auto: from the log")
    ap.add_argument("--ease", type=float, help="TRAJ_EASE_TIME of the flight [s] (default: 8traj.py; 0 before 10 Nov)")
    args = ap.parse_args()
    c = flight_constants()
    if args.ease is not None:
        c["TRAJ_EASE_TIME"] = args.ease
    print("8traj.py: " + ", ".join(f"{k}={v:g}" for k, v in c.items()))

    if args.sim:
        mode = "ramp" if args.ret == "auto" else args.ret
        ph, p0 = nominal_phases(c, mode)
        sim_df, _ = simulate(None, str(int(c["RIGID_BODY_ID"])), np.array([0.0, 0.0, 0.03]), 0.0, ph, ph["EOF"],
                             c, mode, p0)
        out = LOGS_DIR / f"{FLIGHT_CODE}_{datetime.now():%Y%m%d_%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(out / "sim.csv", index=False)
        print(f"Nominal flight saved to {out / 'sim.csv'}")
        return

    log_path = Path(args.log).resolve() if args.log else newest_log()
    print(f"Log: {log_path}")
    did, drone, phases, t_end = load_log(log_path)
    print("Phases: " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))
    if args.ease is None:
        c["TRAJ_EASE_TIME"] = detect_ease(c, phases, t_end, drone)
        print(f"Figure 8 soft start / stop: TRAJ_EASE_TIME = {c['TRAJ_EASE_TIME']:g} s (detected from the log)")
    param = make_param(log_path, 1, t_end)
    _, _, att, _, _ = get_initial_real_offset(param, phase="Take off")
    yaw0 = Rotation.from_matrix(np.asarray(att, float).reshape(3, 3)).as_euler("ZYX")[0]
    p0_ret = (np.array([np.interp(phases["Return to origin"], drone["t"], drone["p"][:, a]) for a in range(3)])
              if "Return to origin" in phases else np.zeros(3))
    tk = phases.get("Main Controller", t_end) - phases["Take off"]
    period = tk / int(c["TAKEOFF_TIME"] / c["SETPOINT_PERIOD"]) if "Main Controller" in phases else c["SETPOINT_PERIOD"]
    mode = return_mode(c, phases, t_end, p0_ret, period) if args.ret == "auto" else args.ret
    print(f"Return to origin: {mode}" + (" (detected from the logged duration)" if args.ret == "auto" else ""))
    sim_df, events = simulate(log_path, did, drone["p"][0], yaw0, phases, t_end, c, mode, p0_ret)
    out = log_path.with_name(log_path.stem + "_sim.csv")
    sim_df.to_csv(out, index=False)
    print(f"{len(events)} setpoints; simulated flight saved to {out}")
    events = resolve(events, c, mode, p0_ret)
    compare(log_path, sim_df, events, phases, drone, log_path.parent / "comparison", show=not args.no_show)


if __name__ == "__main__":
    main()
