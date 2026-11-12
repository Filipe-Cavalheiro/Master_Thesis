"""
Simulation of the swarm take off / hover / return / land flight
(Crazy_Flie/Optitrack/swarm_take_off_hover_land).

    python swarm_take_off_hover_land/main.py                    # simulation only, simulated initial values
    python swarm_take_off_hover_land/main.py --drones 5 --spacing 0.6
    python swarm_take_off_hover_land/main.py --IRL_log          # from the newest real run in Results/
    python swarm_take_off_hover_land/main.py --IRL_log ../../Results/swarm_take_off_hover_land_trajectory_<date>/trajectory_<date>.csv
    python swarm_take_off_hover_land/main.py --IRL_log --no-show

Without --IRL_log: the nominal flight, no real-world values. --drones drones
start at rest on the floor in a row, --spacing m apart along x (yaw 0), with
the nominal setpoint period and phase times. Writes
Results/swarm_take_off_hover_land_<date>/sim.csv and the simulation figures
in its simulation/ folder.

With --IRL_log [LOG] (default: the newest real run): the real flight replayed
from the initial state of the real drones (logged positions and yaw, logged
phase times and setpoint period). Writes <log>_sim.csv next to the log, the
simulation figures in <run folder>/simulation/, and the comparison real vs sim
(figures + metrics.json) in <run folder>/comparison/.

Built on the Crazyflie plant of the thesis simulation (V8), copied into
this folder (cf_plant.py + drone_models.toml, self-contained): same parameters
(Param, drone model cf2x_P250), same drone state (DroneState), same inner loop
(drone_motion: Mellinger controller + full nonlinear quadrotor dynamics, RK4
on SO(3)), and the initial state taken from the real log as in V8 main.py
(get_initial_real_offset). The outer loop (the V8 Fibonacci
encirclement guidance) is replaced by the setpoints of the flight code
(cf_funcs.ramp_to: smoothstep ramps sent with send_position_setpoint):

    Take off  ramp from the measured position to (home x, home y, TAKEOFF_HEIGHT), TAKEOFF_TIME
    Hover     ramp from the measured position to (home x, home y, TAKEOFF_HEIGHT), HOVER_TIME
    Return    ramp to the assigned takeoff point (Hungarian, at TAKEOFF_HEIGHT),
              max(1.5 * max distance / RETURN_SPEED, 2 s)
    Land      ramp from the measured position to (x, y, LAND_HEIGHT), LAND_TIME, motors off at EOF

Each phase starts at the time logged in the flight log, with the setpoint
period measured in it (ramp_to sleeps SETPOINT_PERIOD after every setpoint,
plus the loop overhead). The drones start from their logged position, at
rest on the floor (ground contact: the drone rests on its starting height
until the thrust lifts it).

Comparison metrics (per drone, in comparison/metrics.json):
  * position difference sim - real: RMS per axis, 3D RMS and max, per phase
  * take off: lift-off time (z > floor + 2 cm), time to 90 % of the climb,
    overshoot above the hover height
  * hover (last 5 s before the return): mean height, height error to the setpoint (bias),
    horizontal and vertical jitter (std), horizontal drift from home
  * landing: touchdown time (z < LAND_HEIGHT + 1 cm), horizontal offset at
    touchdown, slide on the floor after it
  * setpoint tracking error |p - sp| of the real and the simulated drone
The real setpoints are rebuilt with the flight code's ramps from the real
measured positions (the log of this flight code has no setpoint columns).

Simulation figures (PDF + PNG): position, top_view.
Comparison figures (PDF + PNG): position, difference, takeoff_landing, summary.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())   # repository root
FLIGHT_CODE = HERE.name         # same name as its flight code, Crazy_Flie/Optitrack/<name>
LOGS_DIR = REPO / "Results"         # one folder per run: Results/<flight code>_trajectory_<date>/
sys.path.insert(0, str(HERE))

# the V8 Crazyflie plant, copied into this folder (cf_plant.py, drone_models.toml)
from cf_plant import (DroneState, drone_motion, get_initial_real_offset,  # noqa: E402
                      ground_contact, make_param, passive_motion)

# Flight code constants (swarm_take_off_hover_land/Structs.py, main.py)
TAKEOFF_HEIGHT = 0.6      # m
TAKEOFF_TIME = 3.0        # s
HOVER_TIME = 8.0          # s
RETURN_SPEED = 0.3        # m/s, peak speed back to the takeoff points
LAND_TIME = 4.0           # s
LAND_HEIGHT = 0.05        # m
SETPOINT_PERIOD = 0.05    # s (nominal; the measured one is used)
PAUSE_BEFORE_RETURN = 0.3  # s, time.sleep(0.3) before "Return to takeoff points" (main.py)
PAUSE_AFTER_LAND = 1.0    # s, time.sleep(1.0) between the landing and "EOF" (main.py)

RETURN = "Return to takeoff points"
PHASES = ("Take off", "Hover", RETURN, "Land")

DT = 0.01                 # s, simulation step (V8 param.sim.dTi)
POS_COLS = ["opti_x", "opti_y", "opti_z"]


# =============================================================================
# Real flight log
# =============================================================================

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

# =============================================================================
# Simulation
# =============================================================================

def simulate(log_path, ids, drones, phases, t_end):
    """The flight of every drone (independent apart from the return assignment)."""
    # -------------------------------------------------------------------------
    # Simulation initialization
    # -------------------------------------------------------------------------
    param = make_param(log_path, len(ids), t_end, DT)
    sched = setpoint_schedule(phases, drones)
    t_takeoff, t_stop = phases["Take off"], phases["EOF"]

    # -------------------------------------------------------------------------
    # Drone initialization: real state at the take off start (V8 main.py),
    # position from OptiTrack (the frame of the comparison), at rest
    # -------------------------------------------------------------------------
    if log_path:
        _, _, attitude_list, _, _ = get_initial_real_offset(param, phase="Take off")
        if len(ids) == 1:
            attitude_list = [attitude_list]
    else:                                    # nominal flight: yaw 0
        attitude_list = [np.eye(3).reshape(-1, 1) for _ in ids]
    homes = homes_of(drones)
    agents, floors = {}, {}
    for i, att in zip(ids, attitude_list):
        a = DroneState(param)
        a.position = homes[i].reshape(3, 1).copy()
        a.velocity = np.zeros((3, 1))
        yaw = Rotation.from_matrix(np.asarray(att, float).reshape(3, 3)).as_euler("ZYX")[0]
        a.attitude = Rotation.from_euler("z", yaw).as_matrix().reshape(-1, 1)   # level on the floor
        a.ang_vel = np.zeros((3, 1))
        a.desired.position = a.position.copy()
        a.desired.psi = a.desired.old_psi = 0.0      # send_position_setpoint(..., yaw=0)
        agents[i], floors[i] = a, float(homes[i][2])

    # -------------------------------------------------------------------------
    # Logs initialization
    # -------------------------------------------------------------------------
    # setpoint events: (time, phase, step index, n)
    events = sorted((t0 + k * period, name, k, n) for name, t0, n, period in sched for k in range(n))
    starts, goals = {}, {}                   # measured positions / goals at each phase start
    sp = {i: None for i in ids}
    log = []

    # -------------------------------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------------------------------
    t = min(0.0, t_takeoff)
    ev = 0
    while t <= t_end:
        # flight code: setpoints due at this time
        while ev < len(events) and events[ev][0] <= t + 1e-9:
            _, name, k, n = events[ev]
            if k == 0:                       # get_position() at the ramp start
                starts[name] = {i: agents[i].position.ravel().copy() for i in ids}
                goals[name] = phase_goals(name, homes, starts[name])
            for i in ids:
                sp[i] = ramp_setpoint(starts[name][i], goals[name][i], k, n)
            ev += 1
        phase = max((p for p in phases.items() if p[1] <= t + 1e-9 and p[0] != "EOF"),
                    key=lambda p: p[1], default=("", 0))[0] if t < t_stop else "EOF"
        for i in ids:
            a = agents[i]
            if t >= t_stop:
                passive_motion(a, param)                 # stop_flight: motors off
            elif t >= t_takeoff and sp[i] is not None:
                a.desired.position = sp[i].reshape(3, 1).copy()
                drone_motion(a, param)
            # before the first setpoint: armed, resting on the floor
            ground_contact(a, floors[i])
            roll, pitch, yaw = Rotation.from_matrix(a.attitude.reshape(3, 3)).as_euler("xyz", degrees=True)
            s = sp[i] if sp[i] is not None else np.full(3, np.nan)
            log.append([t + DT, phase, i, *a.position.ravel(), *a.velocity.ravel(),
                        roll, pitch, yaw, *s])
        t += DT

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    cols = ["time_s", "flight_state", "id", *POS_COLS, "est_vx", "est_vy", "est_vz",
            "roll_deg", "pitch_deg", "yaw_deg", "sp_x", "sp_y", "sp_z"]
    return pd.DataFrame(log, columns=cols), sched


PHASE_COLORS = {"Take off": "#cfe8ff", "Hover": "#d9f2d0", RETURN: "#f3e1f7", "Land": "#ffe0c7"}
SHORT = {"Take off": "Take off", "Hover": "Hover", RETURN: "Return", "Land": "Land"}


def interp_rows(t_src, X, t):
    return np.column_stack([np.interp(t, t_src, X[:, a], left=np.nan, right=np.nan) for a in range(X.shape[1])])


def load_sim(sim_path, ids):
    df = pd.read_csv(sim_path, dtype={"id": str, "flight_state": str})
    df["id"] = df["id"].str.replace(r"\.0$", "", regex=True)
    out = {}
    for i in ids:
        g = df[df["id"] == i].sort_values("time_s")
        out[i] = {"t": g["time_s"].to_numpy(float), "p": g[["opti_x", "opti_y", "opti_z"]].to_numpy(float),
                  "sp": g[["sp_x", "sp_y", "sp_z"]].to_numpy(float)}
    return out


def first_time(t, cond):
    k = np.flatnonzero(cond)
    return float(t[k[0]]) if len(k) else float("nan")


def drone_metrics(t, p, sp, phases, floor, home):
    """Event / phase metrics of one trajectory (real or sim)."""
    t_to, t_h, t_l, t_e = phases["Take off"], phases.get("Hover", np.nan), phases.get("Land", np.nan), phases["EOF"]
    m = {}
    z = p[:, 2]
    after = t >= t_to
    m["liftoff_s"] = first_time(t, after & (z > floor + 0.02)) - t_to
    m["t90_climb_s"] = first_time(t, after & (z > floor + 0.9 * (TAKEOFF_HEIGHT - floor))) - t_to
    if np.isfinite(t_h):
        w = (t >= t_h) & (t <= t_h + 3.0)
        m["overshoot_m"] = float(np.nanmax(z[w]) - TAKEOFF_HEIGHT) if w.any() else float("nan")
    t_he = phases.get(RETURN, t_l)       # end of the hover (+ pause)
    if np.isfinite(t_he) and np.isfinite(t_h):
        w = (t >= max(t_h, t_he - PAUSE_BEFORE_RETURN - 5.0)) & (t <= t_he - PAUSE_BEFORE_RETURN)
        if w.any():
            m["hover_mean_z_m"] = float(np.nanmean(z[w]))
            m["hover_z_bias_m"] = float(np.nanmean(z[w] - sp[w, 2]))
            m["hover_z_std_mm"] = float(1e3 * np.nanstd(z[w]))
            xy = p[w, :2]
            m["hover_xy_std_mm"] = float(1e3 * np.sqrt(np.nanmean(np.sum((xy - np.nanmean(xy, 0)) ** 2, 1))))
            m["hover_xy_drift_mm"] = float(1e3 * np.linalg.norm(np.nanmean(xy, 0) - home[:2]))
    if np.isfinite(t_l):
        # the drones settle on the LAND_HEIGHT setpoint (motors still on until EOF), not on the floor
        t_td = first_time(t, (t >= t_l) & (z < LAND_HEIGHT + 0.01))
        m["touchdown_s"] = t_td - t_l
        if np.isfinite(t_td):
            q = p[np.searchsorted(t, t_td)]
            m["landing_xy_offset_mm"] = float(1e3 * np.linalg.norm(q[:2] - home[:2]))
        # after touchdown (motors on, on the floor): sliding / tipping
        w = t >= (t_td if np.isfinite(t_td) else t_e)
        if w.sum() > 1:
            pw = p[w][np.all(np.isfinite(p[w]), 1)]
            m["slide_after_touchdown_mm"] = float(1e3 * np.max(np.linalg.norm(pw[:, :2] - pw[0, :2], axis=1)))
    for ph in PHASES:
        w = phase_mask(t, phases, ph)
        e = np.linalg.norm(p[w] - sp[w], axis=1)
        e = e[np.isfinite(e)]
        if len(e):
            m[f"tracking_rms_{SHORT[ph]}_mm"] = float(1e3 * np.sqrt(np.mean(e ** 2)))
    return m


def phase_mask(t, phases, ph):
    order = [p for p in PHASES if p in phases]
    if ph not in order:
        return np.zeros(len(t), bool)
    k = order.index(ph)
    t1 = phases[order[k + 1]] if k + 1 < len(order) else phases["EOF"]
    return (t >= phases[ph]) & (t < t1)


def shade(ax, phases):
    order = [p for p in PHASES if p in phases]
    for k, ph in enumerate(order):
        t1 = phases[order[k + 1]] if k + 1 < len(order) else phases["EOF"]
        ax.axvspan(phases[ph], t1, color=PHASE_COLORS[ph], alpha=0.6, lw=0, label=SHORT[ph])


def compare(log_path, sim_path=None, out_dir=None):
    log_path = Path(log_path)
    sim_path = Path(sim_path) if sim_path else log_path.with_name(log_path.stem + "_sim.csv")
    ids, real, phases, t_end = load_log(log_path)
    sim = load_sim(sim_path, ids)
    sched = setpoint_schedule(phases, real)
    out_dir = Path(out_dir) if out_dir else log_path.parent / "comparison"     # in the run folder
    out_dir.mkdir(parents=True, exist_ok=True)

    t_from, t_to = phases["Take off"], phases["EOF"] + 0.5
    res = {"log": str(log_path), "sim": str(sim_path), "phases_s": phases, "drones": {}}
    data = {}
    times = {i: real[i]["t"][(real[i]["t"] >= t_from - 0.5) & (real[i]["t"] <= t_to)] for i in ids}
    sp_real = real_setpoints(real, sched, times)
    for i in ids:
        r = real[i]
        w = (r["t"] >= t_from - 0.5) & (r["t"] <= t_to)
        t, p = r["t"][w], r["p"][w]
        home, floor = r["p"][0], float(r["p"][0, 2])
        sp_r = sp_real[i]
        p_s = interp_rows(sim[i]["t"], sim[i]["p"], t)
        sp_s = interp_rows(sim[i]["t"], sim[i]["sp"], t)
        data[i] = dict(t=t, p=p, sp=sp_r, ps=p_s, sps=sp_s)
        d = {"home": home.tolist()}
        e = p_s - p
        for ph in ("all", *PHASES):
            m = (t >= t_from) & (t <= phases["EOF"]) if ph == "all" else phase_mask(t, phases, ph)
            ee = e[m]
            ee = ee[np.all(np.isfinite(ee), 1)]
            if len(ee):
                d[f"diff_{SHORT.get(ph, ph)}"] = {"rms_xyz_mm": (1e3 * np.sqrt(np.mean(ee ** 2, 0))).round(1).tolist(),
                                   "rms_3d_mm": round(float(1e3 * np.sqrt(np.mean(np.sum(ee ** 2, 1)))), 1),
                                   "max_3d_mm": round(float(1e3 * np.max(np.linalg.norm(ee, axis=1))), 1)}
        d["real"] = drone_metrics(t, p, sp_r, phases, floor, home)
        d["sim"] = drone_metrics(t, p_s, sp_s, phases, floor, home)
        res["drones"][i] = d

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print_report(res)
    plot(ids, data, phases, res, out_dir, log_path.stem)
    print(f"Metrics and figures saved to {out_dir}")
    return res


def print_report(res):
    print("\nSimulation vs real flight: position difference (sim - real)")
    print(f"  {'drone':6s} {'phase':9s} {'RMS x/y/z [mm]':>22s} {'RMS 3D':>8s} {'max 3D':>8s}")
    for i, d in res["drones"].items():
        for ph in ("all", *(SHORT[q] for q in PHASES)):
            if f"diff_{ph}" in d:
                q = d[f"diff_{ph}"]
                x, y, z = q["rms_xyz_mm"]
                print(f"  CF{int(i):02d}   {ph:9s} {x:7.1f} {y:7.1f} {z:7.1f}  {q['rms_3d_mm']:8.1f} {q['max_3d_mm']:8.1f}")
    keys = [("liftoff_s", "lift-off after the takeoff start [s]", 2), ("t90_climb_s", "90 % of the climb [s]", 2),
            ("overshoot_m", "overshoot above 0.6 m [mm]", -3), ("hover_mean_z_m", "hover height [m]", 3),
            ("hover_z_bias_m", "hover height - setpoint [mm]", -3), ("hover_z_std_mm", "hover z jitter (std) [mm]", 1),
            ("hover_xy_std_mm", "hover xy jitter (std) [mm]", 1), ("hover_xy_drift_mm", "hover xy offset from home [mm]", 1),
            ("touchdown_s", "down to 6 cm after the landing start [s]", 2),
            ("landing_xy_offset_mm", "xy offset from home at touchdown [mm]", 1),
            ("slide_after_touchdown_mm", "slide on the floor after touchdown [mm]", 1),
            ("tracking_rms_Take off_mm", "setpoint tracking RMS, take off [mm]", 1),
            ("tracking_rms_Hover_mm", "setpoint tracking RMS, hover [mm]", 1),
            ("tracking_rms_Return_mm", "setpoint tracking RMS, return [mm]", 1),
            ("tracking_rms_Land_mm", "setpoint tracking RMS, landing [mm]", 1)]
    print("\n  real / sim" + "".join(f"{'CF%02d' % int(i):>20s}" for i in res["drones"]))
    for k, label, nd in keys:
        row = f"  {label:42s}"
        for d in res["drones"].values():
            a, b = d["real"].get(k, np.nan), d["sim"].get(k, np.nan)
            if nd < 0:      # metres shown in mm
                a, b, nd2 = 1e3 * a, 1e3 * b, 1
            else:
                nd2 = nd
            row += f"{a:9.{nd2}f} / {b:<8.{nd2}f}"
        print(row)


def plot(ids, data, phases, res, out_dir, title):
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    figs = {}

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    for a, ax in enumerate(axes):
        shade(ax, phases)
        for k, i in enumerate(ids):
            d = data[i]
            ax.plot(d["t"], d["p"][:, a], color=colors(k), lw=1.8, label=f"CF{int(i):02d} real")
            ax.plot(d["t"], d["ps"][:, a], color=colors(k), lw=1.3, ls="--", label=f"CF{int(i):02d} sim")
            ax.plot(d["t"], d["sp"][:, a], color=colors(k), lw=0.8, ls=":", alpha=0.8,
                    label=f"CF{int(i):02d} setpoint" if a == 0 else None)
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(h, lab, ncol=4, fontsize=7, loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title}: position, real (solid) vs simulation (dashed), setpoint (dotted)")
    fig.tight_layout()
    figs["position"] = fig

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    for ax in axes:
        shade(ax, phases)
    for k, i in enumerate(ids):
        d = data[i]
        e = d["ps"] - d["p"]
        axes[0].plot(d["t"], 1e3 * np.linalg.norm(e, axis=1), color=colors(k), lw=1.5, label=f"CF{int(i):02d} |sim - real|")
        axes[1].plot(d["t"], 1e3 * np.linalg.norm(e[:, :2], axis=1), color=colors(k), lw=1.3, label=f"CF{int(i):02d} horizontal")
        axes[1].plot(d["t"], 1e3 * e[:, 2], color=colors(k), lw=1.3, ls="--", label=f"CF{int(i):02d} vertical (z)")
    axes[0].set_ylabel("3D difference [mm]")
    axes[1].set_ylabel("difference [mm]")
    axes[1].axhline(0, color="k", lw=0.6)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title}: simulation - real")
    fig.tight_layout()
    figs["difference"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    wins = [(phases["Take off"] - 0.3, phases.get("Hover", phases["Take off"] + 3) + 2.5, "take off"),
            (phases.get("Land", phases["EOF"] - 4) - 0.8, phases["EOF"] + 0.5, "landing")]
    for ax, (a, b, name) in zip(axes, wins):
        shade(ax, phases)
        for k, i in enumerate(ids):
            d = data[i]
            m = (d["t"] >= a) & (d["t"] <= b)
            ax.plot(d["t"][m], d["p"][m, 2], color=colors(k), lw=1.8, label=f"CF{int(i):02d} real")
            ax.plot(d["t"][m], d["ps"][m, 2], color=colors(k), lw=1.3, ls="--", label=f"CF{int(i):02d} sim")
            ax.plot(d["t"][m], d["sp"][m, 2], color=colors(k), lw=0.8, ls=":")
        ax.set_xlim(a, b)
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    figs["takeoff_landing"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["all", *(SHORT[p] for p in PHASES if p in phases)]
    width = 0.8 / len(ids)
    for k, i in enumerate(ids):
        d = res["drones"][i]
        vals = [d.get(f"diff_{ph}", {}).get("rms_3d_mm", np.nan) for ph in labels]
        ax.bar(np.arange(len(labels)) + (k - (len(ids) - 1) / 2) * width, vals, width, color=colors(k),
               label=f"CF{int(i):02d}")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("RMS 3D difference sim - real [mm]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title(f"{title}: simulation vs real, per phase")
    fig.tight_layout()
    figs["summary"] = fig

    for name, f in figs.items():
        for ext in ("pdf", "png"):
            f.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)


# =============================================================================
# Simulation figures
# =============================================================================

def plot_sim(sim_df, phases, out_dir, title):
    """Simulated flight alone: position and setpoint per axis, top view."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(sim_df["id"].astype(str).unique(), key=lambda s: (len(s), s))
    colors = plt.get_cmap("tab10")
    figs = {}

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    for a, ax in enumerate(axes):
        shade(ax, phases)
        for k, i in enumerate(ids):
            g = sim_df[sim_df["id"].astype(str) == i]
            ax.plot(g["time_s"], g[POS_COLS[a]], color=colors(k), lw=1.5, label=f"CF{int(i):02d} sim")
            ax.plot(g["time_s"], g["sp_" + "xyz"[a]], color=colors(k), lw=0.8, ls=":",
                    label=f"CF{int(i):02d} setpoint" if a == 0 else None)
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, fontsize=7, loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title}: simulated position (solid), setpoint (dotted)")
    fig.tight_layout()
    figs["position"] = fig

    fig, ax = plt.subplots(figsize=(7, 6))
    for k, i in enumerate(ids):
        g = sim_df[sim_df["id"].astype(str) == i]
        ax.plot(g["opti_x"], g["opti_y"], color=colors(k), lw=1.5, label=f"CF{int(i):02d}")
        ax.plot(g["opti_x"].iloc[0], g["opti_y"].iloc[0], "o", color=colors(k), ms=6)
        ax.plot(g["opti_x"].iloc[-1], g["opti_y"].iloc[-1], "x", color=colors(k), ms=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{title}: top view (o start, x end)")
    fig.tight_layout()
    figs["top_view"] = fig

    for name, f in figs.items():
        for ext in ("pdf", "png"):
            f.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    print(f"Simulation figures saved to {out_dir}")


# =============================================================================
# Main
# =============================================================================

def newest_log():
    """Flight log of the newest run: Results/swarm_take_off_hover_land_trajectory_<date>/trajectory_<date>.csv."""
    prefix = f"{FLIGHT_CODE}_"
    for run in sorted(LOGS_DIR.glob(f"{FLIGHT_CODE}_trajectory_*/"), reverse=True):
        log = run / f"{run.name[len(prefix):]}.csv"
        if log.exists():
            return log
    raise SystemExit(f"No {FLIGHT_CODE} run in {LOGS_DIR}")

def nominal_flight(n_drones, spacing):
    """
    Flight of the flight code without a log: drones on the floor in a row
    (`spacing` apart, along x), nominal phase times (every ramp at
    SETPOINT_PERIOD, the pauses of main.py, the return from the hover).
    Same structure as load_log; the positions are the nominal ones at the phase starts.
    """
    ids = [str(k + 1) for k in range(n_drones)]
    homes = {i: np.array([(k - (n_drones - 1) / 2) * spacing, 0.0, 0.0]) for k, i in enumerate(ids)}
    hover = {i: np.array([*h[:2], TAKEOFF_HEIGHT]) for i, h in homes.items()}
    phases = {"Take off": 0.0, "Hover": TAKEOFF_TIME}
    phases[RETURN] = phases["Hover"] + HOVER_TIME + PAUSE_BEFORE_RETURN
    phases["Land"] = phases[RETURN] + return_spots(homes, hover)[1]
    phases["EOF"] = phases["Land"] + LAND_TIME + PAUSE_AFTER_LAND
    t_end = phases["EOF"] + 1.0
    drones = {i: {"t": np.array([0.0, phases["Hover"], t_end]),
                  "p": np.array([homes[i], hover[i], hover[i]]),
                  "yaw_deg": np.zeros(3)} for i in ids}
    return ids, drones, phases, t_end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--IRL_log", nargs="?", const="", metavar="LOG",
                    help="start from the initial values of a real flight and compare with it "
                         f"(LOG: its flight log; default: the newest {FLIGHT_CODE} run in Results/)")
    ap.add_argument("--drones", type=int, default=3, help="without --IRL_log: number of drones")
    ap.add_argument("--spacing", type=float, default=0.5, help="without --IRL_log: distance between the takeoff points, m")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.IRL_log is None:
        # ---------------------------------------------------------------------
        # Simulation only: simulated initial values
        # ---------------------------------------------------------------------
        ids, drones, phases, t_end = nominal_flight(args.drones, args.spacing)
        print(f"Nominal flight, drones {ids}; phases " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))
        sim_df, _ = simulate(None, ids, drones, phases, t_end)
        run_dir = LOGS_DIR / f"{FLIGHT_CODE}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(run_dir / "sim.csv", index=False)
        print(f"Simulated flight saved to {run_dir / 'sim.csv'}")
        plot_sim(sim_df, phases, run_dir / "simulation", f"Nominal flight, {len(ids)} drones")
    else:
        # ---------------------------------------------------------------------
        # Initial values of the real flight, then the comparison with it
        # ---------------------------------------------------------------------
        log_path = Path(args.IRL_log).resolve() if args.IRL_log else newest_log()
        print(f"Log: {log_path}")
        ids, drones, phases, t_end = load_log(log_path)
        print(f"Drones {ids}; phases " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))
        sim_df, sched = simulate(log_path, ids, drones, phases, t_end)
        for name, t0, n, period in sched:
            print(f"  {name:24s}: {n} setpoints from {t0:.2f} s, measured period {1e3 * period:.1f} ms")
        out = log_path.with_name(log_path.stem + "_sim.csv")
        sim_df.to_csv(out, index=False)
        print(f"Simulated flight saved to {out}")
        plot_sim(sim_df, phases, log_path.parent / "simulation", log_path.stem)
        compare(log_path, out)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
