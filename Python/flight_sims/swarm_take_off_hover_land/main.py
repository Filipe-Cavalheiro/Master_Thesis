"""
Simulation of a swarm take off / hover / return / land flight
(Crazy_Flie/Optitrack/swarm_take_off_hover_land), to compare with its log.

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

    python swarm_take_off_hover_land/main.py                       # newest swarm_take_off_hover_land run in Results/
    python swarm_take_off_hover_land/main.py --log ../../Results/swarm_take_off_hover_land_trajectory_20261026_102741/trajectory_20261026_102741.csv
    python swarm_take_off_hover_land/main.py --no-show
    python swarm_take_off_hover_land/main.py --sim --drones 3     # no log: the nominal flight only

Comparison (default): writes <log>_sim.csv next to the flight log, in its
run folder Results/swarm_take_off_hover_land_trajectory_<date>/ (same columns
as the log, plus the setpoints), and runs compare.py (metrics + figures in
<run folder>/comparison/).
Simulation only (--sim): the nominal flight of --drones drones starting on
the floor in a row, saved to Results/swarm_take_off_hover_land_<date>/sim.csv.
"""

import argparse
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
    ap.add_argument("--log", help=f"flight log (default: newest {FLIGHT_CODE} run in Results/)")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    ap.add_argument("--sim", action="store_true", help="no log, no comparison: simulate the nominal flight")
    ap.add_argument("--drones", type=int, default=3, help="--sim: number of drones")
    ap.add_argument("--spacing", type=float, default=0.5, help="--sim: distance between the takeoff points, m")
    args = ap.parse_args()

    if args.sim:
        ids, drones, phases, t_end = nominal_flight(args.drones, args.spacing)
        print(f"Nominal flight, drones {ids}; phases " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))
        sim_df, _ = simulate(None, ids, drones, phases, t_end)
        out_dir = LOGS_DIR / f"{FLIGHT_CODE}_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(out_dir / "sim.csv", index=False)
        print(f"Simulated flight saved to {out_dir / 'sim.csv'}")
        return

    log_path = Path(args.log).resolve() if args.log else newest_log()
    print(f"Log: {log_path}")

    ids, drones, phases, t_end = load_log(log_path)
    print(f"Drones {ids}; phases " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))
    sim_df, sched = simulate(log_path, ids, drones, phases, t_end)
    for name, t0, n, period in sched:
        print(f"  {name:24s}: {n} setpoints from {t0:.2f} s, measured period {1e3 * period:.1f} ms")
    out = log_path.with_name(log_path.stem + "_sim.csv")
    sim_df.to_csv(out, index=False)
    print(f"Simulated flight saved to {out}")

    from compare import compare
    compare(log_path, out, show=not args.no_show)

if __name__ == "__main__":
    main()
