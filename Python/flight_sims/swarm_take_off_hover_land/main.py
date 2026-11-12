"""
Swarm take off / hover / return / land: the simulation, and the replay of its
real flight (Crazy_Flie/Optitrack/swarm_take_off_hover_land) for sim-vs-real.
Same structure as ../swarm_geometric_Fibonacci_sim_target/main.py and ../8traj/main.py.

    python swarm_take_off_hover_land/main.py                    # the simulation alone: the Main Controller (hover), Structs.py
    python swarm_take_off_hover_land/main.py --IRL_log          # replay of the newest real run in Results/, all flight stages
    python swarm_take_off_hover_land/main.py --IRL_log ../../Results/swarm_take_off_hover_land_trajectory_<date>/trajectory_<date>.csv
    python swarm_take_off_hover_land/main.py --IRL_log --no-show

Without --IRL_log: only the Main Controller (the hover), nothing from the
flight code or a real run: every value from Structs.py (param.sim: nD, Tend,
dTi, drone model; param.hover: hover height and spacing of the homes; the
Mellinger gains of ControllerConfig). The drones start at rest at their hover
points and hold them. Writes Results/swarm_take_off_hover_land_<date>/
(sim.csv and the figures in simulation/).

With --IRL_log [LOG] (default: the newest real run): the real flight replayed
from the initial state of the real drones (logged positions and yaw, logged
phase times and setpoint period), all flight stages (take off, hover, return,
land) with the setpoints of the flight code (flight_code_funcs.py). Writes
<log>_sim.csv next to the log (what video_overlay/flight_overlay.py draws over
the phone video), the figures in <run folder>/simulation/, and the comparison
real vs sim (compare_real.py: figures + metrics.json) in <run folder>/comparison/.

Both save Results/simulation_results.pkl (python Animation_display.py redraws the
figures / animations from it) and draw the figures of
Animation_display.create_figures (GraphicalDisplay flags in Structs.py).

Modules (the thesis simulation, V8):
    Structs.py            Param (drone model cf2x_P250, Mellinger gains, hover), DroneState, SimulationData
    drone_motion.py       inner loop: Mellinger controller (drone_mellinger_ctrl.py) + full nonlinear
                          dynamics (drone_3dfull_dyn.py), RK4 on SO(3); passive_motion (motors off),
                          ground_contact
    helper_funcs.py       rotations, RK4, get_initial_real_offset
    Animation_display.py  figures and animations
and, for the real flight:
    flight_code_funcs.py  the flight code: constants, log, setpoint schedule and ramps
    compare_real.py       real vs sim metrics and figures
"""

import argparse
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Structs import Param, DroneState, SimulationData  # noqa: E402
from drone_motion import drone_motion, passive_motion, ground_contact  # noqa: E402
from helper_funcs import get_initial_real_offset  # noqa: E402
from flight_code_funcs import *  # noqa: E402,F401,F403
from compare_real import compare  # noqa: E402

DT = 0.01                       # s, plant step of the replay (as param.sim.dTi of Structs.py)
PKL_FILE = HERE / "Results" / "simulation_results.pkl"   # the local Results folder
CSV_COLS = ["time_s", "flight_state", "id", *POS_COLS, "est_vx", "est_vy", "est_vz",
            "roll_deg", "pitch_deg", "yaw_deg", "sp_x", "sp_y", "sp_z"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--IRL_log", nargs="?", const="", metavar="LOG",
                    help="replay a real flight from its initial values, all flight stages, and compare with it "
                         f"(LOG: its flight log; default: the newest {FLIGHT_CODE} run in Results/). "
                         "Without it: the Main Controller (hover) alone, with the parameters of Structs.py")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Animation_display import create_figures

    if args.IRL_log is None:
        data, param = simulate_isolated()
    else:
        data, param = simulate_flight(args)

    # =========================================================================
    # Results
    # =========================================================================
    data["colors"] = plt.cm.tab10(np.linspace(0, 1, 10))[:param.sim.nD]
    if param.display.store_data == True:
        print("Saving simulation data (.pkl)")
        PKL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PKL_FILE, "wb") as f:
            pickle.dump({**data, "sim": {"nD": param.sim.nD, "dTi": param.sim.dTi, "Nsim": param.sim.Nsim,
                                         "Tend": param.sim.Tend, "real_trajectory_csv": param.sim.real_trajectory_csv},
                         "save_location": param.display.save_location}, f)
        shutil.copy2(PKL_FILE, Path(param.display.save_location) / PKL_FILE.name)   # kept with the run

    if param.display.verbose:
        sim_data = SimulationData(**data)
        create_figures(sim_data, param)          # with --IRL_log also shows the comparison figures (q to quit)


def attitude_rpy(att):
    return Rotation.from_matrix(np.asarray(att, float).reshape(3, 3)).as_euler("xyz", degrees=True)


# =============================================================================
# The simulation alone (no flags): the Main Controller (hover), Structs.py
# =============================================================================

def simulate_isolated():
    # =========================================================================
    # Simulation initialization
    # =========================================================================
    param = Param()                              # real_trajectory_csv None: nothing from a real run
    out_dir = LOGS_DIR / f"{FLIGHT_CODE}_{datetime.now():%Y%m%d_%H%M%S}"
    param.display.save_location = str(out_dir / "simulation")
    Path(param.display.save_location).mkdir(parents=True, exist_ok=True)
    ids = [str(k + 1) for k in range(param.sim.nD)]
    points = param.hover.points(param.sim.nD)
    print(f"Simulation: Main Controller (hover), {param.sim.nD} drones at {param.hover.height} m, "
          f"{param.hover.spacing} m apart, Tend {param.sim.Tend} s, dTi {param.sim.dTi} s")

    # =========================================================================
    # Drone initialization: at rest at the hover points (yaw 0)
    # =========================================================================
    drone_list: list[DroneState] = []
    for iD in range(param.sim.nD):
        drone = DroneState(param)
        drone.position = points[iD].reshape(3, 1).copy()
        drone.desired.position = points[iD].reshape(3, 1).copy()      # hold above the home
        drone_list.append(drone)

    # =========================================================================
    # Logs initialization
    # =========================================================================
    Nsim = param.sim.Nsim
    position_history = np.zeros([param.sim.nD, 3, Nsim])
    p_d_hist = np.zeros((param.sim.nD, 3, Nsim))
    rows = []

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    for k in range(Nsim):
        current_time = param.sim.dTi * k
        for iD, drone in enumerate(drone_list):
            drone_motion(drone, param)
            position_history[iD, :, k:k + 1] = drone.position
            p_d_hist[iD, :, k:k + 1] = drone.desired.position
            rows.append([current_time + param.sim.dTi, "Hover", ids[iD], *drone.position.ravel(),
                         *drone.velocity.ravel(), *attitude_rpy(drone.attitude), *drone.desired.position.ravel()])

    # =========================================================================
    # Results
    # =========================================================================
    out = out_dir / "sim.csv"
    pd.DataFrame(rows, columns=CSV_COLS).to_csv(out, index=False)
    print(f"Simulated flight saved to {out}")
    data = {"position_history": position_history, "p_d_hist": p_d_hist, "ids": ids, "phases": {}}
    return data, param


# =============================================================================
# Replay of a real flight (--IRL_log): all flight stages
# =============================================================================

def simulate_flight(args):
    # =========================================================================
    # Flight log
    # =========================================================================
    print("Flight code: " + ", ".join(f"{k}={v:g}" for k, v in flight_constants().items()))
    log_path = Path(args.IRL_log).resolve() if args.IRL_log else newest_log()
    print(f"Log: {log_path}")
    ids, drones, phases, t_end = load_log(log_path)
    out_dir = log_path.parent
    sim_csv = log_path.with_name(log_path.stem + "_sim.csv")
    print(f"Drones {ids}; phases " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))

    # =========================================================================
    # Simulation initialization
    # =========================================================================
    param = configure_param(Param(), len(ids), t_end, DT, log_path)
    param.display.save_location = str(out_dir / "simulation")
    Path(param.display.save_location).mkdir(parents=True, exist_ok=True)
    sched = setpoint_schedule(phases, drones)
    for name, t0, n, period in sched:
        print(f"  {name:24s}: {n} setpoints from {t0:.2f} s, measured period {1e3 * period:.1f} ms")
    t_takeoff, t_stop = phases["Take off"], phases["EOF"]

    # =========================================================================
    # Drone initialization: real state at the take off start (V8 main.py),
    # position from OptiTrack (the frame of the comparison), at rest
    # =========================================================================
    _, _, attitude_list, _, _ = get_initial_real_offset(param, phase="Take off")
    if len(ids) == 1:
        attitude_list = [attitude_list]
    homes = homes_of(drones)
    drone_list: list[DroneState] = []
    for i, att in zip(ids, attitude_list):
        drone = DroneState(param)
        drone.position = homes[i].reshape(3, 1).copy()
        yaw = Rotation.from_matrix(np.asarray(att, float).reshape(3, 3)).as_euler("ZYX")[0]
        drone.attitude = Rotation.from_euler("z", yaw).as_matrix().reshape(-1, 1)   # level on the floor
        drone.desired.position = drone.position.copy()
        drone.desired.psi = drone.desired.old_psi = 0.0     # send_position_setpoint(..., yaw=0)
        drone_list.append(drone)
    floors = {i: float(homes[i][2]) for i in ids}

    # =========================================================================
    # Logs initialization
    # =========================================================================
    Nsim = param.sim.Nsim
    position_history = np.zeros([param.sim.nD, 3, Nsim])
    p_d_hist = np.full((param.sim.nD, 3, Nsim), np.nan)
    # setpoint events: (time, phase, step index, n)
    events = sorted((t0 + k * period, name, k, n) for name, t0, n, period in sched for k in range(n))
    starts, goals = {}, {}                       # measured positions / goals at each phase start
    sp = {i: None for i in ids}
    rows = []

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    ev = 0
    for k in range(Nsim):
        current_time = param.sim.dTi * k

        # flight code: setpoints due at this time
        while ev < len(events) and events[ev][0] <= current_time + 1e-9:
            _, name, j, n = events[ev]
            if j == 0:                           # get_position() at the ramp start
                starts[name] = {i: d.position.ravel().copy() for i, d in zip(ids, drone_list)}
                goals[name] = phase_goals(name, homes, starts[name])
            for i in ids:
                sp[i] = ramp_setpoint(starts[name][i], goals[name][i], j, n)
            ev += 1
        phase = max((p for p in phases.items() if p[1] <= current_time + 1e-9 and p[0] != "EOF"),
                    key=lambda p: p[1], default=("", 0))[0] if current_time < t_stop else "EOF"

        for iD, (i, drone) in enumerate(zip(ids, drone_list)):
            if current_time >= t_stop:
                passive_motion(drone, param)                 # stop_flight: motors off
            elif current_time >= t_takeoff and sp[i] is not None:
                drone.desired.position = sp[i].reshape(3, 1).copy()
                drone_motion(drone, param)
            # before the first setpoint: armed, resting on the floor
            ground_contact(drone, floors[i])

            position_history[iD, :, k:k + 1] = drone.position
            s = sp[i] if sp[i] is not None else np.full(3, np.nan)
            p_d_hist[iD, :, k] = s
            rows.append([current_time + DT, phase, i, *drone.position.ravel(), *drone.velocity.ravel(),
                         *attitude_rpy(drone.attitude), *s])

    # =========================================================================
    # Results
    # =========================================================================
    pd.DataFrame(rows, columns=CSV_COLS).to_csv(sim_csv, index=False)
    print(f"Simulated flight saved to {sim_csv}")
    compare(log_path, sim_csv)
    data = {"position_history": position_history, "p_d_hist": p_d_hist, "ids": list(ids), "phases": phases}
    return data, param


if __name__ == "__main__":
    main()
