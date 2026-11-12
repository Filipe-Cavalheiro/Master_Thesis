"""
Figure-8 flight: the thesis simulation, and the replay of its real flight
(Crazy_Flie/Optitrack/8traj.py) for sim-vs-real.

    python 8traj/main.py                         # the simulation alone: the Main Controller, Structs.py
    python 8traj/main.py --IRL_log               # replay of the newest real run in Results/, all flight stages
    python 8traj/main.py --IRL_log ../../Results/8traj_trajectory_<date>/trajectory_<date>.csv
    python 8traj/main.py --IRL_log --return step --ease 0     # a flight of the flight code before 10 Nov
    python 8traj/main.py --IRL_log --no-show

Without --IRL_log: only the Main Controller (the figure 8), nothing from the
flight code or a real run: every value from Structs.py (param.sim: Tend, dTi,
drone model; param.traj: the figure 8 of TrajectoryConfig; the Mellinger gains
of ControllerConfig). The drone starts at rest at the start of the trajectory
and tracks its position and velocity (time_alignment.run_simulation). Writes
Results/8traj_<date>/ (sim.csv and the figures in simulation/).

With --IRL_log [LOG] (default: the newest real run): the real flight replayed
from the initial state of the real drone (logged position and yaw, logged
phase times and setpoint period; --return / --ease detected from the log
unless given), all flight stages (take off, figure 8, return, land), with the
setpoint stream of 8traj.py (flight_code_funcs.py). Writes <log>_sim.csv next
to the log (what video_overlay/flight_overlay.py draws over the phone video),
the figures in <run folder>/simulation/, and the comparison real vs sim over
the whole flight (compare_real.py) in <run folder>/comparison/.

Both save Results/simulation_results.pkl (python Animation_display.py redraws the
figures / animations from it) and draw the figures of
Animation_display.create_figures (GraphicalDisplay flags in Structs.py;
real_sim_comparison_* compare the Main Controller section, --IRL_log only).

Modules (the thesis simulation, as legacy_main.py):
    Structs.py            Param, DroneState, SimulationData
    agent_motion.py       inner loop: Mellinger controller + full nonlinear dynamics, RK4 on SO(3);
                          passive_motion (motors off), ground_contact
    time_alignment.py     run_simulation: tracking of param.traj
    helper_funcs.py       get_initial_real_offset
    Animation_display.py  figures and animations
and, for the real flight:
    flight_code_funcs.py  8traj.py: constants, log, the setpoint stream it sends
    compare_real.py       real vs sim metrics and figures, whole flight
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
from agent_motion import agent_motion, passive_motion, ground_contact  # noqa: E402
from helper_funcs import get_initial_real_offset  # noqa: E402
from time_alignment import run_simulation  # noqa: E402
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
                         "Without it: the Main Controller alone, with the parameters of Structs.py")
    ap.add_argument("--return", dest="ret", choices=("auto", "ramp", "step"), default="auto",
                    help="with --IRL_log: return_origin of the flight, smooth ramp (from 10 Nov) or the old step; "
                         "auto: detected from the log")
    ap.add_argument("--ease", type=float,
                    help="with --IRL_log: TRAJ_EASE_TIME of the flight [s] (default: detected from the log; "
                         "0 before 10 Nov)")
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
    data["colors"] = plt.cm.tab10(np.linspace(0, 1, param.sim.nD))
    if param.display.store_data == True:
        print("Saving simulation data (.pkl)")
        PKL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PKL_FILE, "wb") as f:
            pickle.dump({**data, "sim": {"nD": param.sim.nD, "dTi": param.sim.dTi, "Nsim": param.sim.Nsim,
                                         "Tend": param.sim.Tend, "real_trajectory_csv": param.sim.real_trajectory_csv},
                         "display": {k: getattr(param.display, k) for k in REAL_DISPLAY_FLAGS}}, f)
        shutil.copy2(PKL_FILE, Path(param.display.save_location) / PKL_FILE.name)   # kept with the run

    if param.display.verbose:
        sim_data = SimulationData(**data)
        create_figures(sim_data, param)          # with --IRL_log also shows the comparison figures (q to quit)


# figures that need a real flight (off without --IRL_log)
REAL_DISPLAY_FLAGS = ("real_trajectory_plot", "real_sim_comparison_plot", "real_sim_comparison_animate",
                      "trajectory_plot")


def attitude_rpy(att):
    return Rotation.from_matrix(np.asarray(att, float).reshape(3, 3)).as_euler("xyz", degrees=True)


# =============================================================================
# The simulation alone (no flags): the Main Controller, Structs.py
# =============================================================================

def simulate_isolated():
    # =========================================================================
    # Simulation initialization
    # =========================================================================
    param = Param()
    param.sim.real_trajectory_csv = None        # nothing from a real run
    param.display.real_trajectory_plot = False
    param.display.real_sim_comparison_plot = False
    param.display.real_sim_comparison_animate = False
    param.display.trajectory_plot = True
    out_dir = LOGS_DIR / f"{FLIGHT_CODE}_{datetime.now():%Y%m%d_%H%M%S}"
    param.display.save_location = str(out_dir / "simulation")
    Path(param.display.save_location).mkdir(parents=True, exist_ok=True)
    print(f"Simulation: Main Controller, Tend {param.sim.Tend} s, dTi {param.sim.dTi} s, "
          f"trajectory of Structs.TrajectoryConfig (speed {param.traj.default_speed})")

    # =========================================================================
    # Agent initialization: at rest at the start of the trajectory
    # =========================================================================
    position = np.asarray(param.traj.ref_target_traj["pos"](0.0), float).reshape(3, 1)
    init_state = (position, np.zeros((3, 1)), np.eye(3).reshape(-1, 1), np.zeros((3, 1)))

    # =========================================================================
    # Main simulation loop (time_alignment.run_simulation: tracking of param.traj)
    # =========================================================================
    agent_true_hist = run_simulation(param, init_state, traj_offset=0.0)

    # =========================================================================
    # Results
    # =========================================================================
    t = (np.arange(param.sim.Nsim) + 1) * param.sim.dTi    # column k: after step k
    rows = []
    for k in range(param.sim.Nsim):
        h = agent_true_hist[:, k]
        sp = np.asarray(param.traj.ref_target_traj["pos"](t[k]), float).ravel()
        rows.append([t[k], "Main Controller", "1", *h[0:3], *h[3:6], *attitude_rpy(h[6:15]), *sp])
    out = out_dir / "sim.csv"
    pd.DataFrame(rows, columns=CSV_COLS).to_csv(out, index=False)
    print(f"Simulated flight saved to {out}")

    data = {"agent_true_hist": agent_true_hist, "removed_initial_pos": False}
    return data, param


# =============================================================================
# Replay of a real flight (--IRL_log): all flight stages
# =============================================================================

def simulate_flight(args):
    # =========================================================================
    # Flight log
    # =========================================================================
    c = flight_constants()
    print("8traj.py: " + ", ".join(f"{k}={v:g}" for k, v in c.items()))
    log_path = Path(args.IRL_log).resolve() if args.IRL_log else newest_log()
    print(f"Log: {log_path}")
    did, drone, phases, t_end, mode, p0_ret = flight_from_log(log_path, c, args.ret, args.ease)
    home = drone["p"][0]
    out_dir = log_path.parent
    sim_csv = log_path.with_name(log_path.stem + "_sim.csv")

    # =========================================================================
    # Simulation initialization
    # =========================================================================
    param = configure_param(Param(), t_end, DT, log_path)
    param.display.save_location = str(out_dir / "simulation")
    Path(param.display.save_location).mkdir(parents=True, exist_ok=True)
    events = setpoint_events(c, phases, t_end, mode, p0_ret)
    t_takeoff = phases["Take off"]

    # =========================================================================
    # Agent initialization: real yaw at the take off start, at rest on the floor
    # =========================================================================
    _, _, attitude, _, _ = get_initial_real_offset(param, phase="Take off")
    yaw0 = Rotation.from_matrix(np.asarray(attitude, float).reshape(3, 3)).as_euler("ZYX")[0]
    agent = DroneState(param)
    agent.position = np.asarray(home, float).reshape(3, 1).copy()
    agent.attitude = Rotation.from_euler("z", yaw0).as_matrix().reshape(-1, 1)   # level on the floor
    agent.desired.position = agent.position.copy()
    agent.desired.psi = agent.desired.old_psi = 0.0                             # yaw setpoint 0
    floor = float(home[2])

    # =========================================================================
    # Logs initialization
    # =========================================================================
    agent_true_hist = np.zeros((18, param.sim.Nsim))
    rows, sp, ev, ret_p0 = [], None, 0, None

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    for k in range(param.sim.Nsim):
        current_time = param.sim.dTi * k

        # 8traj.py: setpoints due at this time (sample and hold)
        while ev < len(events) and events[ev][0] <= current_time + 1e-9:
            v = events[ev][2]
            if isinstance(v, tuple):                    # return_origin: get_optitrack_position() at its start
                if ret_p0 is None:
                    ret_p0 = agent.position.ravel().copy()
                v = return_setpoint(c, mode, ret_p0, v[1], v[2])
            sp = v
            ev += 1
        phase = max((p for p in phases.items() if p[1] <= current_time + 1e-9 and p[0] != "EOF"),
                    key=lambda p: p[1], default=("", 0))[0]

        if sp is not None and current_time >= t_takeoff:
            agent.desired.position = sp.reshape(3, 1).copy()
            agent_motion(agent, param)
        elif current_time >= t_takeoff:
            passive_motion(agent, param)
        ground_contact(agent, floor)

        agent_true_hist[:, k:k + 1] = np.concatenate([agent.position, agent.velocity,
                                                     agent.attitude, agent.ang_vel])
        s = sp if sp is not None else np.full(3, np.nan)
        rows.append([current_time + DT, phase, did, *agent.position.ravel(), *agent.velocity.ravel(),
                     *attitude_rpy(agent.attitude), *s])

    # =========================================================================
    # Results
    # =========================================================================
    sim_df = pd.DataFrame(rows, columns=CSV_COLS)
    sim_df.to_csv(sim_csv, index=False)
    print(f"{len(events)} setpoints; simulated flight saved to {sim_csv}")
    compare(log_path, sim_df, real_setpoints(c, phases, t_end, mode, p0_ret), phases, drone,
            out_dir / "comparison")

    # The figures of the thesis simulation (real_sim_comparison_*) start at the
    # Main Controller: agent_true_hist[:, j] at t = (j+1) dTi after its first
    # complete estimator sample (as get_initial_real_offset / legacy_main.py)
    t0_main = get_initial_real_offset(param, phase="Main Controller")[4]
    k0 = min(int(round(t0_main / param.sim.dTi)), param.sim.Nsim - 1)
    param.sim.Nsim = param.sim.Nsim - k0
    param.sim.Tend = (param.sim.Nsim - 1) * param.sim.dTi
    data = {"agent_true_hist": agent_true_hist[:, k0:], "removed_initial_pos": False}
    return data, param


if __name__ == "__main__":
    main()
