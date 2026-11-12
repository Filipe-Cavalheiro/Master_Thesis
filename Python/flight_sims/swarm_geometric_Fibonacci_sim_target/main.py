"""
Swarm geometric Fibonacci encirclement of a moving target: the thesis
simulation, and the replay of its real flight
(Crazy_Flie/Optitrack/swarm_geometric_Fibonacci_sim_target) for sim-vs-real.

    python swarm_geometric_Fibonacci_sim_target/main.py                 # the simulation, isolated (Structs.py)
    python swarm_geometric_Fibonacci_sim_target/main.py --IRL_log       # replay of the newest real run in Results/
    python swarm_geometric_Fibonacci_sim_target/main.py --IRL_log ../../Results/swarm_geometric_Fibonacci_sim_target_trajectory_<date>/trajectory_<date>.csv
    python swarm_geometric_Fibonacci_sim_target/main.py --IRL_log --no-show

Without --IRL_log: the simulation alone, nothing from the flight code or a
log. Every parameter from Structs.py (param.sim.nD, sphere_radius, Tend, dTi,
k_rad / k_geometric / k_apf, the target trajectory of TrajectoryConfig): the
chasers start at random positions around the target (up to 15 m), are
assigned to the Fibonacci points (Hungarian, geodesic cost), and every dTi run
the guidance (eq. 11, 14, 15) and the inner loop; the target is a simulated
drone tracking its trajectory. Writes Results/swarm_geometric_Fibonacci_sim_target_<date>/
(sim.csv and the figures in simulation/).

With --IRL_log [LOG]: the real flight replayed from the initial state of the
real drones (logged positions and yaw, phase times, setpoint periods), with the
setpoints of the flight code: smoothstep ramps (take off, return, land) and, in
the Main Controller, every dTi (0.1 s) the guidance of cf_funcs.controller (the
virtual target starts with it; the guidance is integrated with the flight's
dTi, guidance_param, the plant with DT). Runs from 12 Nov log the flight setup
(<log>_meta.json, <log>_setpoints.csv): then the constants of the flight, the
logged Main Controller setpoints, the measured controller period and the
firmware Mellinger position gains are used. Older runs: the old initial
guidance state (q_ref = (1, 0, 0) for every drone, radius 0). Writes
<log>_sim.csv next to the log, the figures in <run folder>/simulation/ and the
comparison real vs sim (compare_real.py) in <run folder>/comparison/.

Both save Results/simulation_results.pkl (python Animation_display.py redraws the
figures / animations from it) and draw the figures of
Animation_display.create_figures (GraphicalDisplay flags in Structs.py).

Modules:
    Structs.py            Param, DroneState, SimulationData
    drone_motion.py       inner loop: Mellinger controller + full nonlinear dynamics, RK4 on SO(3);
                          passive_motion (motors off), ground_contact
    helper_funcs.py       guidance: linear_motion_ref (eq. 11), apf_safe_guard (eq. 14),
                          update_radius (eq. 15); assign_agents_to_fibonacci_points; get_initial_real_offset
    fiboancci_funcs.py    get_fibonacci_points, get_min_safe_distance, geo_distances
    Animation_display.py  figures and animations
and, for the real flight:
    flight_code_funcs.py  the flight code: constants, target trajectory, log + logged flight setup,
                          setpoint schedule, ramps, initial guidance state
    compare_real.py       real vs sim metrics and figures
"""

import argparse
import pickle
import shutil
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Structs import Param, DroneState, SimulationData  # noqa: E402
from drone_motion import drone_motion, passive_motion, ground_contact  # noqa: E402
from helper_funcs import (linear_motion_ref, apf_safe_guard, update_radius,  # noqa: E402
                          assign_agents_to_fibonacci_points, get_initial_real_offset)
from fiboancci_funcs import get_fibonacci_points, get_min_safe_distance, geo_distances  # noqa: E402
from flight_code_funcs import *  # noqa: E402,F401,F403
from compare_real import compare  # noqa: E402

DT = 0.01                       # s, plant step of the replay (as param.sim.dTi of Structs.py)
PKL_FILE = HERE / "Results" / "simulation_results.pkl"   # the local Results folder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--IRL_log", nargs="?", const="", metavar="LOG",
                    help="replay a real flight from its initial values and compare with it "
                         f"(LOG: its flight log; default: the newest {FLIGHT_CODE} run in Results/). "
                         "Without it: the simulation alone, with the parameters of Structs.py")
    ap.add_argument("--ctrl-period", type=float,
                    help="with --IRL_log: period of the controller loop, s (default: measured in the flight if logged, else dTi)")
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
            pickle.dump({**data, "sim": sim_config(param)}, f)
        shutil.copy2(PKL_FILE, Path(param.display.save_location) / PKL_FILE.name)   # kept with the run

    if param.display.verbose:
        sim_data = SimulationData(**data)
        create_figures(sim_data, param)          # with --IRL_log also shows the comparison figures (q to quit)


def check_odd(nD, where):
    """The Fibonacci points (get_fibonacci_points, get_min_safe_distance) need an odd number of drones (or 1)."""
    if nD != 1 and nD % 2 == 0:
        raise SystemExit(f"{where}: nD = {nD}, but the Fibonacci points need an odd number of drones (e.g. {nD - 1} or {nD + 1})")


def save_sim_csv(rows, path):
    cols = ["time_s", "flight_state", "id", *POS_COLS, "est_vx", "est_vy", "est_vz",
            "roll_deg", "pitch_deg", "yaw_deg", "sp_x", "sp_y", "sp_z", "target_x", "target_y", "target_z"]
    sim_df = pd.DataFrame(rows, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(path, index=False)
    print(f"Simulated flight saved to {path}")
    return sim_df


# =============================================================================
# The simulation alone (no flags): the parameters of Structs.py
# =============================================================================

def simulate_isolated():
    # =========================================================================
    # Simulation initialization
    # =========================================================================
    param = Param()
    param.sim.real_trajectory_csv = None         # nothing from a real run
    check_odd(param.sim.nD, "Structs.py")
    out_dir = LOGS_DIR / f"{FLIGHT_CODE}_{datetime.now():%Y%m%d_%H%M%S}"
    param.display.save_location = str(out_dir / "simulation")
    Path(param.display.save_location).mkdir(parents=True, exist_ok=True)
    print(f"Simulation: {param.sim.nD} drones, sphere radius {param.sim.sphere_radius} m, "
          f"Tend {param.sim.Tend} s, dTi {param.sim.dTi} s, k_rad {param.k_rad}, "
          f"k_geometric {param.k_geometric}, k_apf {param.k_apf}")
    fib_ref_points = get_fibonacci_points(param.sim.nD, param.sim.sphere_radius, np.zeros([3, 1]))
    min_safe_dist = get_min_safe_distance(fib_ref_points, param) * 0.5

    # =========================================================================
    # Target initialization
    # =========================================================================
    target = DroneState(param)
    target.position = param.traj.initial_target_pos.copy()

    # =========================================================================
    # Chaser initialization: random positions around the target
    # =========================================================================
    chaser_list: list[DroneState] = []
    for iD in range(param.sim.nD):
        drone = DroneState(param)

        # Random spherical coordinates
        theta0 = 2 * np.pi * np.random.rand()
        phi0 = np.pi * np.random.rand()
        r0 = 15 * np.random.rand()
        initial_pos = r0 * np.array([
            [np.sin(phi0) * np.cos(theta0)],
            [np.sin(phi0) * np.sin(theta0)],
            [np.cos(phi0)],
        ])
        drone.position = initial_pos + np.asarray(target.position).reshape(3, 1)
        drone.radii_to_target = np.float32(np.linalg.norm(drone.position - target.position))

        # Set desired state
        drone.desired.position = fib_ref_points[iD, :] + target.position
        drone.desired.radii_to_target = np.linalg.norm(drone.desired.position - target.position)
        chaser_list.append(drone)

    # Guidance reference state on S^2, integrated by the outer loop (eq. (11)),
    # initialized at each drone's initial direction relative to the target
    q_ref_list = []
    for iD in range(param.sim.nD):
        q_s = chaser_list[iD].position - target.position
        q_ref_list.append(q_s / np.linalg.norm(q_s))

    # Assign every agent to the closest Fibonacci point (Hungarian, geodesic cost)
    if param.sim.nD != 1:
        _, assigned_ref = assign_agents_to_fibonacci_points(chaser_list, target, fib_ref_points, param)
    else:
        assigned_ref = range(param.sim.nD)

    # =========================================================================
    # Logs initialization
    # =========================================================================
    Nsim = param.sim.Nsim
    position_history = np.zeros([param.sim.nD, 3, Nsim])
    p_d_hist = np.zeros((param.sim.nD, 3, Nsim))
    radial_hist = np.zeros((param.sim.nD, Nsim))
    agent_pairs = list(combinations(range(param.sim.nD), 2))
    inter_agent_dist_hist = np.zeros((len(agent_pairs), Nsim))
    target_true_hist = np.zeros([24, Nsim])
    rows = []

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    for k in range(Nsim):
        current_time = param.sim.dTi * k

        # ---------------------------------------------------------
        # Move target
        # ---------------------------------------------------------
        target.desired.position = param.traj.ref_target_traj["pos"](current_time + param.sim.dTi)
        target.desired.velocity = param.traj.ref_target_traj["vel"](current_time + param.sim.dTi)
        drone_motion(target, param)

        # ---------------------------------------------------------
        # Active drones
        # ---------------------------------------------------------
        # Pairwise geodesic distances between guidance references (shared by
        # every drone's APF safeguard within this time step)
        geo_matrix = geo_distances(np.hstack(q_ref_list), 1, np.zeros([3, 1]))

        for iD, agent in enumerate(chaser_list):
            q_s = agent.position - target.position
            agent.radii_to_target = np.linalg.norm(q_s)
            if agent.radii_to_target < 1e-8:
                continue

            # Desired reduced-attitude direction: Fibonacci point on the unit sphere
            q_d = fib_ref_points[assigned_ref[iD], :]
            q_d = q_d / np.linalg.norm(q_d)

            # Position reference generation: integrate the guidance state (eq. (11))
            q_ref_list[iD], e_q = linear_motion_ref(q_ref_list[iD], q_d, param)

            # Safe guard collisions: APF on the guidance references (eq. (14))
            if param.sim.nD != 1:
                q_ref_list[iD] = apf_safe_guard(iD, q_ref_list[iD], q_ref_list, min_safe_dist, param, geo_matrix)
            q_hat = q_ref_list[iD]

            # Update sphere radius reference (eq. (15))
            agent.desired.radii_to_target = update_radius(agent.desired.radii_to_target, e_q, param)

            # ---------- Enforce S² constraint (local) ----------
            agent.desired.position = agent.desired.radii_to_target * q_hat + target.position

            drone_motion(agent, param)

        # logs
        for iD, agent in enumerate(chaser_list):
            position_history[iD, :, k:k + 1] = agent.position
            p_d_hist[iD, :, k:k + 1] = agent.desired.position
            radial_hist[iD, k] = agent.radii_to_target
            roll, pitch, yaw = Rotation.from_matrix(agent.attitude.reshape(3, 3)).as_euler("xyz", degrees=True)
            rows.append([current_time + param.sim.dTi, "", str(iD + 1), *agent.position.ravel(),
                         *agent.velocity.ravel(), roll, pitch, yaw, *agent.desired.position.ravel(),
                         *target.position.ravel()])
        for p_idx, (a, b) in enumerate(agent_pairs):
            inter_agent_dist_hist[p_idx, k] = np.linalg.norm(chaser_list[a].position - chaser_list[b].position)
        target_true_hist[:, k:k + 1] = np.concatenate([target.position, target.velocity, target.attitude,
                                                      target.ang_vel, target.ei, target.eI])

    save_sim_csv(rows, out_dir / "sim.csv")
    data = {
        "position_history": position_history,
        "target_true_hist": target_true_hist,
        "radial_history": radial_hist,
        "inter_agent_dist_hist": inter_agent_dist_hist,
        "agent_pairs": agent_pairs,
        "removed_initial_pos": False,
        "p_d_hist": p_d_hist,
        "min_safe_dist": min_safe_dist,
    }
    return data, param


# =============================================================================
# Replay of a real flight (--IRL_log)
# =============================================================================

def simulate_flight(args):
    # =========================================================================
    # Flight log
    # =========================================================================
    c = flight_constants()
    log_path = Path(args.IRL_log).resolve() if args.IRL_log else newest_log()
    print(f"Log: {log_path}")
    ids, drones, phases, t_end = load_log(log_path)
    check_odd(len(ids), "the log")
    meta, sp_log = load_extras(log_path)
    legacy = meta is None
    if meta:
        apply_meta(c, meta)                  # the constants the drones flew with
        print("Flight setup: <log>_meta.json" + (", setpoints: <log>_setpoints.csv" if sp_log is not None else ""))
    else:
        print("No <log>_meta.json (flight from before 12 Nov): constants of the flight code now, "
              "legacy guidance initial state")
    ctrl_period = args.ctrl_period or measured_ctrl_period(meta) or c.dTi
    out_dir = log_path.parent
    sim_csv = log_path.with_name(log_path.stem + "_sim.csv")
    print("Constants: " + ", ".join(f"{k}={v:g}" for k, v in vars(c).items()))
    print("Phases: " + ", ".join(f"{k} {v:.2f} s" for k, v in phases.items()))

    # =========================================================================
    # Simulation initialization
    # =========================================================================
    param = configure_param(Param(), c, len(ids), t_end, DT, log_path)
    param.display.save_location = str(out_dir / "simulation")
    Path(param.display.save_location).mkdir(parents=True, exist_ok=True)
    gparam = guidance_param(param, c)            # the guidance steps with the flight's dTi
    if meta:
        g = apply_firmware_gains(param, meta, ids)
        if g:
            print("Plant position gains from the firmware: " + ", ".join(f"{k[8:]} {v:g}" for k, v in g.items()))

    sched = schedule(c, phases, ctrl_period)
    for name, t0, n, period in sched:
        print(f"  {name:24s}: {n} setpoints from {t0:.2f} s, period {1e3 * period:.1f} ms")
    t_main = next((t0 for name, t0, n, _ in sched if name == MAIN and n), None)
    t_takeoff, t_stop = phases["Take off"], phases["EOF"]

    fib_ref_points = get_fibonacci_points(param.sim.nD, param.sim.sphere_radius, np.zeros([3, 1]))
    min_safe_dist = get_min_safe_distance(fib_ref_points, param) * 0.5

    # =========================================================================
    # Target initialization (virtual target: update_virtual_target)
    # =========================================================================
    target = DroneState(param)
    target.position = param.traj.initial_target_pos.copy()

    # =========================================================================
    # Chaser initialization: real state at the take off start, at rest on the floor
    # =========================================================================
    _, _, attitude_list, _, _ = get_initial_real_offset(param, phase="Take off")
    if len(ids) == 1:
        attitude_list = [attitude_list]
    homes = homes_of(drones)
    chaser_list: list[DroneState] = []
    for i, att in zip(ids, attitude_list):
        drone = DroneState(param)
        drone.position = homes[i].reshape(3, 1).copy()
        yaw = Rotation.from_matrix(np.asarray(att, float).reshape(3, 3)).as_euler("ZYX")[0]
        drone.attitude = Rotation.from_euler("z", yaw).as_matrix().reshape(-1, 1)   # level on the floor
        drone.desired.position = drone.position.copy()
        drone.desired.psi = drone.desired.old_psi = 0.0     # send_position_setpoint(..., yaw=0)
        chaser_list.append(drone)
    chaser_of = dict(zip(ids, chaser_list))
    floors = {i: float(homes[i][2]) for i in ids}

    # Guidance reference state (q_ref on S^2, radius), in the flight code's drone order
    order, q0, r0 = guidance_init(c, ids, homes, meta, legacy)
    q_ref_list = [q0[i] for i in order]
    for i in order:
        chaser_of[i].desired.radii_to_target = r0[i]
    logged_stream = logged_main_stream(sp_log, ids)
    print("Main Controller setpoints: " + ("logged" if logged_stream else
          "guidance, " + ("logged initial state" if meta else "legacy initial state" if legacy else "current initial state")))

    # =========================================================================
    # Logs initialization
    # =========================================================================
    Nsim = param.sim.Nsim
    position_history = np.zeros([param.sim.nD, 3, Nsim])
    p_d_hist = np.zeros((param.sim.nD, 3, Nsim))
    radial_hist = np.zeros((param.sim.nD, Nsim))
    agent_pairs = list(combinations(range(param.sim.nD), 2))
    inter_agent_dist_hist = np.zeros((len(agent_pairs), Nsim))
    target_true_hist = np.zeros([24, Nsim])
    events = setpoint_events(sched, logged_stream)
    ramp_starts = {}                             # measured positions at each ramp start
    has_setpoint = {i: False for i in ids}
    stream = logged_stream or {i: ([], []) for i in ids}   # Main Controller setpoints sent
    rows = []

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    ev = 0
    for k in range(Nsim):
        current_time = param.sim.dTi * k

        # ---------------------------------------------------------
        # Move target (virtual: starts with the Main Controller)
        # ---------------------------------------------------------
        if t_main is not None and current_time >= t_main:
            target.position = param.traj.ref_target_traj["pos"](current_time - t_main)
            target.velocity = param.traj.ref_target_traj["vel"](current_time - t_main)

        # ---------------------------------------------------------
        # Flight code: setpoints due at this time
        # ---------------------------------------------------------
        while ev < len(events) and events[ev][0] <= current_time + 1e-9:
            t_ev, name, j, n, who = events[ev]
            ev += 1
            if name == MAIN and who is not None:          # logged setpoint
                chaser_of[who].desired.position = logged_stream[who][1][j].reshape(3, 1).copy()
                has_setpoint[who] = True
            elif name == MAIN:                            # one tick of cf_funcs.controller
                # the target thread updates every dTi; the controller reads the latest value
                t_target = c.dTi * np.floor(j * ctrl_period / c.dTi + 1e-9)
                target_pos = param.traj.ref_target_traj["pos"](t_target)
                for iD, i in enumerate(order):
                    agent = chaser_of[i]
                    local = list(q_ref_list)              # chaser_list_ref.copy()

                    # Desired reduced-attitude direction: Fibonacci point iD on the unit sphere
                    q_d = fib_ref_points[iD, :].reshape(3, 1)
                    q_d = q_d / np.linalg.norm(q_d)

                    # Position reference generation: integrate the guidance state (eq. (11))
                    local[iD], e_q = linear_motion_ref(local[iD], q_d, gparam)

                    # Safe guard collisions: APF on the guidance references (eq. (14))
                    local[iD] = apf_safe_guard(iD, local[iD], local, min_safe_dist, gparam)
                    q_ref_list[iD] = local[iD]

                    # Update sphere radius reference (eq. (15))
                    agent.desired.radii_to_target = update_radius(agent.desired.radii_to_target, e_q, gparam)

                    # ---------- Enforce S² constraint (local) ----------
                    agent.desired.position = agent.desired.radii_to_target * q_ref_list[iD] + target_pos
                    has_setpoint[i] = True
                    stream[i][0].append(t_ev)
                    stream[i][1].append(agent.desired.position.ravel().copy())
            else:                                         # ramp_to
                if j == 0:                                # get_position() at the ramp start
                    ramp_starts[name] = {i: chaser_of[i].position.ravel().copy() for i in ids}
                for i in ids:
                    sp = ramp_setpoint(ramp_starts[name][i], ramp_goal(c, name, homes[i]), j, n)
                    chaser_of[i].desired.position = sp.reshape(3, 1)
                    has_setpoint[i] = True

        # ---------------------------------------------------------
        # Drones
        # ---------------------------------------------------------
        phase = max((p for p in phases.items() if p[1] <= current_time + 1e-9 and p[0] != "EOF"),
                    key=lambda p: p[1], default=("", 0))[0] if current_time < t_stop else "EOF"
        for iD, i in enumerate(ids):
            agent = chaser_of[i]
            if current_time >= t_stop:
                passive_motion(agent, param)              # stop_flight: motors off
            elif current_time >= t_takeoff and has_setpoint[i]:
                drone_motion(agent, param)
            ground_contact(agent, floors[i])              # before the first setpoint: armed, on the floor

            agent.radii_to_target = np.linalg.norm(agent.position - target.position)
            position_history[iD, :, k:k + 1] = agent.position
            p_d_hist[iD, :, k:k + 1] = agent.desired.position
            radial_hist[iD, k] = agent.radii_to_target

            roll, pitch, yaw = Rotation.from_matrix(agent.attitude.reshape(3, 3)).as_euler("xyz", degrees=True)
            sp = agent.desired.position.ravel() if has_setpoint[i] and current_time < t_stop else np.full(3, np.nan)
            tg = target.position.ravel() if t_main is not None and current_time >= t_main else np.full(3, np.nan)
            rows.append([current_time + DT, phase, i, *agent.position.ravel(), *agent.velocity.ravel(),
                         roll, pitch, yaw, *sp, *tg])

        for p_idx, (a, b) in enumerate(agent_pairs):
            inter_agent_dist_hist[p_idx, k] = np.linalg.norm(chaser_list[a].position - chaser_list[b].position)
        target_true_hist[:, k:k + 1] = np.concatenate([target.position, target.velocity, target.attitude,
                                                      target.ang_vel, target.ei, target.eI])

    # =========================================================================
    # Results
    # =========================================================================
    sim_df = save_sim_csv(rows, sim_csv)
    stream = {i: (np.asarray(ts, float), np.asarray(vs, float).reshape(-1, 3)) for i, (ts, vs) in stream.items()}
    compare(c, log_path, sim_df, sched, stream, ctrl_period, out_dir / "comparison", meta, sp_log)

    data = {
        "position_history": position_history,
        "target_true_hist": target_true_hist,
        "radial_history": radial_hist,
        "inter_agent_dist_hist": inter_agent_dist_hist,
        "agent_pairs": agent_pairs,
        "removed_initial_pos": False,
        "p_d_hist": p_d_hist,
        "min_safe_dist": min_safe_dist,
        "phases": phases,
    }
    return data, param

def sim_config(param):
    """param.sim values the figures need, stored in the .pkl (Animation_display.py restores them)."""
    return {"nD": param.sim.nD, "Tend": param.sim.Tend, "dTi": param.sim.dTi, "Nsim": param.sim.Nsim,
            "sphere_radius": param.sim.sphere_radius}

if __name__ == "__main__":
    main()
