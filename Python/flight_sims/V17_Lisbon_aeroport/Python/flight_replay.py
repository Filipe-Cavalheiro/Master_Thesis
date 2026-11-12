"""
Real-world simulation of the Crazyflie flights (OptiTrack): runs the flight
code of a swarm folder of Crazy_Flie/Optitrack (its Formation: target and
Voronoi formation; its SwarmMPC, for swarm_MPC_voronoi the V17 MPC of this
folder; its setpoint loop) on a simulated clock, with the drones simulated by
the Crazyflie plant of this simulation (plant.py, CrazyfliePlant: onboard
Mellinger with the firmware gains, experiment_arena.py CRAZYFLIE_ONBOARD).

Two runs (main.py --IRL_log / --arena, or Crazy_Flie/Optitrack/flight_sim.py):
  - replay of a flight log: from the state logged at the start of the
    "Formation" phase, for the same time, at the periods measured in the log,
    with the flight setup of <log>_meta.json when it exists (target mode and
    trajectory, formation and MPC settings, Mellinger position gains read
    back from the firmware). Writes <log>_sim.csv next to the log (same
    columns) and compares: errors printed, figures (PDF and PNG) in
    <run folder>/comparison/;
  - dry run (no log): the flight code with its current settings, the drones
    starting still on the slots it flies them to before the Formation phase.
    Saved to Results/<code>_dry_run_traj<k>_<n>drones_<date>/.

A flight code with a simulated / real target switch (swarm_MPC_voronoi:
main.py --target sim | real) is named <folder>_<sim|real>_target, as its
runs: the mode comes from <log>_meta.json, the run folder of the log,
--target, or that name (default sim). --traj sets TARGET_TRAJ (default: from
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
OPTI_PERIOD = 0.01      # [s] OptiTrack frames (velocity EMA of natnet_funcs)
PLANT_PERIOD = 0.01     # [s] simulation step of the plant (substeps inside CrazyfliePlant)


# ============================================================
# LOG
# ============================================================

def load_flight(csv_path):
    """
    Drone ids (sorted: drone i of the flight code), takeoff points, the
    Formation phase per drone (time from its start, position, setpoint,
    target) and the setpoint / formation update periods measured in it.
    """
    df = pd.read_csv(csv_path, dtype={"id": str, "flight_state": str})
    df["flight_state"] = df["flight_state"].fillna("").str.strip()
    ids = sorted(df["id"].dropna().unique(), key=lambda s: int(s) if s.isdigit() else s)

    takeoff = df[df["flight_state"] == "Take off"]
    homes = np.array([takeoff[takeoff["id"] == i][POS_COLS].iloc[0].to_numpy(float) for i in ids])

    phase = df[df["flight_state"] == PHASE]
    if phase.empty:
        raise ValueError(f"No '{PHASE}' phase in {csv_path}")
    t0 = phase["time_s"].min()
    drones = {}
    for i in ids:
        g = phase[phase["id"] == i].sort_values("time_s")
        drones[i] = {
            "t": g["time_s"].to_numpy(float) - t0,
            "p": g[POS_COLS].to_numpy(float),
            "sp": g[SP_COLS].to_numpy(float),
            "target": g[TARGET_COLS].to_numpy(float),
        }

    def period(t, x):
        # mean interval between changes of x (logged at 50 Hz), without the
        # gaps where x did not change (setpoint already on its goal)
        changed = np.r_[False, np.any(np.diff(x, axis=0) != 0, axis=1)]
        dt = np.diff(t[changed])
        return float(np.mean(dt[dt < 1.8 * np.median(dt)]))

    d0 = drones[ids[0]]
    periods = {"setpoint": period(d0["t"], d0["sp"]), "formation": period(d0["t"], d0["target"])}
    return ids, homes, drones, periods


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


def make_plant(nD, setpoint_mode, gains=None):
    from Structs_sim import Param as SimParam
    from Structs_agent import DroneState as SimDrone
    from experiment_arena import DRONE_MODEL, configure_onboard_gains
    from plant import CrazyfliePlant

    param = SimParam()
    param.sim.load_drone_model(DRONE_MODEL)
    param.sim.dTi = PLANT_PERIOD
    configure_onboard_gains(param)
    if gains:   # Mellinger position gains read back from the firmware (<log>_meta.json)
        param.controller.Mellinger["kp"] = np.diag([gains["ctrlMel.kp_xy"], gains["ctrlMel.kp_xy"], gains["ctrlMel.kp_z"]])
        param.controller.Mellinger["kv"] = np.diag([gains["ctrlMel.kd_xy"], gains["ctrlMel.kd_xy"], gains["ctrlMel.kd_z"]])
    plant = CrazyfliePlant(param, setpoint_mode=setpoint_mode)
    return plant, [SimDrone(param) for _ in range(nD)]


def init_formation(S, Formation, runCVT, homes, start_positions, nD):
    """
    The formation of the flight before the controllers start: converged,
    sites assigned from the takeoff points. Its initial sites are random in
    the flight code, so here they start at the mesh vertices nearest to the
    logged drones (the CVT then converges them as in the flight).
    """
    formation = Formation(S.Param(number_of_drones=nD, traj=S.TARGET_TRAJ))
    v = formation.voronoi
    d = np.linalg.norm(v.vertices[None, :, :] - start_positions[:, None, :], axis=2)
    v.sites = np.argmin(d, axis=1)
    formation.converge()
    formation.assign_slots(np.column_stack([homes[:, :2], np.full(nD, S.TAKEOFF_HEIGHT)]))
    return formation


def simulate(folder, ids, periods, T, start=None, start_vel=None, homes=None, gains=None):
    """
    The flight code of `folder` with the simulated drones `ids` for T
    seconds from the Formation phase. start / start_vel / homes: logged
    drone states and takeoff points (rows in the order of ids); None (dry
    run): the drones start still on the slots the flight flies them to
    before the Formation phase (converged formation, then MPC-settled
    slots). With a target drone (TARGET_DRONE_ID of the folder, one of ids)
    it flies the target path and the formation follows its simulated
    OptiTrack state.
    """
    S, Formation, SwarmMPC, runCVT, pos_based_mesh_transform = import_flight_code(folder)
    use_mpc = SwarmMPC is not None
    setpoint_mode = getattr(S, "SETPOINT_MODE", "position") if use_mpc else "position"
    has_target = getattr(S, "TARGET_URI", None) is not None
    target_id = str(S.TARGET_DRONE_ID) if has_target else None
    ch = [k for k, i in enumerate(ids) if i != target_id]          # chasers (drone k of the flight code)
    ti = ids.index(target_id) if has_target else None
    nD, n_all = len(ch), len(ids)
    mpc = (SwarmMPC(nD, with_target=True) if has_target else SwarmMPC(nD)) if use_mpc else None
    if has_target:
        from formation import target_reference

    if start is None:
        formation = Formation(S.Param(number_of_drones=nD, traj=S.TARGET_TRAJ))
        formation.converge()
        goals = formation.desired_positions.copy()
        goals = formation.assign_slots(np.column_stack([goals[:, :2], np.full(nD, S.TAKEOFF_HEIGHT)]))
        start = np.zeros((n_all, 3))
        if has_target:
            start[ti] = target_reference(formation.param, 0.0)[0]
            start[ch] = mpc.settle(goals, goals, target=start[ti])
        else:
            start[ch] = mpc.settle(goals, goals) if use_mpc else goals
        start_vel = np.zeros((n_all, 3))
        if use_mpc:
            from mpc import min_distance
            print(f"Slots: min distance between slots {min_distance(start):.2f} m")
    else:
        formation = init_formation(S, Formation, runCVT, homes[ch], start[ch], nD)

    plant, drones = make_plant(n_all, setpoint_mode, gains)
    for k in range(n_all):
        drones[k].position = start[k].reshape(3, 1).copy()
        drones[k].velocity = start_vel[k].reshape(3, 1).copy()
        drones[k].desired.position = drones[k].position.copy()

    sp_period, f_period = periods["setpoint"], periods["formation"]
    n_steps = int(round(T / PLANT_PERIOD))

    # flight-code state
    meas_p = np.array(start, dtype=float)       # OptiTrack position / filtered velocity
    meas_v = np.array([drones[k].velocity.ravel() for k in range(n_all)])
    sp = np.array(start, dtype=float)           # setpoints (rate-limited controller: its state)
    sp_v = np.zeros((n_all, 3))
    sp_a = np.zeros((n_all, 3))
    t_next = {"formation": 0.0, "mpc": 0.0, "setpoint": 0.0, "opti": 0.0}
    k_vor = 0

    log = []
    for n in range(n_steps + 1):
        t = n * PLANT_PERIOD
        eps = 1e-9

        if t + eps >= t_next["opti"]:
            pos = np.array([drones[k].position.ravel() for k in range(n_all)])
            vel = (pos - meas_p) / OPTI_PERIOD
            alpha = getattr(S, "VEL_FILTER_ALPHA", 0.3)
            meas_v = alpha * vel + (1 - alpha) * meas_v
            meas_p = pos
            t_next["opti"] += OPTI_PERIOD

        # Formation threads: target (virtual, or the measured target drone),
        # then the Voronoi worker (same period)
        if t + eps >= t_next["formation"]:
            with formation.lock:
                if has_target:
                    formation.target, formation.target_velocity = meas_p[ti].copy(), meas_v[ti].copy()
                else:
                    formation.target = np.array(formation.param.ref_target_traj["pos"](t), dtype=float)
                    if hasattr(formation, "target_velocity"):
                        formation.target_velocity = np.array(formation.param.ref_target_traj["vel"](t), dtype=float)
            v = formation.voronoi
            v.vertices, v.faces, v.mesh_pos = pos_based_mesh_transform(formation.get_target(), v, formation.param)
            if k_vor == 1 or k_vor % 5 == 0:
                v.sites = runCVT(v, v.sites)
            with formation.lock:
                goals = v.vertices[v.sites, :]
                formation.desired_positions = (S.clip_to_arena(goals) if hasattr(S, "clip_to_arena")
                                               else np.clip(goals, S.GOAL_MIN, S.GOAL_MAX))
            k_vor += 1
            t_next["formation"] += f_period

        if use_mpc and t + eps >= t_next["mpc"]:
            goals, goal_vel = formation.get_goals()
            if has_target:
                mpc.plan(meas_p[ch], meas_v[ch], goals, goal_vel, t, (meas_p[ti], meas_v[ti]))
            else:
                mpc.plan(meas_p[ch], meas_v[ch], goals, goal_vel, t)
            t_next["mpc"] += S.MPC_PERIOD

        # controllers
        if t + eps >= t_next["setpoint"]:
            for j, k in enumerate(ch):
                if use_mpc:
                    p, v_, a_, _ = mpc.sample(j, t)
                    sp[k], sp_v[k], sp_a[k] = p, v_, a_
                else:
                    step = formation.goal(j) - sp[k]
                    dist = np.linalg.norm(step)
                    max_step = S.MAX_SETPOINT_SPEED * S.SETPOINT_PERIOD
                    if dist > max_step:
                        step *= max_step / dist
                    sp[k] = sp[k] + step
            if has_target:
                sp[ti], sp_v[ti] = target_reference(formation.param, t)
            for k in range(n_all):
                d = drones[k].desired
                d.position = sp[k].reshape(3, 1).copy()
                d.velocity = sp_v[k].reshape(3, 1).copy()
                d.acceleration = sp_a[k].reshape(3, 1).copy()
            t_next["setpoint"] += sp_period

        target = formation.get_target()
        for k, i in enumerate(ids):
            log.append([t, PHASE, i, *drones[k].position.ravel(), *sp[k], *target])

        for k in range(n_all):
            plant.actuate(drones[k])

    cols = ["time_s", "flight_state", "id", *POS_COLS, *SP_COLS, *TARGET_COLS]
    return pd.DataFrame(log, columns=cols)


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


def print_comparison(ids, real, sim):
    print("\nSimulation vs flight (Formation phase), per drone:")
    print("  id   position error rms x/y/z [m]    max [m]   radius to target real/sim mean [m]")
    r_real, d_real = metrics(real, ids)
    r_sim, d_sim = metrics(sim, ids)
    for i in ids:
        e = sim[i]["p"] - real[i]["p"]
        rms = np.sqrt(np.mean(e ** 2, axis=0))
        print(f"  {i:>3}  {rms[0]:.3f} {rms[1]:.3f} {rms[2]:.3f}"
              f"             {np.max(np.linalg.norm(e, axis=1)):.3f}"
              f"     {r_real[i].mean():.3f} / {r_sim[i].mean():.3f}")
    for pair in d_real:
        print(f"  pair {pair}: inter-drone distance real min/mean {d_real[pair].min():.3f}/{d_real[pair].mean():.3f}"
              f", sim {d_sim[pair].min():.3f}/{d_sim[pair].mean():.3f} m")


def plot_comparison(ids, real, sim, title, save_dir, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    figs = {}

    # positions
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for k, i in enumerate(ids):
        t = real[i]["t"]
        for a, ax in enumerate(axes):
            ax.plot(t, real[i]["p"][:, a], color=colors(k), lw=1.6, label=f"CF{int(i):02d} real")
            ax.plot(t, sim[i]["p"][:, a], color=colors(k), lw=1.2, ls="--", label=f"CF{int(i):02d} sim")
    for a, ax in enumerate(axes):
        ax.plot(real[ids[0]]["t"], real[ids[0]]["target"][:, a], color="k", lw=0.8, alpha=0.5, label="target")
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title}: position, real (solid) vs sim (dashed)")
    figs["position"] = fig

    # top view
    fig, ax = plt.subplots(figsize=(7, 7))
    tgt = real[ids[0]]["target"]
    ax.plot(tgt[:, 0], tgt[:, 1], "k", lw=0.8, alpha=0.5, label="target")
    for k, i in enumerate(ids):
        ax.plot(real[i]["p"][:, 0], real[i]["p"][:, 1], color=colors(k), lw=1.6, label=f"CF{int(i):02d} real")
        ax.plot(sim[i]["p"][:, 0], sim[i]["p"][:, 1], color=colors(k), lw=1.2, ls="--", label=f"CF{int(i):02d} sim")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{title}: top view")
    figs["top_view"] = fig

    # formation metrics
    r_real, d_real = metrics(real, ids)
    r_sim, d_sim = metrics(sim, ids)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for k, i in enumerate(ids):
        t = real[i]["t"]
        axes[0].plot(t, r_real[i], color=colors(k), lw=1.6, label=f"CF{int(i):02d} real")
        axes[0].plot(t, r_sim[i], color=colors(k), lw=1.2, ls="--", label=f"CF{int(i):02d} sim")
        axes[1].plot(t, np.linalg.norm(real[i]["p"] - real[i]["sp"], axis=1), color=colors(k), lw=1.6)
        axes[1].plot(t, np.linalg.norm(sim[i]["p"] - sim[i]["sp"], axis=1), color=colors(k), lw=1.2, ls="--")
    for k, pair in enumerate(d_real):
        t = real[pair[0]]["t"][:len(d_real[pair])]
        axes[2].plot(t, d_real[pair], color=colors(k + len(ids)), lw=1.6, label=f"{pair} real")
        axes[2].plot(t, d_sim[pair][:len(t)], color=colors(k + len(ids)), lw=1.2, ls="--", label=f"{pair} sim")
    axes[0].set_ylabel("distance to\ntarget [m]")
    axes[1].set_ylabel("setpoint\ntracking error [m]")
    axes[2].set_ylabel("inter-drone\ndistance [m]")
    axes[0].legend(ncol=2, fontsize=8)
    axes[2].legend(fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title}: formation, real (solid) vs sim (dashed)")
    figs["formation"] = fig

    save_dir.mkdir(parents=True, exist_ok=True)
    for name, f in figs.items():
        for ext in ("pdf", "png"):
            f.savefig(save_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    print(f"Figures saved to {save_dir}")
    if show:
        plt.show()


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


def resolve_flight_code(name, target=None, log_path=None, meta=None):
    """
    (folder, run code) of a flight code name: a folder of Crazy_Flie/Optitrack,
    or <folder>_<sim|real>_target for a folder with the target switch (CF_TARGET
    in its Structs.py). The target mode (from <log>_meta.json, the run folder
    of log_path, `target`, else the name, default sim) is set in CF_TARGET
    before the flight code is imported.
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
    mode = from_log or target or mode or "sim"
    os.environ["CF_TARGET"] = mode
    return folder.resolve(), f"{folder.name}_{mode}_target"


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


def firmware_gains(meta):
    """Mellinger position gains read back from the firmware (first chaser that has them), None if not logged."""
    keys = ("ctrlMel.kp_xy", "ctrlMel.kp_z", "ctrlMel.kd_xy", "ctrlMel.kd_z")
    fws = [d.get("firmware", {}) for d in (meta or {}).get("drones", {}).values()]
    fws = [f for f in fws if all(f.get(k) is not None for k in keys)]
    if not fws:
        return None
    g = {k: fws[0][k] for k in keys}
    if any(any(abs(f[k] - g[k]) > 1e-9 for k in keys) for f in fws[1:]):
        print("Note: the drones flew with different Mellinger gains; the plant uses those of the first one")
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


def replay(folder, run_code, log_path, meta, args):
    S, *_ = import_flight_code(folder)
    gains = None
    if meta:
        apply_meta(S, meta)
        gains = firmware_gains(meta)
        print("Flight setup: <log>_meta.json" + (", plant gains " + ", ".join(
            f"{k[8:]} {v:g}" for k, v in gains.items()) if gains else ""))
    else:
        print("No <log>_meta.json (flight from before 13 Nov): the settings of the flight code now")
    print(f"Flight code: {run_code}\nLog: {log_path}")

    ids, homes, real, periods = load_flight(log_path)
    print(f"Drones {ids}, {PHASE} phase {min(r['t'][-1] for r in real.values()):.1f} s, "
          f"measured periods: setpoint {periods['setpoint'] * 1e3:.0f} ms, formation {periods['formation'] * 1e3:.0f} ms")

    start = np.array([real[i]["p"][0] for i in ids])
    start_vel = np.array([(real[i]["p"][5] - real[i]["p"][0]) / (real[i]["t"][5] - real[i]["t"][0]) for i in ids])
    T = min(r["t"][-1] for r in real.values())
    sim_df = simulate(folder, ids, periods, T, start, start_vel, homes, gains)
    out = log_path.with_name(log_path.stem + "_sim.csv")
    sim_df.to_csv(out, index=False)
    print(f"Simulated flight saved to {out}")

    sim = sim_tracks(sim_df, ids, {i: real[i]["t"] for i in ids})
    for i in ids:
        sim[i]["t"] = real[i]["t"]
    print_comparison(ids, real, sim)
    if not args.no_plot:
        plot_comparison(ids, real, sim, run_code, log_path.parent / "comparison",
                        show=not args.no_show)


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
    parser.add_argument("--target", choices=("sim", "real"), help="target mode of a flight code with the switch")
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    parser.add_argument("--no-plot", action="store_true", help="no figures")
    parser.add_argument("--drones", type=int, default=3, help="dry run: number of chasers (plus the target drone, if any)")
    parser.add_argument("--traj", type=int, help="trajectory: of the flight code with --IRL_log / --arena "
                        "(TARGET_TRAJ, default: the one of the log, else Structs.py), else of the simulation "
                        "(1-4: straight wall map, 5-6: Lisbon)")


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
