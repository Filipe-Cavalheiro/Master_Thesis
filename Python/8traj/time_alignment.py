"""
Automatic time alignment between the simulation and a real flight log.

The simulation is initialised from the real drone state at the start of a flight
phase and then tracks a reference trajectory evaluated at

        t_ref(k) = dTi * k + traj_offset

`traj_offset` is the only unknown (it absorbs the logger / trajectory-start
delay). We search for the value that minimises the position error norm between
the simulated and the real trajectory over the simulated window.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from Structs import Param, DroneState
from agent_motion import agent_motion

AXES = ("x", "y", "z")


# =============================================================================
# Simulation
# =============================================================================
def run_simulation(param: Param, init_state, traj_offset, verbose=False):
    """
    Run the closed-loop simulation once.

    Parameters
    ----------
    init_state : (position, velocity, attitude, ang_vel)
        Initial state, as returned by the first four outputs of
        get_initial_real_offset().
    traj_offset : float
        Reference-trajectory time at simulation start (s).

    Returns
    -------
    hist : (18, Nsim) ndarray
        [pos(3); vel(3); attitude(9); ang_vel(3)] AFTER each step, i.e. column k
        corresponds to time dTi*(k+1) since the start.
    """
    dTi = param.sim.dTi

    # A fresh agent each run: agent_motion mutates desired.old_psi etc.
    agent = DroneState(param)
    (agent.position, agent.velocity,
     agent.attitude, agent.ang_vel) = (np.array(x, dtype=float, copy=True)
                                       for x in init_state)

    hist = np.zeros((18, param.sim.Nsim))
    for k in range(param.sim.Nsim):
        if verbose:
            print(f"Computing simulation iteration: {k:04d}")
        t_ref = dTi * k + traj_offset

        agent.desired.position = param.traj.ref_target_traj["pos"](t_ref + dTi)
        agent.desired.velocity = param.traj.ref_target_traj["vel"](t_ref + dTi)

        agent_motion(agent, param)

        hist[:, k:k + 1] = np.concatenate([
            agent.position,
            agent.velocity,
            agent.attitude,
            agent.ang_vel,
        ])
    return hist


# =============================================================================
# Real data
# =============================================================================
def load_real_reference(param: Param, time0, phase="Main Controller",
                        source="opti"):
    """
    Load the real position history to compare against.

    Parameters
    ----------
    time0 : float
        `time_s` at which the simulation was initialised
        (last output of get_initial_real_offset).
    source : "opti" | "est"
        "opti" = OptiTrack (ground truth), "est" = onboard Kalman estimate
        (the same signal used to initialise the sim).

    Returns
    -------
    (t_real, pos_real) with t_real relative to time0 (s) and pos_real (3, M),
    or None if there is no usable real data.
    """
    try:
        df = pd.read_csv(param.sim.real_trajectory_csv)
    except (FileNotFoundError, OSError, TypeError):
        return None

    cols = [f"{source}_{a}" for a in AXES]
    needed = ["time_s", "flight_state", *cols]
    if any(c not in df.columns for c in needed):
        return None

    for c in ["time_s", *cols]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["flight_state"].fillna("").astype(str).str.strip().eq(phase)]
    df = df.dropna(subset=["time_s", *cols])
    df = df[df["time_s"] >= time0]
    df = df.sort_values("time_s").drop_duplicates("time_s")

    if len(df) < 2:
        return None

    t_real = df["time_s"].to_numpy() - time0
    pos_real = df[cols].to_numpy(dtype=float).T
    return t_real, pos_real


# =============================================================================
# Error metric
# =============================================================================
def position_error(param: Param, hist, real):
    """
    Position error norm between sim and real at the simulation time stamps.

    Returns (t_sim, err_norm, sim_pos, real_pos) restricted to the time window
    covered by the real data.
    """
    t_real, pos_real = real
    t_sim = param.sim.dTi * (np.arange(param.sim.Nsim) + 1)

    valid = t_sim <= t_real[-1]
    t_sim = t_sim[valid]

    real_pos = np.vstack([np.interp(t_sim, t_real, pos_real[i]) for i in range(3)])
    sim_pos = hist[:3, valid]
    err_norm = np.linalg.norm(sim_pos - real_pos, axis=0)
    return t_sim, err_norm, sim_pos, real_pos


def _cost(param, init_state, real, traj_offset):
    """RMS of the position error norm (inf if the sim blew up)."""
    with np.errstate(all="ignore"):
        try:
            hist = run_simulation(param, init_state, traj_offset)
        except np.linalg.LinAlgError:
            return np.inf
    if not np.all(np.isfinite(hist)):
        return np.inf
    _, err_norm, _, _ = position_error(param, hist, real)
    if err_norm.size == 0:
        return np.inf
    return float(np.sqrt(np.mean(err_norm ** 2)))


# =============================================================================
# Search
# =============================================================================
def find_best_time_offset(param: Param, init_state, real,
                          search_range=(-1.5, 1.5), coarse_step=0.1,
                          refine_tol=1e-3, verbose=True):
    """
    Find the trajectory time offset that minimises the RMS position error.

    1. Coarse grid over `search_range`. The Lissajous reference is periodic, so
       the cost has several local minima and a plain local optimiser could land
       in the wrong one.
    2. Bounded 1-D refinement around the best grid point.

    Returns a dict with: offset, rms, offset_grid, cost_grid.
    """
    lo, hi = search_range
    grid = np.arange(lo, hi + coarse_step / 2, coarse_step)
    costs = np.array([_cost(param, init_state, real, o) for o in grid])

    if not np.isfinite(costs).any():
        raise RuntimeError("Simulation diverged for every candidate offset.")

    i = int(np.argmin(costs))
    res = minimize_scalar(
        lambda o: _cost(param, init_state, real, o),
        bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]),
        method="bounded",
        options={"xatol": refine_tol},
    )

    best, rms = (res.x, res.fun) if res.fun <= costs[i] else (grid[i], costs[i])

    if verbose:
        print(f"[align] best trajectory offset = {best:+.3f} s  "
              f"(RMS position error = {rms * 100:.2f} cm)")
        if i in (0, len(grid) - 1):
            print("[align] WARNING: optimum is at the edge of the search range; "
                  "widen `search_range`.")
        if param.sim.Tend < 1.0:
            print(f"[align] NOTE: simulated window is only {param.sim.Tend:.2f} s; "
                  "a longer Tend gives a better-determined offset.")

    return {"offset": float(best), "rms": float(rms),
            "offset_grid": grid, "cost_grid": costs}


# =============================================================================
# Plots
# =============================================================================
def plot_alignment(param: Param, init_state, real, result):
    """Cost vs offset, plus sim-vs-real position and err_norm at the best offset."""
    hist = run_simulation(param, init_state, result["offset"])
    t, err_norm, sim_pos, real_pos = position_error(param, hist, real)

    fig1, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result["offset_grid"], result["cost_grid"] * 100)
    ax.axvline(result["offset"], color="r", ls="--",
               label=f"best = {result['offset']:+.3f} s")
    ax.set_xlabel("Trajectory time offset (s)")
    ax.set_ylabel("RMS position error (cm)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig1.tight_layout()

    fig2, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, a in enumerate(AXES):
        axes[i].plot(t, real_pos[i], label="real")
        axes[i].plot(t, sim_pos[i], "--", label="sim")
        axes[i].set_ylabel(f"{a} (m)")
        axes[i].grid(True, alpha=0.4)
    axes[0].legend()
    axes[3].plot(t, err_norm * 100, "k")
    axes[3].set_ylabel("err_norm (cm)")
    axes[3].set_xlabel("Time since sim start (s)")
    axes[3].grid(True, alpha=0.4)
    fig2.tight_layout()
    return fig1, fig2