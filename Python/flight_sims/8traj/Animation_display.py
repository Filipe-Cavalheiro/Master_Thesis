import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from pathlib import Path
import pandas as pd
import pickle

from Structs import Param, SimulationData
from helper_funcs import EST_COLS
from plot_real_data import *

def create_figures(data: SimulationData, param: Param):
    if param.display.real_trajectory_plot is True:
        real_trajectory_plot(data, param)

    if param.display.real_sim_comparison_plot is True:
        real_sim_comparison_plot(data, param)

    if param.display.real_sim_comparison_animate is True:
        real_sim_comparison_animation(data, param)

    if param.display.trajectory_plot is True:
        trajectory_plot(data, param)

    # if param.display.trajectory_animate is True:
    #     animate_simulation(data, param)

    # Get all figures created by the plotting functions
    figs = [plt.figure(num) for num in plt.get_fignums()]

    quit_flag = {"pressed": False}

    def on_key(event):
        if event.key == "q":
            quit_flag["pressed"] = True

    # Connect key handler to all figures
    for fig in figs:
        fig.canvas.mpl_connect("key_press_event", on_key)

    # Show all figures without blocking
    plt.show(block=False)

    # Wait without freezing GUI. Only wait for "q" when there actually are
    # figures on an interactive backend — otherwise no window can ever
    # receive the key press and the loop would hang forever.
    import matplotlib
    if figs and matplotlib.get_backend().lower() != "agg":
        while not quit_flag["pressed"]:
            plt.pause(0.1)

    # Close everything
    plt.close("all")

    print("Exited on q")

def real_trajectory_plot(data: SimulationData, param: Param):
    """
    Plot real drone trajectories recorded in an OptiTrack CSV file.

    Parameters
    ----------
    data : SimulationData
        Kept for compatibility with the plotting framework.
    param : Param
        Must provide:
            param.display.real_trajectory_csv
            param.display.save_location

    Returns
    -------
    df : pandas.DataFrame
        Valid drone position samples.
    fig_3d : matplotlib.figure.Figure
        3D trajectory figure.
    fig_xyz : matplotlib.figure.Figure
        XYZ components over time.
    """
    # if a specific test is desired define its path here else the newest one is used
    csv_path = None

    # Load the real world data
    csv_path = Path(csv_path ) if csv_path else newest_log()
    print(f"Loading {csv_path}")

    df, figs = plot_flight_log(csv_path)

    duration = df["t"].iloc[-1]
    print(f"{len(df)} samples over {duration:.1f} s "
          f"(~{len(df) / duration:.0f} Hz)")
    print(f"Phases: {[s[0] for s in phase_spans(df)]}")
    n_missing = int(df["est_x"].isna().sum())
    if n_missing:
        print(f"Note: {n_missing} rows have no estimator data yet.")

#    out = Path(args.save_figs)
#    out.mkdir(parents=True, exist_ok=True)
#    for name, fig in figs.items():
#        fig.savefig(out / f"{csv_path.stem}_{name}.png", dpi=150)
#    print(f"Figures saved to {out}")

#    anim = None
#    if args.animate or args.save:
#        _, anim = animate_flight(
#            df,
#            trail_length=args.trail,
#            playback_speed=args.speed,
#            save_path=args.save,
#        )

    plt.show()

def _load_real_main_controller(param: Param, phase="Main Controller",
                               source="est"):
    """
    Load the real-flight positions of the `phase` section only.
 
    The time axis starts at 0 at the first complete estimator sample of the
    phase, i.e. the same sample get_initial_real_offset() uses to initialise
    the simulation, so real and simulated runs share the same t = 0.
 
    Parameters
    ----------
    source : "est" | "opti"
        Which real-world position to use: the Kalman estimate (the same
        signal the simulation is initialised from) or OptiTrack (ground truth).
 
    Returns
    -------
    t : (M,) ndarray  time since the start of the phase [s]
    pos : (3, M) ndarray  real position [m]
    """
    if source not in ("est", "opti"):
        raise ValueError("source must be 'est' or 'opti'")
 
    csv_path = Path(param.sim.real_trajectory_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Real trajectory CSV not found: {csv_path}")
 
    pos_cols = [f"{source}_{a}" for a in ("x", "y", "z")]
 
    df = pd.read_csv(csv_path)
    for col in ["time_s", *EST_COLS, *pos_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    states = df["flight_state"].fillna("").astype(str).str.strip()
    df_phase = df[states.eq(phase)]
    if df_phase.empty:
        available = sorted(s for s in states.unique() if s)
        raise ValueError(
            f"Flight state '{phase}' not found in CSV. Available: {available}"
        )
 
    # Same start sample as get_initial_real_offset(): first complete estimate
    first_complete = df_phase.dropna(subset=["time_s", *EST_COLS])
    if first_complete.empty:
        raise ValueError(f"No complete estimator samples in phase '{phase}'.")
    time0 = float(first_complete["time_s"].iloc[0])
 
    df_phase = (
        df_phase.dropna(subset=["time_s", *pos_cols])
        .loc[lambda d: d["time_s"] >= time0]
        .sort_values("time_s")
        .drop_duplicates("time_s")
    )
 
    t = df_phase["time_s"].to_numpy(dtype=float) - time0
    pos = df_phase[pos_cols].to_numpy(dtype=float).T
    return t, pos
 
 
def real_sim_comparison_plot(data: SimulationData, param: Param,
                        phase="Main Controller", source="est"):
    """
    Compare the simulated trajectory with the real flight, using only the
    `phase` ("Main Controller") section of the real log.
 
    Produces two figures:
        1. 3D overlay of the real and simulated trajectories.
        2. Difference (sim - real) over time: x, y, z components and the
           3D error norm.
 
    The real data is linearly interpolated onto the simulation time grid, and
    only the overlap between the two time spans is compared.
 
    Returns
    -------
    fig_traj, fig_diff : matplotlib.figure.Figure
    """
    # ------------------------------------------------------------------
    # Real data (Main Controller section only)
    # ------------------------------------------------------------------
    t_real, p_real = _load_real_main_controller(param, phase, source)
 
    # ------------------------------------------------------------------
    # Simulation data
    # agent_true_hist[:, k] is stored AFTER step k, i.e. at t = (k+1)*dTi
    # ------------------------------------------------------------------
    hist = np.asarray(data.agent_true_hist)
    p_sim = hist[0:3, :]
    t_sim = (np.arange(p_sim.shape[1]) + 1) * param.sim.dTi
 
    # Keep only the time span covered by both
    overlap = t_sim <= t_real[-1]
    if not overlap.any():
        raise ValueError(
            f"No overlap between simulation (t=[{t_sim[0]:.2f}, "
            f"{t_sim[-1]:.2f}] s) and the '{phase}' section of the real "
            f"flight (t=[0, {t_real[-1]:.2f}] s)."
        )
    t_sim = t_sim[overlap]
    p_sim = p_sim[:, overlap]
 
    # Real data on the simulation time grid
    p_real_i = np.vstack([np.interp(t_sim, t_real, p_real[i]) for i in range(3)])
 
    # Real trajectory limited to the simulated time span (for plotting)
    in_span = t_real <= t_sim[-1] + param.sim.dTi
    p_real_plot = p_real[:, in_span]
 
    # Difference sim - real
    diff = p_sim - p_real_i
    err_norm = np.linalg.norm(diff, axis=0)
    rms = np.sqrt(np.mean(err_norm ** 2))
 
    label_real = "Real (OptiTrack)" if source == "opti" else "Real (estimate)"
 
    # ------------------------------------------------------------------
    # Figure 1: trajectories overlapped
    # ------------------------------------------------------------------
    fig_traj = plt.figure(figsize=(10, 8))
    ax = fig_traj.add_subplot(111, projection="3d")
 
    ax.plot(p_real_plot[0], p_real_plot[1], p_real_plot[2],
            color="tab:blue", lw=2.5, label=label_real)
    ax.plot(p_sim[0], p_sim[1], p_sim[2],
            color="tab:red", lw=2.0, linestyle="--", label="Simulation")
 
    ax.scatter(*p_real_plot[:, 0], marker="o", s=70, color="green",
               label="Start", zorder=5)
    ax.scatter(*p_real_plot[:, -1], marker="X", s=70, color="tab:blue",
               label="End (real)", zorder=5)
    ax.scatter(*p_sim[:, -1], marker="X", s=70, color="tab:red",
               label="End (sim)", zorder=5)
 
    # Equal axes scaling
    all_pts = np.hstack([p_real_plot, p_sim])
    mins, maxs = all_pts.min(axis=1), all_pts.max(axis=1)
    center = (mins + maxs) / 2
    half = max((maxs - mins).max() / 2, 0.05) * 1.2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
 
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Real vs simulated trajectory ({phase})")
    ax.legend(loc="upper right")
    fig_traj.tight_layout()
 
    # ------------------------------------------------------------------
    # Figure 2: difference over time
    # ------------------------------------------------------------------
    fig_diff, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
 
    for i, name in enumerate(("X", "Y", "Z")):
        axes[i].plot(t_sim, diff[i], lw=1.8)
        axes[i].axhline(0.0, color="k", lw=0.8, alpha=0.5)
        axes[i].set_ylabel(f"Δ{name} [m]")
        axes[i].grid(True, alpha=0.4)
 
    axes[3].plot(t_sim, err_norm, color="black", lw=1.8)
    axes[3].set_ylabel("|Δ| [m]")
    axes[3].set_xlabel("Time since start of Main Controller [s]")
    axes[3].grid(True, alpha=0.4)
 
    fig_diff.suptitle(
        f"Simulation − real position difference   "
        f"(final = {err_norm[-1] * 100:.2f} cm, RMS = {rms * 100:.2f} cm)"
    )
    fig_diff.tight_layout()
 
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    current_folder = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_folder, param.display.save_location)
    os.makedirs(save_dir, exist_ok=True)
    fig_traj.savefig(os.path.join(save_dir, "real_sim_trajectory.pdf"),
                     bbox_inches="tight")
    fig_diff.savefig(os.path.join(save_dir, "real_sim_difference.pdf"),
                     bbox_inches="tight")
 
    print(f"Real vs sim: final error {err_norm[-1] * 100:.2f} cm, "
          f"RMS {rms * 100:.2f} cm over {t_sim[-1]:.2f} s")
 
    return fig_traj, fig_diff

# Keeps FuncAnimation objects alive (matplotlib stops animating if the
# animation object is garbage collected).
_ANIMATIONS = []
 
def _align_real_sim(data: SimulationData, param: Param,
                    phase="Main Controller", source="est"):
    """
    Load the real `phase` section and the simulation, and put both on the
    simulation time grid (t = 0 is the start of the phase for both).
 
    agent_true_hist[:, k] is stored AFTER step k, i.e. at t = (k+1)*dTi.
 
    Returns
    -------
    t_real : (M,)   raw real time stamps [s]
    p_real : (3, M) raw real positions [m]
    t_sim : (N,)    sim time stamps within the overlap [s]
    p_sim : (3, N)  sim positions [m]
    p_real_i : (3, N) real positions interpolated at t_sim [m]
    """
    t_real, p_real = _load_real_main_controller(param, phase, source)
 
    hist = np.asarray(data.agent_true_hist)
    p_sim = hist[0:3, :]
    t_sim = (np.arange(p_sim.shape[1]) + 1) * param.sim.dTi
 
    # Keep only the time span covered by both
    overlap = t_sim <= t_real[-1]
    if not overlap.any():
        raise ValueError(
            f"No overlap between simulation (t=[{t_sim[0]:.2f}, "
            f"{t_sim[-1]:.2f}] s) and the '{phase}' section of the real "
            f"flight (t=[0, {t_real[-1]:.2f}] s)."
        )
    t_sim = t_sim[overlap]
    p_sim = p_sim[:, overlap]
 
    p_real_i = np.vstack(
        [np.interp(t_sim, t_real, p_real[i]) for i in range(3)]
    )
    return t_real, p_real, t_sim, p_sim, p_real_i

def real_sim_comparison_animation(data: SimulationData, param: Param,
                                phase="Main Controller", source="est",
                                playback_speed=1.0, trail_length=None):
    """
    Animated version of real_sim_comparison.
 
    Left: 3D view with the real and simulated trajectories being drawn, a
    moving marker for each, and a line joining the two current positions.
    Right: the 3D error norm over time with a moving marker.
 
    Parameters
    ----------
    playback_speed : float
        1.0 = real time. The frame interval is clamped to >= 20 ms, which is
        the smallest value matplotlib handles reliably, so with dTi = 0.01 s
        the effective maximum speed is 0.5x.
    trail_length : int or None
        Number of real samples / sim steps of history to draw (None = all).
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    anim : matplotlib.animation.FuncAnimation
    """
    if playback_speed <= 0:
        raise ValueError("playback_speed must be greater than zero.")
 
    t_real, p_real, t_sim, p_sim, p_real_i = _align_real_sim(
        data, param, phase, source
    )
    n_frames = len(t_sim)
    diff = p_sim - p_real_i
    err_norm = np.linalg.norm(diff, axis=0)
 
    label_real = "Real (OptiTrack)" if source == "opti" else "Real (estimate)"
 
    # Real samples within the simulated time span (for axis limits)
    p_real_span = p_real[:, t_real <= t_sim[-1] + param.sim.dTi]
 
    # ------------------------------------------------------------------
    # Figure and axes
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    ax = fig.add_subplot(gs[0], projection="3d")
    ax_e = fig.add_subplot(gs[1])
 
    all_pts = np.hstack([p_real_span, p_sim])
    mins, maxs = all_pts.min(axis=1), all_pts.max(axis=1)
    center = (mins + maxs) / 2
    half = max((maxs - mins).max() / 2, 0.05) * 1.2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Real vs simulation ({phase})")
 
    real_line, = ax.plot([], [], [], color="tab:blue", lw=2.5,
                         label=label_real)
    sim_line, = ax.plot([], [], [], color="tab:red", lw=2.0, ls="--",
                        label="Simulation")
    real_dot, = ax.plot([], [], [], "o", color="tab:blue", markersize=8)
    sim_dot, = ax.plot([], [], [], "s", color="tab:red", markersize=7)
    gap_line, = ax.plot([], [], [], color="gray", lw=1.2, label="Difference")
    ax.legend(loc="upper right")
 
    ax_e.plot(t_sim, err_norm, color="lightgray", lw=1.5)   # full curve, faint
    err_line, = ax_e.plot([], [], color="black", lw=2.0)
    err_dot, = ax_e.plot([], [], "o", color="tab:red", markersize=7)
    ax_e.set_xlim(t_sim[0], t_sim[-1])
    ax_e.set_ylim(0, max(err_norm.max() * 1.1, 1e-3))
    ax_e.set_xlabel("Time since start of Main Controller [s]")
    ax_e.set_ylabel("|sim − real| [m]")
    ax_e.set_title("Position difference")
    ax_e.grid(True, alpha=0.4)
 
    info = fig.text(0.5, 0.02, "", ha="center")
 
    def set_3d(line, x, y, z):
        line.set_data(x, y)
        line.set_3d_properties(z)
 
    def update(i):
        t = t_sim[i]
 
        # Real raw samples up to the current time
        n = int(np.searchsorted(t_real, t, side="right"))
        r0 = 0 if trail_length is None else max(0, n - trail_length)
        s0 = 0 if trail_length is None else max(0, i + 1 - trail_length)
 
        set_3d(real_line, p_real[0, r0:n], p_real[1, r0:n], p_real[2, r0:n])
        set_3d(sim_line, p_sim[0, s0:i + 1], p_sim[1, s0:i + 1],
               p_sim[2, s0:i + 1])
 
        # Markers are drawn at the same instant (real interpolated at t)
        set_3d(real_dot, [p_real_i[0, i]], [p_real_i[1, i]], [p_real_i[2, i]])
        set_3d(sim_dot, [p_sim[0, i]], [p_sim[1, i]], [p_sim[2, i]])
        set_3d(gap_line,
               [p_real_i[0, i], p_sim[0, i]],
               [p_real_i[1, i], p_sim[1, i]],
               [p_real_i[2, i], p_sim[2, i]])
 
        err_line.set_data(t_sim[:i + 1], err_norm[:i + 1])
        err_dot.set_data([t], [err_norm[i]])
 
        info.set_text(
            f"t = {t:.2f} s | error = {err_norm[i] * 100:.2f} cm | "
            f"playback {playback_speed:g}x"
        )
        return (real_line, sim_line, real_dot, sim_dot, gap_line,
                err_line, err_dot, info)
 
    interval_ms = max(20, int(round(param.sim.dTi * 1000 / playback_speed)))
 
    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=interval_ms,
        blit=False,          # must be False for 3D plots
        repeat=False,
    )
    _ANIMATIONS.append(anim)
 
    if param.display.save_to_mp4:
        print("Saving real vs sim animation")
        current_folder = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_folder, param.display.save_location)
        os.makedirs(save_dir, exist_ok=True)
        anim.save(
            os.path.join(save_dir, "real_sim_comparison.mp4"),
            writer="ffmpeg",
            fps=max(1, round(1000 / interval_ms)),
        )
 
    return fig, anim

def trajectory_plot(data: SimulationData, param: Param):
    """
    Static 3D plot of the complete trajectories of the agent
    """

    # -----------------------------
    # Compute axis limits (full history, not just sampled frames)
    # -----------------------------
    all_x = data.agent_true_hist[0]
    all_y = data.agent_true_hist[1]
    all_z = data.agent_true_hist[2]

    margin = 2

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_zlim(all_z.min() - margin, all_z.max() + margin)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    # Bigger, bolder tick labels and axis labels
    ax.tick_params(axis='both', labelsize=11, width=1.5)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.zaxis.label.set_fontsize(16)

    # Target: full trajectory + start/end markers
    ax.plot(
        data.agent_true_hist[0],
        data.agent_true_hist[1],
        data.agent_true_hist[2],
        'k', lw=3, label='agent',
    )
    ax.plot(
        [data.agent_true_hist[0, 0]],
        [data.agent_true_hist[1, 0]],
        [data.agent_true_hist[2, 0]],
    )
    ax.plot(
        [data.agent_true_hist[0, -1]],
        [data.agent_true_hist[1, -1]],
        [data.agent_true_hist[2, -1]],
    )

    if param.sim.nD <= 6: 
        ax.legend(loc='upper right', fontsize=16)

    print("saving trajectory plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, param.display.save_location)
    plt.savefig(
        os.path.join(
            file_path,
            f"trajectory_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )
    print("Done.")

    plt.show()

def animate_simulation(data, param):
    """
    FuncAnimation of all agents and target trajectory
    """

    frame_indices = np.linspace(
        0,
        param.sim.Nsim - 1,
        int(param.sim.Tend * param.display.mp4_fps),
        dtype=int,
    )

    # -----------------------------
    # Compute axis limits
    # -----------------------------
    all_x = data.agent_true_hist[0]

    all_y = data.agent_true_hist[1]

    all_z = data.agent_true_hist[2]

    margin = 2

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Set limits
    ax.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax.set_ylim(all_y.min()-margin, all_y.max()+margin)
    ax.set_zlim(all_z.min()-margin, all_z.max()+margin)

    # ----------------------------------------
    # Create artists
    # ----------------------------------------

    initial_pts = []
    traj_lines = []
    drone_pts = []
    desired_pts = []

    agent_pt, = ax.plot([], [], [], 'ko', markersize=8)
    agent_line, = ax.plot([], [], [], 'k', lw=3)

    def _camera_view_dir(ax):
        """
        Unit vector pointing from the scene toward the camera,
        derived from the current azimuth/elevation. Projecting a
        point's position onto this vector gives a usable proxy for
        'distance toward the viewer' -> larger = closer to camera.
        """
        azim = np.deg2rad(ax.azim)
        elev = np.deg2rad(ax.elev)
        return np.array([
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ])

    def update(frame):
        idx = frame_indices[:frame + 1]
        start = max(0, idx[-1] - param.display.trail_length)
        idx = idx[(idx >= start)]

        # agent 
        agent_pt.set_data(
            [data.agent_true_hist[0, idx[-1]]],
            [data.agent_true_hist[1, idx[-1]]]
        )
        agent_pt.set_3d_properties(
            [data.agent_true_hist[2, idx[-1]]]
        )

        # agent trajectory
        agent_line.set_data(
            data.agent_true_hist[0, idx],
            data.agent_true_hist[1, idx]
        )
        agent_line.set_3d_properties(
            data.agent_true_hist[2, idx]
        )

        # Current position of every agent (for depth ranking) plus target
        current_positions = [
            data.position_history[iD, :, idx[-1]] for iD in range(param.sim.nD)
        ]
        current_positions.append(np.array([
            data.agent_true_hist[0, idx[-1]],
            data.agent_true_hist[1, idx[-1]],
            data.agent_true_hist[2, idx[-1]],
        ]))

        artists = (
            [agent_pt, agent_line]
        )

        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000/param.display.mp4_fps,  # milliseconds
        blit=False,       # should be False for 3D plots
        repeat=False
    )

    # Save animation (requires ffmpeg)
    if param.display.save_to_mp4:
        print("Saving trajectory animation")
        current_folder = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_folder, param.display.save_location)
        ani.save(
            os.path.join(
                file_path,
                f"trajectory_{param.sim.nD}_agents.mp4",
            ),
            writer="ffmpeg",
            fps=param.display.mp4_fps,
        )

    plt.show()

if __name__ == "__main__":
    param = Param()
    param.display.save_to_mp4 = False # we are assuming that the user as 
    # previously saved the plots
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "simulation_results.pkl")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    sim_data = SimulationData(**data)

    create_figures(sim_data, param)