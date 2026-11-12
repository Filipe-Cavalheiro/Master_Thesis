import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

from Structs import Param, SimulationData
from flight_code_funcs import PHASE_COLORS, PHASES

def create_figures(data: SimulationData, param: Param):
    if param.display.composed is True:
        composed_fig(data, param)

    if param.display.radial_hist_plot is True:
        radial_hist_plot(data, param)

    if param.display.radial_hist_animation is True:
        radial_hist_animation(data, param)

    if param.display.consensus_plot is True:
        consensus_plot(data, param)

    if param.display.tracking_error_plot is True:
        tracking_error_plot(data, param)
 
    if param.display.interagent_geodesic_distance_plot is True:
        interagent_geodesic_distance_plot(data, param)

    if param.display.trajectory_plot is True:
        trajectory_plot(data, param)

    if param.display.trajectory_animate is True:
        animate_simulation(data, param)

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

def shade_flight_phases(ax, data: SimulationData):
    """Background color of every flight phase (Take off, Main Controller, ...), when the run has them (main.py)."""
    if not data.phases:
        return
    order = [p for p in PHASES if p in data.phases]
    for k, ph in enumerate(order):
        t1 = data.phases[order[k + 1]] if k + 1 < len(order) else data.phases.get("EOF")
        if t1 is None:
            continue
        ax.axvspan(data.phases[ph], t1, color=PHASE_COLORS[ph], alpha=0.6, lw=0, zorder=0, label=ph)

def composed_fig(data: SimulationData, param: Param):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=None, azim=None)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    target_hist = data.target_true_hist
    position_history = data.position_history

    # Remove steps above simulation time
    steps = np.array([50, 300, 600, 1000, 1500, 2000, 3000])
    steps = steps[steps < param.sim.Nsim]

    # === Create legend handles ===
    hdrone = ax.scatter(
        [], [], [],
        s=50,
        c='k',
        marker='x',
        label=r'$p_i$'
    )

    htarget_path, = ax.plot(
        [], [], [],
        'k--',
        label=r'$p_T$ path'
    )

    htarget_pos = ax.scatter(
        [],
        [],
        [],
        c='k',
        marker='o',
        label=r'$p_T$'
    )

    hpath, = ax.plot(
        [],
        [],
        [],
        'k',
        label=r'$p_i$ path'
    )

    hfinal_pos = ax.scatter(
        [],
        [],
        [],
        c='k',
        marker='d',
        label=r'$p^\star$'
    )

    lgd = ax.legend(
        handles=[
            hdrone,
            hpath,
            htarget_pos,
            htarget_path,
            hfinal_pos
        ],
        fontsize=10,
        loc='upper right'
    )
    # === End legend creation ===

    trail_length = param.display.trail_length
    for step in steps:
        idx = np.arange(
            max(0, step - trail_length),
            step
        )

        # --- target position at timestep step ---
        ax.scatter(
            target_hist[0, step],
            target_hist[1, step],
            target_hist[2, step],
            c='k',
            marker='o'
        )

        ax.text(
            target_hist[0, step] - 2,
            target_hist[1, step] - 2,
            target_hist[2, step] + 8,
            f"   {step/100:.2f} sec",
            fontsize=8
        )

        # --- chaser positions ---
        for iD in range(param.sim.nD):
            pos = position_history[iD,:,idx].T

            ax.plot(
                pos[0, :],
                pos[1, :],
                pos[2, :],
                color=data.colors[iD, :]
            )

            ax.scatter(
                pos[0, -1],
                pos[1, -1],
                pos[2, -1],
                marker='x',
                color=data.colors[iD, :]
            )

    if not os.path.isdir(param.display.save_location):
        os.makedirs(param.display.save_location)

    print("saving composed plot")
    plt.savefig(
        f"{param.display.save_location}/composed_{param.sim.nD}_agents.pdf",
        bbox_inches='tight'
    )

    plt.show(block=False)

def radial_hist_plot(data: SimulationData, param: Param):
    """
    Plot radial distance history between all chasers and target.

    Parameters
    ----------
    data.target_position : ndarray
        Target position history (3 x N)
    data.chaser : list
        List containing chaser state histories
    param : dict
        Contains:
            - dTi : timestep
            - Nsim: number of simulation steps
            - nD  : number of chasers
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)

    t = np.arange(param.sim.Nsim) * param.sim.dTi
    shade_flight_phases(ax, data)

    for iD in range(param.sim.nD):
        ax.plot(
            t,
            data.radial_history[iD],
            linewidth=2,
            color=data.colors[iD],
            label=f"Agent {iD+1}"
        )

    # Desired sphere radius
    ax.axhline(
        param.sim.sphere_radius,
        color="k",
        linestyle="--",
        linewidth=2,
        label="Desired radius"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance to target [m]")

    ax.grid(True)
    if param.sim.nD <= 6:
        ax.legend(fontsize=16)

    plt.tight_layout()

    os.makedirs(param.display.save_location, exist_ok=True)

    print("saving radial_hist plot")
    plt.savefig(
        os.path.join(
            param.display.save_location,
            f"radial_hist_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )
    print("Done.")

    plt.show(block=False)

def radial_hist_animation(data: SimulationData, param: Param):
    """
    Animate radial distance history between all chasers and the target.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    t = np.arange(param.sim.Nsim) * param.sim.dTi
    shade_flight_phases(ax, data)

    # Create an empty line for each agent
    lines = []
    for iD in range(param.sim.nD):
        line, = ax.plot(
            [],
            [],
            linewidth=2,
            color=data.colors[iD],
            label=f"Agent {iD+1}"
        )
        lines.append(line)

    # Desired radius
    ax.axhline(
        param.sim.sphere_radius,
        color="k",
        linestyle="--",
        linewidth=2,
        label="Desired radius"
    )

    ax.set_xlim(t[0], t[-1])

    # Automatically scale y-axis
    ymin = min(np.min(r) for r in data.radial_history)
    ymax = max(np.max(r) for r in data.radial_history)
    margin = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance to target [m]")
    ax.grid(True)
    ax.legend()

    frame_indices = np.linspace(
        0,
        param.sim.Nsim - 1,
        int(param.sim.Tend * param.display.mp4_fps),
        dtype=int,
    )

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        idx = frame_indices[:frame + 1]
        for iD, line in enumerate(lines):
            line.set_data(
                t[idx],
                data.radial_history[iD][idx],
            )

        return lines
        
    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000/param.display.mp4_fps,  # milliseconds
        blit=True,
        repeat=False,
    )

    plt.tight_layout()

    os.makedirs(param.display.save_location, exist_ok=True)

    # Save animation (requires ffmpeg)
    if param.display.save_to_mp4:
        print("Saving radial_history animation")
        ani.save(
            os.path.join(
                param.display.save_location,
                f"radial_hist_{param.sim.nD}_agents.mp4",
            ),
            writer="ffmpeg",
            fps = param.display.mp4_fps
        )

    plt.show()

def consensus_plot(data: SimulationData, param: Param):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])

    target_position = data.target_true_hist
    perceived_target_hist = data.perceived_target_hist

    # Plot true target trajectory
    ax.plot(
        target_position[0, :],
        target_position[1, :],
        target_position[2, :],
        color="k",
        linewidth=3,
        label="True target"
    )

    Ks = len(perceived_target_hist)
    Is = len(perceived_target_hist[0])
    for i in range(Is):
        x = np.full(Ks, np.nan)
        y = np.full(Ks, np.nan)
        z = np.full(Ks, np.nan)

        for k in range(Ks):

            est = perceived_target_hist[k][i]

            if est is not None and len(est) > 0:
                # expected 18x1 vector
                x[k] = est[0]
                y[k] = est[1]
                z[k] = est[2]

        # Plot estimated trajectory
        ax.plot(
            x,
            y,
            z,
            linewidth=2,
            label=f"Drone {i+1} est"
        )

        # Find valid indices
        valid_idx = np.where(~np.isnan(x))[0]

        if len(valid_idx) > 0:
            # Start point
            ax.scatter(
                x[valid_idx[0]],
                y[valid_idx[0]],
                z[valid_idx[0]],
                s=80,
                marker="o"
            )

            # End point
            ax.scatter(
                x[valid_idx[-1]],
                y[valid_idx[-1]],
                z[valid_idx[-1]],
                s=80,
                marker="d"
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()

    if not os.path.isdir(param.display.save_location):
        os.makedirs(param.display.save_location)

    plt.savefig(
        f"{param.display.save_location}/consensus_{param.sim.nD}_agents.pdf",
        bbox_inches="tight"
    )

    plt.show(block=False)


def tracking_error_plot(data: SimulationData, param: Param):
    # ---------------------------------------------------------
    # Plot position tracking error (actual vs desired) per agent
    # ---------------------------------------------------------
    time_vec = np.arange(param.sim.Nsim) * param.sim.dTi
    # position_history, p_d_hist shape: [nD, 3, Nsim]
    pos_error = data.position_history - data.p_d_hist          # component-wise diff, shape [nD, 3, Nsim]
    pos_error_norm = np.linalg.norm(pos_error, axis=1)  # Euclidean norm per agent, per timestep -> [nD, Nsim]

    fig, ax = plt.subplots(figsize=(9, 5))
    shade_flight_phases(ax, data)
    for iD in range(param.sim.nD):
        ax.plot(time_vec, pos_error_norm[iD, :],
                 label=f"Drone {iD}", color=data.colors[iD], linewidth=1.2)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Position tracking error $\|p - r_i q^\star\|$ [m]")
    ax.set_title("Agent Position Tracking Error Over Time")
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.grid(True, alpha=0.3)
    if param.sim.nD <= 6:
        ax.legend(loc="upper right", fontsize=16, ncol=2)
    fig.tight_layout()

    print("saving tracking_error plot")
    plt.savefig(
        os.path.join(
            param.display.save_location,
            f"tracking_error_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )
    print("Done.")

    plt.show(block=False)

def interagent_geodesic_distance_plot(data: SimulationData, param: Param):
    # ---------------------------------------------------------
    # Plot inter-agent pairwise distances over time
    # ---------------------------------------------------------
    agent_pairs = data.agent_pairs
    inter_agent_dist_hist = data.inter_agent_dist_hist
    time_vec = np.arange(param.sim.Nsim) * param.sim.dTi

    fig, ax = plt.subplots(figsize=(9, 5))
    shade_flight_phases(ax, data)
    for p_idx, (i, j) in enumerate(agent_pairs):
        ax.plot(time_vec, inter_agent_dist_hist[p_idx, :],
                label=f"Drone {i}-{j}", linewidth=1.2)

    # Highlight the minimum inter-agent distance at every timestep
    ax.plot(time_vec, np.ones([len(time_vec),1])*data.min_safe_dist, color="k", linewidth=2.0,
            linestyle="--", label="Min distance")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Inter-agent distance [m]")
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    
    ax.grid(True, alpha=0.3)
    if param.sim.nD <= 6:
        ax.legend(loc="upper right", fontsize=16, ncol=2)
    fig.tight_layout()

    print("saving inter_agent_dist plot")
    plt.savefig(
        os.path.join(
            param.display.save_location,
            f"inter_agent_dist_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )
    print("Done.")

    plt.show(block=False)

def trajectory_plot(data, param):
    """
    Static 3D plot of the complete trajectories of all agents and the
    target, using the full simulation history (no frame subsampling,
    no animation).
    """

    # -----------------------------
    # Compute axis limits (full history, not just sampled frames)
    # -----------------------------
    all_x = np.concatenate([
        data.position_history[:, 0, :].ravel(),
        data.target_true_hist[0]
    ])
    all_y = np.concatenate([
        data.position_history[:, 1, :].ravel(),
        data.target_true_hist[1]
    ])
    all_z = np.concatenate([
        data.position_history[:, 2, :].ravel(),
        data.target_true_hist[2]
    ])

    margin = 2

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Disable mplot3d's unreliable automatic depth sort; we assign
    # zorder ourselves below so crossing trajectories occlude correctly.
    ax.computed_zorder = False

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

    view_dir = np.array([
        np.cos(np.deg2rad(ax.elev)) * np.cos(np.deg2rad(ax.azim)),
        np.cos(np.deg2rad(ax.elev)) * np.sin(np.deg2rad(ax.azim)),
        np.sin(np.deg2rad(ax.elev)),
    ])

    # Rank agents + target by average distance toward the camera so
    # nearer trajectories are drawn on top of farther ones.
    agent_centroids = [
        data.position_history[iD, :, :].mean(axis=1) for iD in range(param.sim.nD)
    ]
    target_centroid = np.array([
        data.target_true_hist[0].mean(),
        data.target_true_hist[1].mean(),
        data.target_true_hist[2].mean(),
    ])
    centroids = agent_centroids + [target_centroid]
    depths = np.array([np.dot(c, view_dir) for c in centroids])
    order = np.argsort(depths)  # farthest first
    zorder_for = {item_idx: rank + 1 for rank, item_idx in enumerate(order)}
    target_idx = param.sim.nD

    # Target: full trajectory + start/end markers
    ax.plot(
        data.target_true_hist[0],
        data.target_true_hist[1],
        data.target_true_hist[2],
        'k', lw=3, label='target',
        zorder=zorder_for[target_idx],
    )
    ax.plot(
        [data.target_true_hist[0, 0]],
        [data.target_true_hist[1, 0]],
        [data.target_true_hist[2, 0]],
        'ko', markersize=8, zorder=zorder_for[target_idx],
    )
    ax.plot(
        [data.target_true_hist[0, -1]],
        [data.target_true_hist[1, -1]],
        [data.target_true_hist[2, -1]],
        'k*', markersize=12, zorder=zorder_for[target_idx],
    )

    for iD in range(param.sim.nD):
        pos = data.position_history[iD]
        z = zorder_for[iD]

        # Full trajectory
        ax.plot(
            pos[0, :], pos[1, :], pos[2, :],
            color=data.colors[iD], lw=2, label=f'agent {iD}',
            zorder=z,
        )

        # Start marker
        ax.plot(
            [pos[0, 0]], [pos[1, 0]], [pos[2, 0]],
            marker='x', linestyle='', color=data.colors[iD],
            markersize=8, zorder=z,
        )

        # End marker
        ax.plot(
            [pos[0, -1]], [pos[1, -1]], [pos[2, -1]],
            marker='o', linestyle='', color=data.colors[iD],
            markersize=8, zorder=z,
        )

    if param.sim.nD <= 6: 
        ax.legend(loc='upper right', fontsize=16)

    print("saving trajectory plot")
    plt.savefig(
        os.path.join(
            param.display.save_location,
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
    all_x = np.concatenate([
        data.position_history[:, 0, frame_indices].ravel(),
        data.target_true_hist[0]
    ])

    all_y = np.concatenate([
        data.position_history[:, 1, frame_indices].ravel(),
        data.target_true_hist[1]
    ])

    all_z = np.concatenate([
        data.position_history[:, 2, frame_indices].ravel(),
        data.target_true_hist[2]
    ])

    margin = 2

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --------------------------------------------------------------
    # Disable matplotlib's built-in (unreliable) depth sorting.
    # We'll assign zorder manually every frame based on true camera
    # distance, so agents in front correctly occlude agents behind.
    # --------------------------------------------------------------
    ax.computed_zorder = False

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

    target_pt, = ax.plot([], [], [], 'ko', markersize=8)
    target_line, = ax.plot([], [], [], 'k', lw=3)

    for iD in range(param.sim.nD):

        # Initial position
        p, = ax.plot(
            [data.position_history[iD,0,0]],
            [data.position_history[iD,1,0]],
            [data.position_history[iD,2,0]],
            marker='x',
            linestyle='',
            color=data.colors[iD]
        )
        initial_pts.append(p)

        # Trajectory
        line, = ax.plot([], [], [], color=data.colors[iD], lw=2)
        traj_lines.append(line)

        # Current position
        p, = ax.plot([], [], [], marker='x',
                    linestyle='',
                    color=data.colors[iD],
                    markersize=8)
        drone_pts.append(p)

        # Desired position
        p, = ax.plot([], [], [], marker='D',
                    linestyle='',
                    color=data.colors[iD],
                    markersize=6)
        desired_pts.append(p)

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

        # Target
        target_pt.set_data(
            [data.target_true_hist[0, idx[-1]]],
            [data.target_true_hist[1, idx[-1]]]
        )
        target_pt.set_3d_properties(
            [data.target_true_hist[2, idx[-1]]]
        )

        # Target trajectory
        target_line.set_data(
            data.target_true_hist[0, idx],
            data.target_true_hist[1, idx]
        )
        target_line.set_3d_properties(
            data.target_true_hist[2, idx]
        )

        view_dir = _camera_view_dir(ax)

        # Current position of every agent (for depth ranking) plus target
        current_positions = [
            data.position_history[iD, :, idx[-1]] for iD in range(param.sim.nD)
        ]
        current_positions.append(np.array([
            data.target_true_hist[0, idx[-1]],
            data.target_true_hist[1, idx[-1]],
            data.target_true_hist[2, idx[-1]],
        ]))

        # Depth = projection onto the camera direction; larger -> nearer camera
        depths = np.array([np.dot(p, view_dir) for p in current_positions])

        # Rank from farthest (drawn first / lowest zorder) to nearest (drawn last / highest)
        order = np.argsort(depths)
        n_items = len(current_positions)
        zorder_for = {}
        for rank, item_idx in enumerate(order):
            zorder_for[item_idx] = rank + 1  # 1..n_items, higher = nearer camera

        target_idx = param.sim.nD  # target is the last entry appended above
        target_pt.set_zorder(zorder_for[target_idx])
        target_line.set_zorder(zorder_for[target_idx])

        for iD in range(param.sim.nD):

            pos = data.position_history[iD]
            z = zorder_for[iD]

            # Initial position
            initial_pts[iD].set_visible(start == 0)
            initial_pts[iD].set_zorder(z)

            # Trajectory
            traj_lines[iD].set_data(
                pos[0, idx],
                pos[1, idx]
            )
            traj_lines[iD].set_3d_properties(
                pos[2, idx]
            )
            traj_lines[iD].set_zorder(z)

            # Drone position
            drone_pts[iD].set_data(
                [pos[0, idx[-1]]],
                [pos[1, idx[-1]]]
            )
            drone_pts[iD].set_3d_properties(
                [pos[2, idx[-1]]]
            )
            drone_pts[iD].set_zorder(z)

            # Desired position
            desired_pts[iD].set_data(
                [data.p_d_hist[iD][0, idx[-1]]],
                [data.p_d_hist[iD][1, idx[-1]]]
            )
            desired_pts[iD].set_3d_properties(
                [data.p_d_hist[iD][2, idx[-1]]]
            )
            desired_pts[iD].set_zorder(z)

        artists = (
            initial_pts +
            traj_lines +
            drone_pts +
            desired_pts +
            [target_pt, target_line]
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
        ani.save(
            os.path.join(
                param.display.save_location,
                f"trajectory_{param.sim.nD}_agents.mp4",
            ),
            writer="ffmpeg",
            fps=param.display.mp4_fps,
        )

    plt.show()


if __name__ == "__main__":
    # Figures / animations from the stored simulation (simulation_results.pkl,
    # written by main.py when param.display.store_data is True), without
    # re-simulating. Which figures: the GraphicalDisplay flags in Structs.py.
    import pickle
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # flight_sims/ (root_results)
    import root_results

    param = Param()
    root_results.mirror_on_exit("swarm_geometric_Fibonacci_sim_target", param.display.save_location)
    param.display.save_to_mp4 = False   # we are assuming that the user has previously saved the mp4s

    file_path = Path(__file__).resolve().parent / "simulation_results.pkl"
    if not file_path.exists():
        raise SystemExit(f"No {file_path.name}: run main.py first (with param.display.store_data = True)")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # param.sim of the stored run (main.py saves it): the plots index the histories with it
    for key, value in data.pop("sim", {}).items():
        setattr(param.sim, key, value)
    sim_data = SimulationData(**data)

    nD, _, Nsim = sim_data.position_history.shape
    if (nD, Nsim) != (param.sim.nD, param.sim.Nsim):
        raise SystemExit(f"{file_path.name} has {nD} drones x {Nsim} steps, Structs.py now "
                         f"{param.sim.nD} x {param.sim.Nsim}: set nD / Tend / dTi as in that run, or re-run the simulation")

    os.makedirs(param.display.save_location, exist_ok=True)

    create_figures(sim_data, param)
