import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import pickle
import itertools
import warnings
from voronoi_mesh_funcs import pos_based_mesh_transform
from Structs_sim import Param, SimulationData, Voronoi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from voronoi_mesh_funcs import pre_compute_desired_pos

def create_figures(data: SimulationData, param: Param):
    voronoi = Voronoi(param)
    origin_p_d, swarm_center, voronoi = pre_compute_desired_pos(voronoi, param)
    if param.display.composed is True:
        composed_plot(data, voronoi, param)

    detailed_close_up_bool = True
    if detailed_close_up_bool is True:
        detailed_close_up(data, voronoi, param)

    if param.display.three_view_plot is True:
        three_view_plot(data, param)

    # Radial History
    if param.display.radial_hist_plot is True:
        radial_hist_plot(data, param)

    if param.display.radial_hist_animation is True:
        radial_hist_animation(data, param)

    # Inter agent distance
    if param.display.inter_agent_dist is True:
        inter_agent_dist_plot(data, param)

    # Mellinger tracking error
    if param.display.low_level_tracking_error is True:
        low_level_tracking_error_plot(data, param)

    # Consensus
    if param.display.consensus_plot is True:
        consensus_plot(data, param)

    # Velocitu
    if param.display.velocity_plot is True:
        velocity_plot(data, param)

    if param.display.velocity_animate is True:
        velocity_animate(data, param)

#    # Acceleration
#    if param.display.acceleration_plot is True:
#        acceleration_plot(data, param)
#
#    if param.display.acceleration_animate is True:
#        acceleration_animate(data, param)

    # Trajectory
    if param.display.trajectory_plot is True:
        trajectory_plot(data, param)

    if param.display.trajectory_animate is True:
       trajectory_animate(data, param)

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

def composed_plot(data: SimulationData, voronoi: Voronoi, param: Param):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(True)
    ax.view_init(elev=None, azim=None)
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_zlabel("Z [m]", fontsize=12)

    target_hist = data.target_true_hist
    position_history = data.position_history

    # Remove steps above simulation time
    steps = np.array([100, 250, 400, 1000, 1300, 1900])
    steps = steps[steps < param.sim.Nsim]

    # === Create legend handles ===
    htarget_pos = ax.scatter(
        [], [], [],
        c='r',
        marker='o',
        label=r'$p_T$'
    )

    lgd = ax.legend(
        handles=[
            htarget_pos
        ],
        fontsize=10,
        loc='upper right'
    )
    # === End legend creation ===


    trail_length = param.display.trail_length

    # --------------------------------------------------
    # Save directory
    # --------------------------------------------------

    current_folder = os.path.dirname(os.path.abspath(__file__))
    save_location = current_folder + param.display.save_location
    print("Saving composed plot")

    for step in steps:

        idx = np.arange(
            max(0, step - trail_length),
            step
        )

        # --------------------------------------------------
        # Target position and mesh at this timestep
        # --------------------------------------------------
        target_pos = target_hist[0:3, step].reshape(-1, 1)

        (voronoi.vertices,
         voronoi.faces,
         voronoi.mesh_pos) = pos_based_mesh_transform(
            target_pos,
            voronoi,
            param
        )

        # --------------------------------------------------
        # Mesh
        # --------------------------------------------------
        mesh = Poly3DCollection(
            voronoi.vertices[voronoi.faces],
            alpha=0.15,
            edgecolor='k',
            linewidth=0.5
        )

        ax.add_collection3d(mesh)

        # --------------------------------------------------
        # Time label above the mesh
        # --------------------------------------------------
        mesh_z_max = np.max(voronoi.vertices[:, 2])

        mesh_x_center = np.mean(voronoi.vertices[:, 0])
        mesh_y_center = np.mean(voronoi.vertices[:, 1])

        # --------------------------------------------------
        # Target position
        # --------------------------------------------------
        ax.scatter(
            target_hist[0, step],
            target_hist[1, step],
            target_hist[2, step],
            c='r',
            marker='o'
        )

    # Make X, Y, Z use the same scale
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    center = np.array([
        np.mean(xlim),
        np.mean(ylim),
        np.mean(zlim)
    ])

    half_range = max(
        np.ptp(xlim),
        np.ptp(ylim),
        np.ptp(zlim)
    ) / 3

    ax.set_xlim3d(center[0] - half_range, center[0] + half_range)
    ax.set_ylim3d(center[1] - half_range, center[1] + half_range)
    ax.set_zlim3d(center[2] - half_range, center[2] + half_range)
    ax.view_init(elev=30, azim=120)
    ax.set_box_aspect([1, 1, 1])

    plt.savefig(
        os.path.join(save_location, f"composed_{param.sim.nD}_agents.pdf"),
        bbox_inches="tight"
    )
    
    plt.show(block=False)

def detailed_close_up(data: SimulationData, voronoi: Voronoi, param: Param):
    target_hist = data.target_true_hist
    position_history = data.position_history
    desired_positions = data.p_d_hist

    # Time steps to inspect
    steps = np.array([100, 250, 400, 1000, 1300, 1900])
    steps = steps[steps < param.sim.Nsim]

    # Agents that remain active after the drop
    active = set(int(i) for i in data.active_ids)

    # Desired close-up range
    close_up_range = 20.0

    # Extra margin around the plotted objects
    margin = 2.0

    # --------------------------------------------------
    # Save directory
    # --------------------------------------------------

    current_folder = os.path.dirname(os.path.abspath(__file__))
    save_location = current_folder + param.display.save_location

    os.makedirs(save_location, exist_ok=True)

    print("Saving detailed close up plots")

    for step in steps:

        # -----------------------------------------------------
        # Create a NEW figure for this timestep
        # -----------------------------------------------------

        fig = plt.figure(figsize=(7, 7))

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        # -----------------------------------------------------
        # Target position
        # -----------------------------------------------------

        target_pos = target_hist[0:3, step].reshape(-1)

        # -----------------------------------------------------
        # Determine which agents should be plotted
        # -----------------------------------------------------

        if step <= param.sim.nD_drop_time:
            agent_ids = list(range(param.sim.nD))
        else:
            agent_ids = list(active)

        # -----------------------------------------------------
        # Plot target
        # -----------------------------------------------------

        ax.scatter(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            c="k",
            marker="o",
            s=100,
            label=r"$p_T$",
            depthshade=True
        )

        # -----------------------------------------------------
        # Collect positions for axis limits
        # -----------------------------------------------------

        positions_to_show = [target_pos]

        # -----------------------------------------------------
        # Plot agents and desired positions
        # -----------------------------------------------------

        for iD in agent_ids:

            # Actual position
            pos = position_history[iD, :, step]

            # Desired position
            p_d = desired_positions[iD, :, step]

            # ---------------------------------------------
            # Actual position
            # ---------------------------------------------

            if np.all(np.isfinite(pos)):

                positions_to_show.append(pos)

                ax.scatter(
                    pos[0],
                    pos[1],
                    pos[2],
                    marker="x",
                    s=100,
                    color=data.colors[iD, :],
                    linewidths=2.5,
                    depthshade=False
                )

            # ---------------------------------------------
            # Desired position
            # ---------------------------------------------

            if np.all(np.isfinite(p_d)):

                positions_to_show.append(p_d)

                ax.scatter(
                    p_d[0],
                    p_d[1],
                    p_d[2],
                    marker="o",
                    s=60,
                    facecolors="none",
                    edgecolors=data.colors[iD, :],
                    linewidths=1.8,
                    depthshade=False
                )

        # -----------------------------------------------------
        # Plot mesh
        # -----------------------------------------------------

        (voronoi.vertices,
         voronoi.faces,
         voronoi.mesh_pos) = pos_based_mesh_transform(
            target_pos.reshape(-1, 1),
            voronoi,
            param
        )

        mesh_vertices = voronoi.vertices
        mesh_faces = voronoi.faces

        mesh_polygons = [
            mesh_vertices[face]
            for face in mesh_faces
        ]

        mesh = Poly3DCollection(
            mesh_polygons,
            alpha=0.15,
            edgecolor="k",
            linewidth=0.5
        )

        ax.add_collection3d(mesh)

        # -----------------------------------------------------
        # Determine axis limits
        #
        # Make sure target + ALL agents + desired positions
        # are visible.
        # -----------------------------------------------------

        positions_to_show = np.asarray(positions_to_show)

        min_pos = np.min(positions_to_show, axis=0)
        max_pos = np.max(positions_to_show, axis=0)

        # Center of all relevant objects
        center = 0.5 * (min_pos + max_pos)

        # Largest required half-range
        required_half_range = 0.5 * np.max(max_pos - min_pos)

        # At least the requested close-up range
        half_range = max(
            close_up_range / 2,
            required_half_range + margin
        )

        # -----------------------------------------------------
        # Set equal axis limits
        # -----------------------------------------------------

        ax.set_xlim(
            center[0] - half_range,
            center[0] + half_range
        )

        ax.set_ylim(
            center[1] - half_range,
            center[1] + half_range
        )

        ax.set_zlim(
            center[2] - half_range,
            center[2] + half_range
        )

        # -----------------------------------------------------
        # Appearance
        # -----------------------------------------------------

        ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("X [m]", fontsize=16)
        ax.set_ylabel("Y [m]", fontsize=16)
        ax.set_zlabel("Z [m]", fontsize=16)

        ax.set_title(
            f"$t = {step / 100:.2f}$ s",
            fontsize=14
        )

        ax.grid(True)

        ax.view_init(
            elev=30,
            azim=120
        )

        plt.tight_layout()

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        plt.savefig(
            os.path.join(
                save_location,
                f"close_up_{step}_{param.sim.nD}_agents.pdf"
            ),
            bbox_inches="tight"
        )

        # Non-blocking so all figures can be generated
        plt.show(block=False)


def _draw_formation(ax, target_pos, desired_positions, valid_desired,
                    colors, dims):
    """
    Draw target and desired drone positions on ax.
    dims: tuple of coordinate indices to plot, e.g. (0, 1, 2), (0, 1), (0, 2)
    """
    # Target
    ax.scatter(
        *[target_pos[d] for d in dims],
        c="k",
        marker="o",
        s=100,
        label="Target",
    )

    # Desired drone positions (label only the first one)
    first = True
    for iD in range(len(desired_positions)):
        if not valid_desired[iD]:
            continue

        pos_d = desired_positions[iD]

        ax.scatter(
            *[pos_d[d] for d in dims],
            facecolors="none",
            edgecolors=colors[iD],
            marker="D",
            s=60,
            linewidths=1.5,
            label="Desired position" if first else None,
        )
        first = False

def three_view_plot(data: SimulationData, param: Param):
    """
    Create and save three views of the final formation:

        1. 3D view
        2. Top view  (X-Y)
        3. Front view (X-Z)

    The views show:
        - True target position
        - Desired final drone positions

    All views are centered on the final target position and the axis
    limits are chosen so that all actual and desired positions are visible.
    """

    # --------------------------------------------------
    # Final simulation state
    # --------------------------------------------------

    target_pos = data.target_true_hist[:3, -1]
    desired_positions = data.p_d_hist[:, :, -1]  # (nD, 3)

    valid_desired = ~np.isnan(desired_positions).any(axis=1)
    desired_valid = desired_positions[valid_desired]

    # --------------------------------------------------
    # Axis limits centered on target
    # --------------------------------------------------

    all_positions = np.vstack([target_pos.reshape(1, 3), desired_valid])

    center = target_pos
    max_dist = np.max(np.abs(all_positions - center), axis=0)
    max_dist = np.maximum(max_dist, 1.0)      # prevent zero-sized axes

    margin = 0.15
    half_ranges = max_dist * (1.0 + margin)

    # --------------------------------------------------
    # Save directory
    # --------------------------------------------------

    current_folder = os.path.dirname(os.path.abspath(__file__))
    save_location = current_folder + param.display.save_location

    print("Saving 3-view plots")

    # ==================================================
    # 1. 3D VIEW
    # ==================================================

    fig3d = plt.figure(figsize=(9, 8))
    ax3d = fig3d.add_subplot(111, projection="3d")

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.grid(True)
    ax3d.set_box_aspect([1, 1, 1])

    _draw_formation(ax3d, target_pos, desired_positions, valid_desired,
                    data.colors, dims=(0, 1, 2))

    ax3d.set_xlim(center[0] - half_ranges[0], center[0] + half_ranges[0])
    ax3d.set_ylim(center[1] - half_ranges[1], center[1] + half_ranges[1])
    ax3d.set_zlim(center[2] - half_ranges[2], center[2] + half_ranges[2])

    ax3d.legend(fontsize=9, loc="upper right")
    fig3d.tight_layout()

    print("saving 3d view plot")
    fig3d.savefig(
        os.path.join(save_location, f"3_view_3d_{param.sim.nD}_agents.pdf"),
        bbox_inches="tight"
    )

    # ==================================================
    # 2. TOP VIEW — X-Y
    # ==================================================

    fig_top, ax_top = plt.subplots(figsize=(8, 8))

    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")
    ax_top.grid(True)
    ax_top.set_aspect("equal", adjustable="box")

    _draw_formation(ax_top, target_pos, desired_positions, valid_desired,
                    data.colors, dims=(0, 1))

    ax_top.set_xlim(center[0] - half_ranges[0], center[0] + half_ranges[0])
    ax_top.set_ylim(center[1] - half_ranges[1], center[1] + half_ranges[1])

    ax_top.legend(fontsize=9, loc="upper right")
    fig_top.tight_layout()

    print("saving top view plot")
    fig_top.savefig(
        os.path.join(save_location, f"3_view_top_{param.sim.nD}_agents.pdf"),
        bbox_inches="tight"
    )

    # ==================================================
    # 3. FRONT VIEW — X-Z
    # ==================================================

    fig_front, ax_front = plt.subplots(figsize=(8, 8))

    ax_front.set_xlabel("X [m]")
    ax_front.set_ylabel("Z [m]")
    ax_front.grid(True)
    ax_front.set_aspect("equal", adjustable="box")

    _draw_formation(ax_front, target_pos, desired_positions, valid_desired,
                    data.colors, dims=(0, 2))

    ax_front.set_xlim(center[0] - half_ranges[0], center[0] + half_ranges[0])
    ax_front.set_ylim(center[2] - half_ranges[2], center[2] + half_ranges[2])

    ax_front.legend(fontsize=9, loc="upper right")
    fig_front.tight_layout()

    print("saving front view plot")
    fig_front.savefig(
        os.path.join(save_location, f"3_view_front_{param.sim.nD}_agents.pdf"),
        bbox_inches="tight"
    )

    plt.show(block=False)

def radial_hist_plot(data: SimulationData, param: Param):
    """
    Plot radial distance history of the agents that were dropped
    (i.e. not in data.active_ids), from t=0 up to param.sim.nD_drop_time.

    Parameters
    ----------
    data.radial_history : list
        Radial distance history per agent
    data.active_ids : iterable
        Ids of agents that survived the entire simulation
    param : Param
        Contains:
            - sim.dTi            : timestep
            - sim.Nsim           : number of simulation steps
            - sim.nD             : total number of chasers
            - sim.nD_drop_time   : time at which agents are dropped
    """

    # Agents that died at the drop time
    active = set(int(i) for i in data.active_ids)
    dropped_ids = [iD for iD in range(param.sim.nD) if iD not in active]

    if not dropped_ids:
        print("No dropped agents to plot in radial_hist_plot.")
        return

    # Only use the time window up to the drop time
    N_drop = param.sim.nD_drop_time
    t = np.arange(N_drop) * param.sim.dTi

    fig, ax = plt.subplots(figsize=(8, 5))

    for iD in dropped_ids:
        ax.plot(
            t,
            np.asarray(data.radial_history[iD])[:N_drop],
            linewidth=2,
            color=data.colors[iD],
        )

    t = np.arange(param.sim.Nsim) * param.sim.dTi
    for iD in active:
        ax.plot(
            t,
            np.asarray(data.radial_history[iD]),
            linewidth=2,
            color=data.colors[iD],
        )

    # Desired sphere radius
    ax.axhline(
        param.sim.sphere_radius,
        color="k",
        linestyle="--",
        linewidth=2,
        label="Desired radius"
    )

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Distance to target [m]", fontsize=16)

    ax.grid(True)

    print("saving radial_hist plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"radial_hist_{len(dropped_ids)}_dropped_agents.pdf"
        ),
        bbox_inches="tight"
    )

    plt.show(block=False)

def radial_hist_animation(data: SimulationData, param: Param):
    """
    Animate radial distance history of the agents that were dropped
    (i.e. not in data.active_ids), from t=0 up to param.sim.nD_drop_time.
    """

    # Agents that died at the drop time
    active = set(int(i) for i in data.active_ids)
    dropped_ids = [iD for iD in range(param.sim.nD) if iD not in active]

    if not dropped_ids:
        print("No dropped agents to plot in radial_hist_animation.")
        return

    # Only use the time window up to the drop time
    t_full = np.arange(param.sim.Nsim) * param.sim.dTi
    N_drop = min(int(round(param.sim.nD_drop_time / param.sim.dTi)) + 1,
                 param.sim.Nsim)
    t = t_full[:N_drop]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create an empty line for each dropped agent
    lines = []
    for iD in dropped_ids:
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

    # Automatically scale y-axis using only the plotted data
    ymin = min(np.min(data.radial_history[iD][:N_drop]) for iD in dropped_ids)
    ymax = max(np.max(data.radial_history[iD][:N_drop]) for iD in dropped_ids)
    ymin = min(ymin, param.sim.sphere_radius)
    ymax = max(ymax, param.sim.sphere_radius)
    margin = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Distance to target [m]", fontsize=16)
    ax.grid(True)
    ax.legend()

    # Frames spread over the truncated window
    n_frames = max(int(param.sim.nD_drop_time * param.display.mp4_fps), 2)
    frame_indices = np.linspace(0, N_drop - 1, n_frames, dtype=int)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        idx = frame_indices[:frame + 1]
        for iD, line in zip(dropped_ids, lines):
            line.set_data(
                t[idx],
                np.asarray(data.radial_history[iD])[idx],
            )
        return lines

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000 / param.display.mp4_fps,  # milliseconds
        blit=True,
        repeat=False,
    )

    plt.tight_layout()

    # Save animation (requires ffmpeg)
    if param.display.save_to_mp4:
        print("Saving radial_history animation")
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ani.save(
            os.path.join(
                current_folder + param.display.save_location,
                f"radial_hist_{len(dropped_ids)}_dropped_agents.mp4",
            ),
            writer="ffmpeg",
            fps=param.display.mp4_fps
        )

    plt.show()


def inter_agent_dist_plot(data: SimulationData, param: Param):
    """
    Plot inter-agent Euclidean distance for every unique pair of agents,
    plus the maximum, mean and minimum across pairs.

    Pairs involving a dropped agent (i.e. not in data.active_ids) are only
    plotted up to param.sim.nD_drop_time. After that, the summary curves
    are computed over the remaining (active-active) pairs only.

    Parameters
    ----------
    data.position_history : ndarray
        Shape (nD, >=3, Nsim)
    data.active_ids : iterable
        Ids of agents that survived the entire simulation
    param : Param
        Uses param.sim.{Nsim, dTi, nD, nD_drop_time} and
        param.display.save_location
    """

    plt.rcParams.update({
        "font.size": 14,
        "axes.linewidth": 1.5,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

    Nsim = param.sim.Nsim
    nD = param.sim.nD
    dTi = param.sim.dTi
    t = np.arange(Nsim) * dTi

    # Agents that were dropped at the drop time
    active = set(int(i) for i in data.active_ids)
    dropped = set(range(nD)) - active
    N_drop = param.sim.nD_drop_time   # same convention as radial_hist_plot

    # Positions: (nD, 3, Nsim)
    pos = np.stack([data.position_history[iD, :3, :Nsim] for iD in range(nD)])

    # All unique pairs (equivalent of nchoosek(1:nD, 2))
    pairs = list(itertools.combinations(range(nD), 2))
    dist_pairs = np.array(
        [np.linalg.norm(pos[i] - pos[j], axis=0) for i, j in pairs]
    )  # (nP, Nsim)

    # Mask out pairs involving a dropped agent after the drop time
    for k, (i, j) in enumerate(pairs):
        if i in dropped or j in dropped:
            dist_pairs[k, N_drop:] = np.nan

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100, facecolor="w")
    ax.set_facecolor("w")

    # Every pair: thin light grey, single legend entry
    grey = (0.7, 0.7, 0.7)
    for k, d in enumerate(dist_pairs):
        ax.plot(
            t, d, color=grey, linewidth=1,
            label="Individual pairs" if k == 0 else "_nolegend_",
        )

    # Summary statistics over the pairs that still exist at each timestep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        d_max = np.nanmax(dist_pairs, axis=0)
        d_mean = np.nanmean(dist_pairs, axis=0)
        d_min = np.nanmin(dist_pairs, axis=0)

    ax.plot(t, d_max,  "r-",  linewidth=2.5, label="Maximum")
    ax.plot(t, d_mean, "k--", linewidth=3,   label="Mean")
    ax.plot(t, d_min,  "b-",  linewidth=2.5, label="Minimum")

    # Optional safety distance line
    if hasattr(param.sim, "obstacle_avoidance_dim"):
        ax.axhline(
            param.sim.obstacle_avoidance_dim,
            color="0.3", linestyle=":", linewidth=2,
            label="Minimum safety distance",
        )

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel(r"$\|\mathbf{p}_i - \mathbf{p}_j\|$ [m]", fontsize=20)

    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(width=1.5, labelsize=14)

    ax.legend(loc="best", fontsize=12)
    fig.tight_layout()

    print("saving inter agent distance plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"inter_agent_dist_{param.sim.nD}_agents.pdf",
        ),
        bbox_inches="tight",
    )

    plt.show(block=False)

def low_level_tracking_error_plot(data: SimulationData, param: Param):
    """
    Plot the tracking error between each agent's desired and actual position.

    For each agent, plots the Euclidean distance between its current position
    and its desired position over time. Agents that were dropped
    (i.e. not in data.active_ids) are only plotted from t=0 up to
    param.sim.nD_drop_time.

    Parameters
    ----------
    data.position_history : ndarray
        Agent position histories (nD x 3 x Nsim)
    data.p_d_hist : ndarray
        Desired position histories of each agent (nD x 3 x Nsim)
    data.active_ids : iterable
        Ids of agents that survived the entire simulation
    param : Param
        Uses param.sim.{Nsim, dTi, nD, nD_drop_time} and
        param.display.save_location
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    Nsim = param.sim.Nsim
    nD = param.sim.nD
    t = np.arange(Nsim) * param.sim.dTi

    # Agents that survived vs. agents that were dropped
    active = set(int(i) for i in data.active_ids)
    N_drop = param.sim.nD_drop_time   # same convention as radial_hist_plot

    # Positions and desired positions: (nD, 3, Nsim)
    pos = np.stack([data.position_history[iD, :, :Nsim] for iD in range(nD)])
    p_d = np.stack([data.p_d_hist[iD, :, :Nsim] for iD in range(nD)])

    # Tracking error per agent (own position vs own desired position): (nD, Nsim)
    tracking_error = np.linalg.norm(pos - p_d, axis=1)

    for iD in range(nD):
        if iD in active:
            t_plot = t
            err_plot = tracking_error[iD]
        else:
            # Dropped agent: only plot up to the drop time
            t_plot = t[:N_drop]
            err_plot = tracking_error[iD, :N_drop]

        ax.plot(
            t_plot,
            err_plot,
            linewidth=2,
            color=data.colors[iD],
        )

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Tracking error distance [m]", fontsize=20)

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    print("saving low level tracking error plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"low_level_tracking_error_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )

    plt.show(block=False)

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

    ax.set_xlabel("X [M]", fontsize=16)
    ax.set_ylabel("Y [M]", fontsize=16)
    ax.set_zlabel("Z [M]", fontsize=16)

    ax.legend()

    print("saving consensus plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"consensus_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )

    plt.show(block=False)

def velocity_plot(data: SimulationData, param: Param):
    """
    Plot the velocity history of all active agents.
    Shows vx, vy, vz as separate subplots, with the v_min / v_max limits.
    Dropped agents are not plotted.
    """

    nD = param.sim.nD

    # Agents that were dropped at the drop time
    active = set(int(i) for i in data.active_ids)
    dropped = set(range(nD)) - active
    N_drop = param.sim.nD_drop_time   # same convention as radial_hist_plot

    t_dropped = np.arange(N_drop) * param.sim.dTi
    t = np.arange(param.sim.Nsim) * param.sim.dTi

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    labels = ["$v_x [m/s]$", "$v_y [m/s]$", "$v_z [m/s]$"]

    vmin = -param.sim.v_max
    vmax = param.sim.v_max
    margin = 0.1 * max(abs(vmin), abs(vmax), 1.0)

    for axis, (ax, label) in enumerate(zip(axes, labels)):
        for iD in active:
            ax.plot(
                t,
                data.velocity_history[iD, axis, :],
                color=data.colors[iD],
                lw=2,
            )

        for iD in dropped:
            ax.plot(
                t_dropped,
                data.velocity_history[iD, axis, :N_drop],
                color=data.colors[iD],
                lw=2,
            )

        # Min / max velocity limit lines
        ax.axhline(
            vmax,
            color="k",
            ls="--",
            lw=1.5,
            alpha=0.8,
            label="$v_{max}$" if axis == 0 else None,
        )
        ax.axhline(
            vmin,
            color="k",
            ls="--",
            lw=1.5,
            alpha=0.8,
            label="$v_{min}$" if axis == 0 else None,
        )

        ax.set_ylabel(label, fontsize=20)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")

    print("saving velocity plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))

    plt.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"velocity_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight",
    )

    plt.show(block=False)


def velocity_animate(data: SimulationData, param: Param):
    """
    Animate the velocity history of all agents.
    Shows vx, vy, vz as separate curves that grow with time.
    """

    frame_indices = np.linspace(
        0,
        param.sim.Nsim - 1,
        int(param.sim.Tend * param.display.mp4_fps),
        dtype=int,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    labels = ["$v_x$", "$v_y$", "$v_z$"]

    vel_lines = []

    for ax, label in zip(axes, labels):
        ax.set_ylabel(label, fontsize=16)
        ax.grid(True)

        lines = []

        for iD in range(param.sim.nD):
            line, = ax.plot(
                [],
                [],
                color=data.colors[iD],
                lw=2,
                label=f"Drone {iD+1}",
            )
            lines.append(line)

        vel_lines.append(lines)

    
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend()

    # Fixed limits
    vmin = - param.sim.v_max
    vmax = param.sim.v_max
    margin = 0.1 * max(abs(vmin), abs(vmax), 1.0)

    for ax in axes:
        ax.set_xlim(0, param.sim.Nsim * param.sim.dTi - param.sim.dTi)
        ax.set_ylim(vmin - margin, vmax + margin)

        # Min / max velocity limit lines
        ax.axhline(vmax, color="k", ls="--", lw=1.5, alpha=0.8)
        ax.axhline(vmin, color="k", ls="--", lw=1.5, alpha=0.8)

    def update(frame):
        print(f"current frame: {frame}")

        idx = frame_indices[:frame + 1]
        t = idx * param.sim.dTi

        for axis in range(3):
            for iD in range(param.sim.nD):
                vel_lines[axis][iD].set_data(
                    t,
                    data.velocity_history[iD, axis, idx],
                )

        artists = []
        for lines in vel_lines:
            artists.extend(lines)

        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / param.display.mp4_fps,
        blit=False,
        repeat=True,
    )

    plt.show()

    if param.display.save_to_mp4:
        print("Saving velocity animation")
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ani.save(
            os.path.join(
                current_folder + param.display.save_location,
                f"velocity_{param.sim.nD}_agents.mp4",
            ),
            writer="ffmpeg",
            fps=param.display.mp4_fps,
        )

def acceleration_plot(data: SimulationData, param: Param):
    """
    Plot the acceleration history of all agents.
    Shows ax, ay, az as separate subplots, with the a_min / a_max limits.
    """

    t = np.arange(param.sim.Nsim) * param.sim.dTi

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    labels = ["$a_x$", "$a_y$", "$a_z$"]

    amin = -param.sim.a_max *0.2
    amax = param.sim.a_max *0.2
    margin = 0.1 * max(abs(amin), abs(amax), 1.0)

    for axis, (ax, label) in enumerate(zip(axes, labels)):
        for iD in range(param.sim.nD):
            ax.plot(
                t,
                data.acceleration_history[iD, axis, :],
                color=data.colors[iD],
                lw=2,
                label=f"Drone {iD+1}",
            )

        # Min / max velocity limit lines
        ax.axhline(
            amax, color="k", ls="--", lw=1.5, alpha=0.8,
            label="$a_{max}$" if axis == 0 else None,
        )
        ax.axhline(
            amin, color="k", ls="--", lw=1.5, alpha=0.8,
            label="$a_{min}$" if axis == 0 else None,
        )

        ax.set_ylabel(label, fontsize=16)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(amin - margin, amax + margin)
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")

    fig.tight_layout()

    print("saving acceleration plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"acceleration_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )
    plt.show(block=False)

def trajectory_plot(data, param):
    """
    Static 3D plot of all agent trajectories and the target trajectory.
    """

    # -----------------------------
    # Compute axis limits
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

    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_zlim(all_z.min() - margin, all_z.max() + margin)

    # -----------------------------
    # Plot target
    # -----------------------------
    ax.plot(
        data.target_true_hist[0],
        data.target_true_hist[1],
        data.target_true_hist[2],
        "k",
        lw=3,
        label="Target"
    )

    ax.plot(
        data.target_true_hist[0, 0],
        data.target_true_hist[1, 0],
        data.target_true_hist[2, 0],
        "ko",
        markersize=8,
        label="Target Start"
    )

    ax.plot(
        data.target_true_hist[0, -1],
        data.target_true_hist[1, -1],
        data.target_true_hist[2, -1],
        "ks",
        markersize=8,
        label="Target End"
    )

    # -----------------------------
    # Plot drones
    # -----------------------------
    for iD in range(param.sim.nD):

        pos = data.position_history[iD]

        # Trajectory
        ax.plot(
            pos[0],
            pos[1],
            pos[2],
            color=data.colors[iD],
            lw=2,
            label=f"Drone {iD+1}"
        )

        # Initial position
        ax.scatter(
            pos[0, 0],
            pos[1, 0],
            pos[2, 0],
            marker="x",
            s=60,
            color=data.colors[iD]
        )

        # Final position
        ax.scatter(
            pos[0, -1],
            pos[1, -1],
            pos[2, -1],
            marker="o",
            s=40,
            color=data.colors[iD]
        )

        # Desired final position (optional)
        ax.scatter(
            data.p_d_hist[iD][0, -1],
            data.p_d_hist[iD][1, -1],
            data.p_d_hist[iD][2, -1],
            marker="D",
            s=40,
            color=data.colors[iD]
        )

    ax.set_xlabel("X [m]", fontsize=16)
    ax.set_ylabel("Y [m]", fontsize=16)
    ax.set_zlabel("Z [m]", fontsize=16)
    ax.legend()
    ax.view_init(elev=30, azim=120)
    plt.tight_layout()

    print("saving trajectory plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"trajectory_{param.sim.nD}_agents.pdf"
        ),
        bbox_inches="tight"
    )
    plt.show(block=False)

def trajectory_animate(data: SimulationData, param: Param):
    """
    FuncAnimation of all agents and target trajectory
    """

    frame_indices = np.linspace(
        0,
        param.sim.Nsim - 1,
        int(param.sim.Tend * param.display.mp4_fps),
        dtype=int,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

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

    def update(frame):
        print(f"current frame: {frame}")
        # Current data
        idx = frame_indices[:frame + 1]
        start = max(0, idx[-1] - param.display.trail_length)
        idx = idx[(idx >= start)]

        # update frame
        all_x = np.concatenate([
            data.p_d_hist[:, 0, idx[-1]], data.position_history[:, 0, idx[-1]],
            [data.target_true_hist[0, idx[-1]]]
        ])
        all_x = all_x[~np.isnan(all_x)]

        all_y = np.concatenate([
            data.p_d_hist[:, 1, idx[-1]], data.position_history[:, 1, idx[-1]],
            [data.target_true_hist[1, idx[-1]]]
        ])
        all_y = all_y[~np.isnan(all_y)]

        all_z = np.concatenate([
            data.p_d_hist[:, 2, idx[-1]], data.position_history[:, 2, idx[-1]],
            [data.target_true_hist[2, idx[-1]]]
        ])
        all_z = all_z[~np.isnan(all_z)]

        margin = 2

        xmin, xmax = np.min(all_x) - margin, np.max(all_x) + margin
        ymin, ymax = np.min(all_y) - margin, np.max(all_y) + margin
        zmin, zmax = np.min(all_z) - margin, np.max(all_z) + margin

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

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

        for iD in range(param.sim.nD):
            pos = data.position_history[iD]

            # Initial position
            initial_pts[iD].set_visible(start == 0)

            # Trajectory
            traj_lines[iD].set_data(
                pos[0, idx],
                pos[1, idx]
            )
            traj_lines[iD].set_3d_properties(
                pos[2, idx]
            )

            # Drone position
            drone_pts[iD].set_data(
                [pos[0, idx[-1]]],
                [pos[1, idx[-1]]]
            )
            drone_pts[iD].set_3d_properties(
                [pos[2, idx[-1]]]
            )

            # Desired position
            if not np.isnan(data.p_d_hist[iD][0, idx[-1]]):
                desired_pts[iD].set_data(
                    [data.p_d_hist[iD][0, idx[-1]]],
                    [data.p_d_hist[iD][1, idx[-1]]]
                )
                desired_pts[iD].set_3d_properties(
                    [data.p_d_hist[iD][2, idx[-1]]]
                )
            else:
                desired_pts[iD].set_data([], [])
                desired_pts[iD].set_3d_properties([])   

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
        repeat=True
    )
    
    plt.show()

    # Save animation (requires ffmpeg)
    if param.display.save_to_mp4:
        print("Saving trajectory animation")
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ani.save(
            os.path.join(
                current_folder + param.display.save_location,
                f"trajectory_{param.sim.nD}_agents.mp4",
            ),
            writer="ffmpeg",
            fps=param.display.mp4_fps,
        )


if __name__ == "__main__":
    param = Param()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # flight_sims/ (root_results)
    import root_results
    root_results.mirror_on_exit("V16_generic_mesh", str(Path(__file__).resolve().parent) + param.display.save_location)
    param.display.save_to_mp4 = False # we are assuming that the user as 
    # previously saved the plots
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "simulation_results.pkl")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    sim_data = SimulationData(**data)

    os.makedirs(
        os.path.join(
            current_folder + param.display.save_location,
            f"trajectory_{param.sim.nD}_agents.mp4",
        ),
        exist_ok=True
    )

    create_figures(sim_data, param)
