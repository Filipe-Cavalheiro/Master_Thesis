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

def formation_phases(data: SimulationData, param: Param, min_duration=0.5):
    """
    Formation type of the mesh over time ("wall", "cone", "sphere", "red border",
    voronoi_mesh_funcs.formation_type) from the mean target estimate of the
    active chasers, as a list of (start time [s], formation). Phases shorter
    than min_duration (estimate noise at a threshold) are merged into the
    previous one.
    """
    from voronoi_mesh_funcs import formation_type
    est = np.asarray(data.perceived_target_hist)[:, :3, :param.sim.Nsim]       # (nD, 3, N)
    alive = ~np.isnan(np.asarray(data.p_d_hist)[:, :, :param.sim.Nsim]).any(axis=1)   # (nD, N)
    est = np.where(alive[:, None, :], est, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        target = np.nanmean(est, axis=0)                                         # (3, N)
    t = np.arange(target.shape[1]) * param.sim.dTi
    kinds = [formation_type(target[:, k], param.airspace, param.sim.sphere_radius) for k in range(len(t))]

    phases = []
    for k, kind in enumerate(kinds):
        if not phases or kind != phases[-1][1]:
            phases.append((t[k], kind))
    merged = []
    for i, (t0, kind) in enumerate(phases):
        t1 = phases[i + 1][0] if i + 1 < len(phases) else t[-1]
        if merged and (t1 - t0 < min_duration or kind == merged[-1][1]):
            continue
        merged.append((t0, kind))
    # drop repeated kinds left by the merge
    return [p for i, p in enumerate(merged) if i == 0 or p[1] != merged[i - 1][1]]


def mark_formation_phases(axes, phases, label_ax=None, fontsize=14):
    """
    Grey dashed vertical line at every change of formation, with the name of
    the new formation (as Figure 6.13 marks the wall, the red border and the
    loss); the first formation is written at the start of the plot.
    """
    import matplotlib.transforms as mtransforms
    axes = np.atleast_1d(axes)
    label_ax = axes[0] if label_ax is None else label_ax
    trans = mtransforms.blended_transform_factory(label_ax.transData, label_ax.transAxes)
    x0, x1 = label_ax.get_xlim()
    y_label, last_t = 0.98, None
    for i, (t0, kind) in enumerate(phases):
        if i > 0:
            for ax in axes:
                ax.axvline(t0, color="0.45", ls="--", lw=1.2, zorder=0.5)
        # a label too close to the previous one goes one line lower
        close = last_t is not None and (t0 - last_t) < 0.08 * (x1 - x0)
        y_label = (0.90 if y_label == 0.98 else 0.98) if close else 0.98
        label_ax.text(t0, y_label, " " + kind, transform=trans, color="0.35", fontsize=fontsize,
                      ha="left", va="top", clip_on=False)
        last_t = t0


def create_figures(data: SimulationData, param: Param):
    voronoi = Voronoi(param)
    origin_p_d, swarm_center, voronoi = pre_compute_desired_pos(voronoi, param)
    # references to the animations: with non-blocking show they would be
    # garbage collected before rendering anything
    animations = []
    if param.display.composed is True:
        composed_plot(data, voronoi, param)

    if param.display.detailed_close_up_bool is True:
        detailed_close_up(data, voronoi, param)

    if param.display.three_view_plot is True:
        three_view_plot(data, param)

    # Radial History
    if param.display.radial_hist_plot is True:
        radial_hist_plot(data, param)

    if param.display.radial_hist_animation is True:
        animations.append(radial_hist_animation(data, param))

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
        animations.append(velocity_animate(data, param))

    # Acceleration
    if param.display.acceleration_plot is True:
        acceleration_plot(data, param)

    if param.display.acceleration_animate is True:
        animations.append(acceleration_animate(data, param))

    # Trajectory
    if param.display.trajectory_plot is True:
        trajectory_plot(data, param)

    if param.display.trajectory_overview_plot is True:
        trajectory_overview_plot(data, param)

    if param.display.trajectory_animate is True:
        animations.append(trajectory_animate(data, param))

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
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(True)
    ax.view_init(elev=None, azim=None)
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_zlabel("Z [m]", fontsize=12)

    target_hist = data.target_true_hist
    position_history = data.position_history

    # Remove steps above simulation time
    steps = np.array(getattr(param.display, "close_up_steps", [100, 250, 400, 1000, 1300, 1900]))
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

    drawn_pts = []      # meshes and targets: the axis limits (add_collection3d does not autoscale)
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
        drawn_pts.append(voronoi.vertices)
        drawn_pts.append(target_pos.reshape(1, 3))

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

    # Make X, Y, Z use the same scale, around the drawn meshes and targets
    pts = np.vstack(drawn_pts)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    half_range = np.ptp(pts, axis=0).max() / 2 + 1.0

    ax.set_xlim3d(center[0] - half_range, center[0] + half_range)
    ax.set_ylim3d(center[1] - half_range, center[1] + half_range)
    ax.set_zlim3d(center[2] - half_range, center[2] + half_range)
    # the wall (dashed) and the zone borders on the floor of the plot
    _draw_airspace(ax, param, z_offset=center[2] - half_range)
    ax.set_xlim3d(center[0] - half_range, center[0] + half_range)
    ax.set_ylim3d(center[1] - half_range, center[1] + half_range)
    ax.legend(fontsize=10, loc='upper right')
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
    steps = np.array(getattr(param.display, "close_up_steps", [100, 250, 400, 1000, 1300, 1900]))
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

def _airspace_fields(param: Param, xlim, ylim, n):
    """Grid over xlim/ylim with the red signed distance and the orange mask."""
    airspace = param.airspace
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    P = np.column_stack([X.ravel(), Y.ravel()])
    sd_red = airspace.red.signed_distance(P).reshape(X.shape)
    orange = np.zeros(len(P), dtype=bool)
    for poly in airspace.orange:
        orange |= poly.contains(P)
    return X, Y, sd_red, orange.reshape(X.shape)


def _draw_airspace_floor(ax, param: Param, n=150):
    """
    Paint the restricted airspace on the z = 0 floor of a 3D axes over its
    current x/y limits: orange zone floor (outside the wall), grey zone floor
    (the band between the deterrent wall and the red border, where the
    drones fly), red zone floor, the red border and the wall line
    (wall_offset outside the red border).
    Returns the created artists so they can be removed on the next frame.
    """
    airspace = param.airspace
    if not airspace.enabled:
        return []
    X, Y, sd_red, orange = _airspace_fields(param, ax.get_xlim(), ax.get_ylim(), n)
    grey = (sd_red >= 0) & (sd_red <= airspace.wall_offset)
    zone = np.where(sd_red < 0, 3.0, np.where(grey, 2.0, np.where(orange, 1.0, 0.0)))
    kw = {"zdir": "z", "offset": 0.0}
    artists = []
    if zone.max() > 0:
        artists.append(ax.contourf(X, Y, zone, levels=[0.5, 1.5, 2.5, 3.5],
                                   colors=["orange", "grey", "red"], alpha=0.35, **kw))
    for level, style in [(0.0, "solid"), (airspace.wall_offset, "dashed")]:
        if sd_red.min() < level < sd_red.max():
            artists.append(ax.contour(X, Y, sd_red, levels=[level], colors="darkred",
                                      linestyles=style, linewidths=2, **kw))
    return artists


def _draw_airspace(ax, param: Param, z_offset=None, n=300):
    """
    Contour the restricted airspace over the current x/y limits: red border,
    deterrent wall (wall_offset outside red) and orange zone border.
    For 3D axes the contours are drawn on the plane z = z_offset.
    """
    airspace = param.airspace
    if not airspace.enabled:
        return
    X, Y, sd_red, orange = _airspace_fields(param, ax.get_xlim(), ax.get_ylim(), n)
    orange = orange.astype(float)

    kw = {} if z_offset is None else {"zdir": "z", "offset": z_offset}
    for field, level, color, style in [
        (orange, 0.5, "darkorange", "solid"),
        (sd_red, 0.0, "red", "solid"),          # on top of the shared orange edges
        (sd_red, airspace.wall_offset, "red", "dashed"),
    ]:
        if field.min() < level < field.max():
            ax.contour(X, Y, field, levels=[level], colors=color, linestyles=style, linewidths=1.5, **kw)
    ax.plot([], [], color="red", label="Red zone border")
    ax.plot([], [], color="red", ls="--", label=f"Wall ({airspace.wall_offset:.0f} m)")
    ax.plot([], [], color="darkorange", label="Orange zone border")


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
    _draw_airspace(ax_top, param)

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
    mark_formation_phases(ax, formation_phases(data, param))

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

    plt.show(block=False)

    return ani


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
    mark_formation_phases(ax, formation_phases(data, param))

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
    mark_formation_phases(ax, formation_phases(data, param))

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
    mark_formation_phases(axes, formation_phases(data, param), fontsize=17)

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

    plt.show(block=False)

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

    return ani

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

def acceleration_animate(data: SimulationData, param: Param):
    """
    Animate the acceleration history of all agents.
    Shows ax, ay, az as separate curves that grow with time.
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

    labels = ["$a_x$", "$a_y$", "$a_z$"]

    acc_lines = []

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

        acc_lines.append(lines)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")

    # Fixed limits (same as acceleration_plot)
    amin = -param.sim.a_max * 0.2
    amax = param.sim.a_max * 0.2
    margin = 0.1 * max(abs(amin), abs(amax), 1.0)

    for ax in axes:
        ax.set_xlim(0, param.sim.Nsim * param.sim.dTi - param.sim.dTi)
        ax.set_ylim(amin - margin, amax + margin)

        # Min / max acceleration limit lines
        ax.axhline(amax, color="k", ls="--", lw=1.5, alpha=0.8)
        ax.axhline(amin, color="k", ls="--", lw=1.5, alpha=0.8)

    def update(frame):
        print(f"current frame: {frame}")

        idx = frame_indices[:frame + 1]
        t = idx * param.sim.dTi

        for axis in range(3):
            for iD in range(param.sim.nD):
                acc_lines[axis][iD].set_data(
                    t,
                    data.acceleration_history[iD, axis, idx],
                )

        artists = []
        for lines in acc_lines:
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

    plt.show(block=False)

    if param.display.save_to_mp4:
        print("Saving acceleration animation")
        current_folder = os.path.dirname(os.path.abspath(__file__))
        ani.save(
            os.path.join(
                current_folder + param.display.save_location,
                f"acceleration_{param.sim.nD}_agents.mp4",
            ),
            writer="ffmpeg",
            fps=param.display.mp4_fps,
        )

    return ani

def _draw_trajectory_3d(ax, data: SimulationData, param: Param):
    """
    Draw all agent trajectories, the target trajectory and the restricted
    airspace (on the floor of the plot) on a 3D axes.
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

    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_zlim(all_z.min() - margin, all_z.max() + margin)
    _draw_airspace(ax, param, z_offset=all_z.min() - margin)

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
    ax.view_init(elev=30, azim=120)


def trajectory_plot(data, param):
    """
    Static 3D plot of all agent trajectories and the target trajectory.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_trajectory_3d(ax, data, param)
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=10)
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


def trajectory_overview_plot(data: SimulationData, param: Param):
    """
    Two panels:
      left : top-down view of every KMZ zone around the Lisbon airport, with
             a square marking where the simulation takes place;
      right: the 3D trajectory plot of that location.
    """
    from matplotlib.patches import Polygon as PolygonPatch, Rectangle, Patch
    from matplotlib.lines import Line2D

    airspace = param.airspace
    if airspace.kind != "lisbon":
        print("trajectory overview plot needs the Lisbon scene (scene = \"lisbon\"), skipped")
        return
    fig = plt.figure(figsize=(18, 8))
    ax_map = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    # -----------------------------
    # Left: global top-down view
    # -----------------------------
    handles = []
    for name, polys in airspace.zones:
        if name.endswith(airspace.red_zone_name):
            color, order = "red", 2
        elif name.endswith(airspace.orange_zone_name):
            color, order = "orange", 1
        else:
            color, order = "grey", 1
        for poly in polys:
            ax_map.add_patch(PolygonPatch(poly / 1000.0, closed=True, facecolor=color,
                                          edgecolor=color, alpha=0.35, lw=1.5, zorder=order))
        handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.35, label=name))

    # airport reference point (frame origin)
    ax_map.plot(0, 0, marker="*", color="k", markersize=14, linestyle="", zorder=4)
    handles.append(Line2D([], [], marker="*", color="k", markersize=12, linestyle="",
                          label="LPPT airport reference point (origin)"))

    # target path and the "action" square
    T = data.target_true_hist[:2] / 1000.0
    P = data.position_history[:, :2, :].transpose(1, 0, 2).reshape(2, -1) / 1000.0
    pts = np.hstack([T, P])
    lo, hi = pts.min(axis=1), pts.max(axis=1)
    centre = (lo + hi) / 2
    half = max(np.max(hi - lo) / 2 * 1.2, 0.25)          # at least 500 m wide [km]
    ax_map.plot(T[0], T[1], "k", lw=1.5, zorder=4)
    ax_map.add_patch(Rectangle(centre - half, 2 * half, 2 * half, fill=False,
                               edgecolor="blue", lw=2, zorder=5))
    handles.append(Line2D([], [], color="blue", lw=2, label="Simulation area (right)"))

    all_poly = np.vstack([p for _, polys in airspace.zones for p in polys]) / 1000.0
    pad = 0.5
    ax_map.set_xlim(all_poly[:, 0].min() - pad, all_poly[:, 0].max() + pad)
    ax_map.set_ylim(all_poly[:, 1].min() - pad, all_poly[:, 1].max() + pad)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("East [km]", fontsize=14)
    ax_map.set_ylabel("North [km]", fontsize=14)
    ax_map.grid(True, alpha=0.4)
    ax_map.legend(handles=handles, fontsize=9, loc="upper left")
    ax_map.set_title("Lisbon airport restricted airspace", fontsize=14)

    # -----------------------------
    # Right: 3D trajectories
    # -----------------------------
    _draw_trajectory_3d(ax3d, data, param)
    ax3d.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    ax3d.set_title("Simulation area", fontsize=14, color="blue")

    plt.tight_layout()

    print("saving trajectory overview plot")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        os.path.join(
            current_folder + param.display.save_location,
            f"trajectory_overview_{param.sim.nD}_agents.pdf"
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

    # restricted airspace painted on the z = 0 floor (redrawn every frame)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    if param.airspace.enabled:
        ax.legend(handles=[
            Patch(color="red", alpha=0.35, label="Red zone"),
            Patch(color="grey", alpha=0.35, label="Grey zone (drones fly)"),
            Patch(color="orange", alpha=0.35, label="Orange zone"),
            Line2D([], [], color="darkred", lw=2, label="Red zone border"),
            Line2D([], [], color="darkred", lw=2, ls="--",
                   label=f"Wall ({param.airspace.wall_offset:.0f} m from border)"),
        ], loc="upper right", fontsize=9)
    floor_artists = []

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
        zmin = min(zmin, 0.0)   # keep the airspace floor (z = 0) in view

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        for a in floor_artists:
            a.remove()
        floor_artists[:] = _draw_airspace_floor(ax, param)

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
    
    plt.show(block=False)

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

    return ani


if __name__ == "__main__":
    param = Param()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # flight_sims/ (root_results)
    import root_results
    root_results.mirror_on_exit("swarm_MPC_voronoi", Path(__file__).resolve().parent / "Results")
    param.display.results_file = None
    current_folder = os.path.dirname(os.path.abspath(__file__))
    from Structs_sim import results_file_path, param_from_config, simulation_data
    file_path = results_file_path(param.display, latest=True)   # None: the newest .pkl of this simulation
    print(f"Results: {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    if "sim" in data:
        # the run's own scene, trajectory and settings (older .pkl: those of Structs_sim.py now)
        param = param_from_config(data["sim"])
        print("Run: " + ", ".join(f"{k} {v}" for k, v in data["sim"].items()))
    if "flight" in data:
        print("Simulation of the real flight " + data["flight"].get("log", ""))

    sim_data = simulation_data(data)

    os.makedirs(
        current_folder + param.display.save_location,
        exist_ok=True
    )

    create_figures(sim_data, param)
