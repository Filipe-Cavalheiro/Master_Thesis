import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import pickle

from Structs import Param, SimulationData

def create_figures(data: SimulationData, param: Param):
    if param.display.composed is True:
        composed_plot(data, param)

    if param.display.radial_hist_plot is True:
        radial_hist_plot(data, param)

    if param.display.radial_hist_animation is True:
        radial_hist_animation(data, param)

    if param.display.consensus_plot is True:
        consensus_plot(data, param)

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

def composed_plot(data: SimulationData, param: Param):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=None, azim=None)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

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

    t = np.arange(param.sim.Nsim) * param.sim.dTi

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
    ax.set_title("Radial Distance History")

    ax.grid(True)
    ax.legend()

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

    plt.show(block=False)

def radial_hist_animation(data: SimulationData, param: Param):
    """
    Animate radial distance history between all chasers and the target.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    t = np.arange(param.sim.Nsim) * param.sim.dTi

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
    ax.set_title("Radial Distance History")
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

    ax.set_title(
        "Consensus Tracking of Moving Target "
        "(EMA + Noisy Measurements)"
    )

    ax.legend()

    if not os.path.isdir(param.display.save_location):
        os.makedirs(param.display.save_location)

    plt.savefig(
        f"{param.display.save_location}/consensus_{param.sim.nD}_agents.pdf",
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

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Trajectories ({param.sim.nD} Agents)")
    ax.legend()

    plt.tight_layout()
    plt.show()

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

    def update(frame):
        print(f"Showing frame: {frame}")
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
            desired_pts[iD].set_data(
                [data.p_d_hist[iD][0, idx[-1]]],
                [data.p_d_hist[iD][1, idx[-1]]]
            )
            desired_pts[iD].set_3d_properties(
                [data.p_d_hist[iD][2, idx[-1]]]
            )

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
    param = Param()
    with open("CVT_weighted_sites/simulation_results.pkl", "rb") as f:
        data = pickle.load(f)

    sim_data = SimulationData(**data)
    create_figures(sim_data, param)