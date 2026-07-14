import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from Structs import Param

# ============================================================
# Plot
# ============================================================
def visualise(results, param:Param):
    executed_path = results["executed_path"]
    velocity_history = results["velocity_history"]
    acc_history = results["acc_history"]
    horizon_history = results["horizon_history"]
    start = results["start"]
    goal = results["goal"]
    #d_history = results["d_history"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
  
    colors = plt.cm.tab10(np.linspace(0, 1,     ))

    for iD in range(param.sim.nD):
        ax.plot(
            executed_path[iD][:,0],
            executed_path[iD][:,1],
            executed_path[iD][:,2],
            linewidth=3,
            color=colors[iD],
            label=f"Drone {iD+1}"
        )

        ax.scatter(
            start[0,iD],
            start[1,iD],
            start[2,iD],
            marker='o',
            color='green',
            s=80
        )

        ax.scatter(
            goal[0,iD],
            goal[1,iD],
            goal[2,iD],
            marker='*',
            color='orange',
            s=120
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.axis('equal')
    ax.legend()
    plt.show(block=False)

    
    # # ============================================================
    # # d plot
    # # ============================================================    
    # plt.figure(figsize=(8,4))

    # time = np.arange(len(d_history[0])) * param.sim.dTi

    # for iD in range(param.sim.nD):
    #     plt.plot(time, d_history[iD], label=f"Drone {iD}")

    # plt.axhline(1.0, color='r', linestyle='--', label='Minimum safe distance')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Normalized distance $d$")
    # plt.title("Obstacle-avoidance distance over time")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ============================================================
    # Velocity plot
    # ============================================================

    time = np.arange(len(velocity_history[0])) * param.sim.dTi

    plt.figure(figsize=(8,5))
    for iD in range(param.sim.nD):
        plt.plot(time, velocity_history[iD][:,0], label=f"Drone {iD+1} vx")
        plt.plot(time, velocity_history[iD][:,1], '--', label=f"Drone {iD+1} vy")
        plt.plot(time, velocity_history[iD][:,2], ':', label=f"Drone {iD+1} vz")

    plt.axhline(param.v_max, color='k', linestyle='--', alpha=0.6, label='v_max')
    plt.axhline(-param.v_max, color='k', linestyle='--', alpha=0.6)

    plt.xlabel("Normalized time")
    plt.ylabel("Velocity")
    plt.title("Velocity Profile")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    # ============================================================
    # Acceleration plot
    # ============================================================
    plt.figure(figsize=(8,5))
    for iD in range(param.sim.nD):
        plt.plot(time, acc_history[iD][:,0], label=f"Drone {iD+1} ax")
        plt.plot(time, acc_history[iD][:,1], '--', label=f"Drone {iD+1} ay")
        plt.plot(time, acc_history[iD][:,2], ':', label=f"Drone {iD+1} az")

    plt.axhline(param.a_max, color='k', linestyle='--', alpha=0.6, label='a_max')
    plt.axhline(-param.a_max, color='k', linestyle='--', alpha=0.6)

    plt.xlabel("Normalized time")
    plt.ylabel("Acceleration")
    plt.title("Acceleration Profile")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    # ============================================================
    # Inter-agent distance plot
    # ============================================================

    plt.figure(figsize=(8,5))

    n_steps = executed_path.shape[1]
    time = np.arange(n_steps) * param.sim.dTi

    for i in range(param.sim.nD):
        for j in range(i + 1, param.sim.nD):

            dist = np.linalg.norm(
                executed_path[i] - executed_path[j],
                axis=1
            )

            plt.plot(
                time,
                dist,
                linewidth=2,
                label=f"Drone {i+1} - Drone {j+1}"
            )

    plt.axhline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Minimum separation"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Inter-agent distance [m]")
    plt.title("Inter-agent Distance")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    # ============================================================
    # Animation plot
    # ============================================================

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    path_lines = []
    drones = []
    horizon_lines = []

    for iD in range(param.sim.nD):

        line, = ax.plot([], [], [], lw=2,
                        color=colors[iD],
                        label=f"Drone {iD+1}")

        drone, = ax.plot([], [], [], "o",
                        color=colors[iD],
                        markersize=8)
        
        horizon, = ax.plot(
            [],
            [],
            [],
            "--",
            lw=1.5,
            color=colors[iD],
            alpha=0.6
        )

        path_lines.append(line)
        drones.append(drone)
        horizon_lines.append(horizon)

        # Start
        ax.scatter(start[0,iD],
                start[1,iD],
                start[2,iD],
                color="green",
                s=80)

        # Goal (always visible)
        ax.scatter(goal[0,iD],
                goal[1,iD],
                goal[2,iD],
                color="orange",
                marker="*",
                s=180)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis("equal")
    ax.legend()

    def init():

        artists = []

        for iD in range(param.sim.nD):
            path_lines[iD].set_data([], [])
            path_lines[iD].set_3d_properties([])

            drones[iD].set_data(
                [executed_path[iD,0,0]],
                [executed_path[iD,0,1]]
            )
            drones[iD].set_3d_properties(
                [executed_path[iD,0,2]]
            )

            horizon_lines[iD].set_data([], [])
            horizon_lines[iD].set_3d_properties([])

            artists.extend([
                path_lines[iD],
                horizon_lines[iD],
                drones[iD]
            ])

        return artists


    def update(frame):

        artists = []

        for iD in range(param.sim.nD):

            path_lines[iD].set_data(
                executed_path[iD,:frame+1,0],
                executed_path[iD,:frame+1,1]
            )

            path_lines[iD].set_3d_properties(
                executed_path[iD,:frame+1,2]
            )

            drones[iD].set_data(
                [executed_path[iD,frame,0]],
                [executed_path[iD,frame,1]]
            )

            drones[iD].set_3d_properties(
                [executed_path[iD,frame,2]]
            )

            pred = horizon_history[iD, frame]
            horizon_lines[iD].set_data(
                pred[:,0],
                pred[:,1]
            )

            horizon_lines[iD].set_3d_properties(
                pred[:,2]
            )

            artists.extend([
                path_lines[iD],
                horizon_lines[iD],
                drones[iD]
            ])

        pts = executed_path.reshape(-1,3)
        margin = 0.5

        ax.set_xlim(pts[:,0].min()-margin, pts[:,0].max()+margin)
        ax.set_ylim(pts[:,1].min()-margin, pts[:,1].max()+margin)
        ax.set_zlim(pts[:,2].min()-margin, pts[:,2].max()+margin)

        return artists


    ani = FuncAnimation(
        fig,
        update,
        frames=executed_path.shape[1]-1,
        init_func=init,
        interval=50,
        blit=False
    )

    if param.display.save_to_mp4:
        ani.save("2_drone_collision_avoidance.mp4", writer="ffmpeg", fps=24)

    plt.show()

if __name__ == "__main__":
    param = Param()
    with open("simulation_results.pkl", "rb") as f:
        data = pickle.load(f)

    visualise(data, param)