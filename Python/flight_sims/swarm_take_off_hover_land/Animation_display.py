"""
Figures and animations of the simulation (as Animation_display.py of
../swarm_geometric_Fibonacci_sim_target/ and ../8traj/). Which ones: the
GraphicalDisplay flags in Structs.py.

    python Animation_display.py     # redraw them from simulation_results.pkl (written by main.py)
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from Structs import Param, SimulationData
from flight_code_funcs import PHASE_COLORS, PHASES, SHORT

_ANIMATIONS = []    # keeps FuncAnimation objects alive


def create_figures(data: SimulationData, param: Param):
    if param.display.position_plot is True:
        position_plot(data, param)

    if param.display.top_view_plot is True:
        top_view_plot(data, param)

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
    # figures on an interactive backend, otherwise no window can ever
    # receive the key press and the loop would hang forever.
    import matplotlib
    if figs and matplotlib.get_backend().lower() != "agg":
        while not quit_flag["pressed"]:
            plt.pause(0.1)

    # Close everything
    plt.close("all")

    print("Exited on q")


def _time(param: Param):
    return np.arange(param.sim.Nsim) * param.sim.dTi


def _shade(ax, phases):
    order = [p for p in PHASES if p in phases]
    for k, ph in enumerate(order):
        t1 = phases[order[k + 1]] if k + 1 < len(order) else phases.get("EOF", ax.get_xlim()[1])
        ax.axvspan(phases[ph], t1, color=PHASE_COLORS[ph], alpha=0.6, lw=0, label=SHORT[ph])


def _save(fig, name, param: Param):
    os.makedirs(param.display.save_location, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(param.display.save_location, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    print(f"saving {name} plot")


def position_plot(data: SimulationData, param: Param):
    """x, y, z of every drone (solid) and its setpoint (dotted), the flight phases shaded."""
    t = _time(param)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    for a, ax in enumerate(axes):
        _shade(ax, data.phases)
        for k, i in enumerate(data.ids):
            ax.plot(t, data.position_history[k, a], color=data.colors[k], lw=1.5, label=f"CF{int(i):02d} sim")
            ax.plot(t, data.p_d_hist[k, a], color=data.colors[k], lw=0.8, ls=":",
                    label=f"CF{int(i):02d} setpoint" if a == 0 else None)
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, fontsize=7, loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Simulated position (solid), setpoint (dotted)")
    fig.tight_layout()
    _save(fig, "position", param)
    plt.show(block=False)


def top_view_plot(data: SimulationData, param: Param):
    """x-y path of every drone (o start, x end)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, i in enumerate(data.ids):
        P = data.position_history[k]
        ax.plot(P[0], P[1], color=data.colors[k], lw=1.5, label=f"CF{int(i):02d}")
        ax.plot(P[0, 0], P[1, 0], "o", color=data.colors[k], ms=6)
        ax.plot(P[0, -1], P[1, -1], "x", color=data.colors[k], ms=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Top view (o start, x end)")
    fig.tight_layout()
    _save(fig, "top_view", param)
    plt.show(block=False)


def _equal_3d(ax, P):
    lo, hi = np.nanmin(P, axis=(0, 2)), np.nanmax(P, axis=(0, 2))
    c, r = (lo + hi) / 2, max(np.max(hi - lo) / 2, 0.3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(max(0.0, c[2] - r), c[2] + r)


def trajectory_plot(data: SimulationData, param: Param):
    """3D path of every drone."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for k, i in enumerate(data.ids):
        P = data.position_history[k]
        ax.plot(P[0], P[1], P[2], color=data.colors[k], lw=1.5, label=f"CF{int(i):02d}")
        ax.scatter(P[0, 0], P[1, 0], P[2, 0], color=data.colors[k], marker="o")
    _equal_3d(ax, data.position_history)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, f"trajectory_{param.sim.nD}_agents", param)
    plt.show(block=False)


def animate_simulation(data: SimulationData, param: Param):
    """3D animation of the drones with a trail (mp4 if param.display.save_to_mp4)."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _equal_3d(ax, data.position_history)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    dots = [ax.plot([], [], [], "o", color=data.colors[k])[0] for k in range(len(data.ids))]
    trails = [ax.plot([], [], [], "-", color=data.colors[k], lw=1.0)[0] for k in range(len(data.ids))]
    title = ax.set_title("")
    frames = np.linspace(0, param.sim.Nsim - 1, max(2, int(param.sim.Tend * param.display.mp4_fps))).astype(int)
    trail = int(param.display.trail_length)

    def update(f):
        n = frames[f]
        for k in range(len(data.ids)):
            P = data.position_history[k]
            dots[k].set_data_3d([P[0, n]], [P[1, n]], [P[2, n]])
            a = max(0, n - trail)
            trails[k].set_data_3d(P[0, a:n + 1], P[1, a:n + 1], P[2, a:n + 1])
        title.set_text(f"t = {n * param.sim.dTi:.2f} s")
        return dots + trails + [title]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000 / param.display.mp4_fps,
                        blit=False, repeat=False)
    _ANIMATIONS.append(ani)
    if param.display.save_to_mp4:
        print("Saving trajectory animation")
        os.makedirs(param.display.save_location, exist_ok=True)
        ani.save(os.path.join(param.display.save_location, f"trajectory_{param.sim.nD}_agents.mp4"),
                 writer="ffmpeg", fps=param.display.mp4_fps)


if __name__ == "__main__":
    # Figures / animations from the stored simulation (simulation_results.pkl,
    # written by main.py when param.display.store_data is True), without re-simulating.
    from pathlib import Path

    param = Param()
    param.display.save_to_mp4 = False   # we are assuming that the user has previously saved the mp4s

    file_path = Path(__file__).resolve().parent / "simulation_results.pkl"
    if not file_path.exists():
        raise SystemExit(f"No {file_path.name}: run main.py first (with param.display.store_data = True)")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # param.sim of the stored run (main.py saves it): the plots use it
    for key, value in data.pop("sim", {}).items():
        setattr(param.sim, key, value)
    param.display.save_location = data.pop("save_location", param.display.save_location)
    sim_data = SimulationData(**data)

    create_figures(sim_data, param)
