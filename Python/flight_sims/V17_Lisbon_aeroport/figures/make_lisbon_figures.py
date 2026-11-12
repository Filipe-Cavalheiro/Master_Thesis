"""
Lisbon-airport figures and metrics (V17: spherical envelope, scaled ADMM).
Copy of the paper's figures/make_figures.py with the V17 paths; the
Crazyflie flight figure (fig_arena.pdf) is left out (the flight logs are in
the air-space-deterrent repository, Crazy_Flies/swarm_MPC_voronoi/logs).

    cd V17/figures && python make_lisbon_figures.py

Reads ../Python/Results/python/*.pkl, writes fig_mesh.pdf, fig_airspace.pdf,
fig_lisbon.pdf and fig_traj5_metrics.pdf next to this script and prints the
metrics of the paper's Table II.
"""
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = REPO / "Python"
sys.path.insert(0, str(PY))
os.chdir(PY)   # the simulation modules resolve their paths from here

from Structs_sim import SimulationConfig                       # noqa: E402
from restricted_airspace import RestrictedAirspace, NoAirspace  # noqa: E402
from voronoi_mesh_funcs import pre_compute_meshes, pos_based_mesh_transform  # noqa: E402
import experiment_arena as arena                                # noqa: E402

RESULTS = PY / "Results" / "python"
FLIGHTS = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists()) / "Results"   # flight runs

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL_W = 3.5      # IEEE column width [in]
PAGE_W = 7.16    # IEEE text width [in]
RED = "#c0392b"
ORANGE = "#e67e22"
WALL = "#34495e"
TARGET = "k"


def load_pkl(name):
    with open(RESULTS / name, "rb") as f:
        return pickle.load(f)


TRAJ5 = "simulation_results_traj_5_12drop4.pkl"    # 12 chasers, 4 dropped at 25 s


def pairwise_min(P, alive=None):
    """P (n, 3, N) -> (N,) minimum pairwise distance (between alive drones, alive (n, N))."""
    D = np.linalg.norm(P[:, None] - P[None], axis=2)          # (n, n, N)
    D[np.eye(P.shape[0], dtype=bool)] = np.inf
    if alive is not None:
        D[~(alive[:, None] & alive[None])] = np.inf
    return D.min(axis=(0, 1))


def alive_mask(d):
    """(n, N) True while the drone is active (a dropped drone's goal is logged as NaN)."""
    return ~np.isnan(d["p_d_hist"]).any(axis=1)


def drone_color(i, n):
    return plt.cm.tab10(i) if n <= 10 else plt.cm.tab20(i)


# ============================================================
# Fig. mesh: circle -> cone -> sphere -> circle on the red border
# ============================================================
def fig_mesh():
    R = 5.0
    sim = SimpleNamespace(sphere_radius=R, red_margin=1.0, red_zone_name="", orange_zone_name="")
    param = SimpleNamespace(sim=sim)
    v_circle, v_cone, v_sphere = pre_compute_meshes(401, param)
    vor = SimpleNamespace(airspace=NoAirspace(sim), v_circle=v_circle, v_cone=v_cone,
                          v_sphere=v_sphere, faces=np.empty((0, 3), int), ran_u2_eq_1=False)

    # wall at x = 0 (NoAirspace), target crossing it along +x
    cases = [(-8.0, r"(a) $d<-R/2$: disc on the wall"),
             (0.0, r"(b) $|d|<R/2$: cone"),
             (3.75, r"(c) $d>R/2$: cone$\to$sphere"),
             (8.0, r"(d) $d\geq R$: sphere")]
    fig = plt.figure(figsize=(PAGE_W, 1.9))
    for i, (x_t, title) in enumerate(cases):
        tgt = np.array([[x_t], [0.0], [0.0]])
        V, _, _ = pos_based_mesh_transform(tgt, vor, param)
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1.2, c=V[:, 0], cmap="viridis", depthshade=False)
        ax.scatter([x_t], [0], [0], c=TARGET, s=14, marker="*")
        yy, zz = np.meshgrid([-6, 6], [-6, 6])
        ax.plot_surface(np.zeros_like(yy), yy, zz, color=WALL, alpha=0.12, linewidth=0)
        ax.set_xlim(-10, 14); ax.set_ylim(-7, 7); ax.set_zlim(-7, 7)
        ax.set_box_aspect((24, 14, 14))
        ax.view_init(elev=18, azim=-58)
        ax.set_xticks([-8, 0, 8]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(r"$\hat n$ [m]", labelpad=-6)
        ax.tick_params(pad=-3)
        ax.set_title(title, fontsize=7, pad=-2)
    fig.subplots_adjust(wspace=0.0, left=0, right=1)
    fig.savefig(HERE / "fig_mesh.pdf")
    plt.close(fig)


# ============================================================
# Fig. Lisbon: top view of trajectories 5 and 6
# ============================================================
def wall_contour(ax, airspace, lims, level):
    (x0, x1), (y0, y1) = lims
    X, Y = np.meshgrid(np.linspace(x0, x1, 260), np.linspace(y0, y1, 260))
    sd = airspace.red.signed_distance(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    ax.contourf(X, Y, sd, levels=[-1e9, 0], colors=[RED], alpha=0.25)
    ax.contour(X, Y, sd, levels=[0], colors=[RED], linewidths=1.0)
    ax.contour(X, Y, sd, levels=[level], colors=[WALL], linewidths=1.0, linestyles="--")


def fig_lisbon(airspace):
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.7))
    for ax, name, title in [(axes[0], TRAJ5, "(a) Trajectory 5: straight border edge"),
                            (axes[1], "simulation_results_traj_6.pkl", r"(b) Trajectory 6: $34.5^\circ$ notch")]:
        d = load_pkl(name)
        P, T = d["position_history"].copy(), d["target_true_hist"][0:3]
        alive = alive_mask(d)
        lost = [i for i in range(P.shape[0]) if not alive[i].all()]
        k_drop = {i: int(np.argmin(alive[i])) for i in lost}
        P[np.broadcast_to(~alive[:, None], P.shape)] = np.nan    # dropped drones: path up to the drop
        pts = np.concatenate([P.transpose(1, 0, 2).reshape(3, -1), T], axis=1)
        pts = pts[:, np.isfinite(pts).all(axis=0)]
        c = 0.5 * (pts[:2].min(axis=1) + pts[:2].max(axis=1))
        h = 0.5 * (pts[:2].max(axis=1) - pts[:2].min(axis=1)).max() + 12
        lims = ((c[0] - h, c[0] + h), (c[1] - h, c[1] + h))
        wall_contour(ax, airspace, lims, airspace.wall_offset)
        for i in range(P.shape[0]):
            ax.plot(P[i, 0], P[i, 1], lw=0.7, color=drone_color(i, P.shape[0]))
        for i in lost:
            k = k_drop[i] - 1
            ax.plot(P[i, 0, k], P[i, 1, k], "v", color=drone_color(i, P.shape[0]), ms=4, mec="k", mew=0.4)
        ax.plot(T[0], T[1], color=TARGET, lw=1.3, label="target")
        ax.plot(T[0, 0], T[1, 0], "o", color=TARGET, ms=3)
        for k in range(0, T.shape[1], 1000):
            ax.plot(P[:, 0, k], P[:, 1, k], "x", color="0.25", ms=3, mew=0.7)
            ax.plot(T[0, k], T[1, k], ".", color=TARGET, ms=4)
            ax.annotate(f"{k / 100:.0f} s", (T[0, k], T[1, k]), fontsize=5.5,
                        xytext=(3, -8), textcoords="offset points")
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect("equal")
        ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
        ax.set_title(title, fontsize=8)
    axes[0].plot([], [], color=RED, label="red border"); axes[0].plot([], [], "--", color=WALL, label="wall")
    axes[0].plot([], [], "x", color="0.25", label="chasers every 10 s")
    axes[0].plot([], [], "v", color="0.6", mec="k", mew=0.4, ls="none", label="chaser lost")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=6)
    fig.tight_layout()
    fig.savefig(HERE / "fig_lisbon.pdf")
    plt.close(fig)


# ============================================================
# Fig. airspace: red/orange zones, wall and offset delta_w (problem statement)
# ============================================================
def fig_airspace(airspace):
    from matplotlib.patches import Polygon as MplPolygon, Rectangle, ConnectionPatch
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(COL_W, 1.95), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) overview of the Lisbon airport zones [km]
    for o in airspace.orange:
        ax0.add_patch(MplPolygon(o.vertices / 1e3, closed=True, fc=ORANGE, ec=ORANGE, alpha=0.25, lw=0.6))
    ax0.add_patch(MplPolygon(airspace.red.vertices / 1e3, closed=True, fc=RED, ec=RED, alpha=0.35, lw=0.9))
    ax0.plot(0, 0, "+", color="k", ms=4, mew=0.8)
    ax0.text(0.3, -0.4, "ARP", fontsize=5.5, va="top")
    ax0.text(-0.4, 1.6, r"$\mathcal{R}$", color=RED, fontsize=9, ha="center")
    ax0.text(-6.6, 5.5, r"$\mathcal{O}_\ell$", color=ORANGE, fontsize=9, ha="center")
    ax0.set_xlim(-9.5, 8.5); ax0.set_ylim(-11.5, 11); ax0.set_aspect("equal")
    ax0.set_xlabel("East [km]"); ax0.set_ylabel("North [km]", labelpad=1)
    ax0.set_title("(a) Lisbon airport (LPPT)", fontsize=7)

    # (b) zoom on the straight edge crossed by trajectory 5 [m]
    c = np.array([1696.5, 6725.0])                     # wall crossing of trajectory 5
    nhat = airspace.wall_normal(c)
    h = 42.0
    X, Y = np.meshgrid(np.linspace(-h, h, 300), np.linspace(-h, h, 300))
    sd = airspace.red.signed_distance(np.column_stack([X.ravel() + c[0], Y.ravel() + c[1]])).reshape(X.shape)
    dw, mr = airspace.wall_offset, airspace.red_margin
    ax1.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fc=ORANGE, alpha=0.12, lw=0))
    ax1.contourf(X, Y, sd, levels=[-1e9, 0], colors=[RED], alpha=0.35)
    ax1.contourf(X, Y, sd, levels=[mr, dw], colors=["0.6"], alpha=0.18)
    ax1.contour(X, Y, sd, levels=[0], colors=[RED], linewidths=1.0)
    ax1.contour(X, Y, sd, levels=[mr], colors=[RED], linewidths=0.6, linestyles=":")
    ax1.contour(X, Y, sd, levels=[dw], colors=[WALL], linewidths=1.1, linestyles="--")

    # delta_w along the normal through the crossing point; nhat points into R
    w = np.zeros(2)                                    # wall point (crossing), local coords
    b = w + dw * nhat                                  # red border point
    t = np.array([-nhat[1], nhat[0]])                  # tangent to the wall
    off = -14.0 * t
    ax1.annotate("", xy=b + off, xytext=w + off,
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color="k", shrinkA=0, shrinkB=0))
    m = 0.5 * (w + b) + off - 5.5 * t
    ax1.text(*m, r"$\delta_w$", fontsize=8, ha="center", va="center")
    ax1.annotate("", xy=w + 13 * t + 9 * nhat, xytext=w + 13 * t,
                 arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k", shrinkA=0, shrinkB=0, mutation_scale=7))
    ax1.text(*(w + 16.5 * t + 5 * nhat), r"$\hat{\mathbf{n}}$", fontsize=8, ha="center", va="center")

    # labels of the level sets and regions
    ax1.text(0.86, 0.13, r"$\mathcal{R}$" "\n" r"$s<0$", color=RED, fontsize=7, ha="center", va="center",
             transform=ax1.transAxes)
    rot = (np.degrees(np.arctan2(t[1], t[0])) + 90) % 180 - 90       # wall direction, kept readable
    ax1.text(*(0.5 * (w + b) + 28 * t), r"$0<s\leq\delta_w$", fontsize=6, ha="center", va="center", rotation=rot)
    ax1.text(*(w - 15 * nhat - 28 * t), r"$s>\delta_w$", fontsize=6, ha="center", va="center")
    ax1.text(*(w - 4 * nhat + 33 * t), r"$\mathcal{W}$", color=WALL, fontsize=8, ha="center", va="center")
    ax1.text(*(b - 4.5 * nhat + 33 * t), r"$m_r$", color=RED, fontsize=6, ha="center", va="center")

    # an incoming target ahead of the wall
    pT = w - 22 * nhat + 6 * t
    ax1.plot(*pT, "*", color=TARGET, ms=6)
    ax1.annotate("", xy=pT + 9 * nhat + 3 * t, xytext=pT,
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color=TARGET, shrinkA=2, shrinkB=0, mutation_scale=6))
    ax1.text(*(pT - 7 * t), r"$\mathbf{p}_T$", fontsize=7, ha="center", va="center")

    ax1.set_xlim(-h, h); ax1.set_ylim(-h, h); ax1.set_aspect("equal")
    ax1.set_xticks([-30, 0, 30]); ax1.set_yticks([-30, 0, 30])
    ax1.set_xlabel("East [m]"); ax1.grid(False)
    ax1.set_title("(b) Detail at the wall", fontsize=7)

    # zoom indicator
    ax0.plot(c[0] / 1e3, c[1] / 1e3, "s", mfc="none", mec="k", ms=4, mew=0.7)
    for ya, yb in [(c[1] / 1e3, -h), (c[1] / 1e3, h)]:
        fig.add_artist(ConnectionPatch(xyA=(c[0] / 1e3, ya), coordsA=ax0.transData, xyB=(-h, yb), coordsB=ax1.transData,
                                       color="0.4", lw=0.4, ls="-"))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(HERE / "fig_airspace.pdf")
    plt.close(fig)


# ============================================================
# Fig. trajectory 5 metrics
# ============================================================
def fig_traj5_metrics(airspace, sim):
    d = load_pkl(TRAJ5)
    P, V, Pd = d["position_history"], d["velocity_history"], d["p_d_hist"]
    T = d["target_true_hist"][0:3]
    alive = alive_mask(d)
    n = P.shape[0]
    t = np.arange(P.shape[2]) * sim.dTi
    lost = ~alive.all(axis=0)
    t_drop = t[np.argmax(lost)] if lost.any() else None
    sd_t = airspace.red_signed_distance(T.T)
    t_wall = t[np.argmax(sd_t < airspace.wall_offset)]
    t_red = t[np.argmax(sd_t < 0)]

    fig, ax = plt.subplots(4, 1, figsize=(COL_W, 4.6), sharex=True)
    r = np.where(alive, np.linalg.norm(P - T[None], axis=1), np.nan)
    for i in range(n):
        ax[0].plot(t, r[i], color=drone_color(i, n))
    ax[0].axhline(sim.sphere_radius, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel(r"$\|p_i-p_T\|$ [m]")
    dmin = pairwise_min(P, alive)
    ax[1].plot(t, dmin, color="C0")
    ax[1].axhline(sim.obstacle_avoidance_dim, color=RED, ls="--", lw=0.8)
    ax[1].set_ylabel(r"$\min_{i\neq j}\|p_i-p_j\|$ [m]")
    spd = np.where(alive, np.linalg.norm(V, axis=1), np.nan)
    for i in range(n):
        ax[2].plot(t, spd[i], color=drone_color(i, n))
    ax[2].axhline(sim.v_max, color=RED, ls="--", lw=0.8)
    ax[2].set_ylabel(r"$\|\dot p_i\|$ [m/s]")
    e = np.linalg.norm(P - Pd, axis=1)
    for i in range(n):
        ax[3].plot(t, e[i], color=drone_color(i, n))
    ax[3].set_ylabel(r"$\|p_i-p_i^\star\|$ [m]")
    ax[3].set_xlabel("Time [s]")
    for a in ax:
        a.axvline(t_wall, color=WALL, ls="--", lw=0.7)
        a.axvline(t_red, color=RED, ls="--", lw=0.7)
        if t_drop is not None:
            a.axvline(t_drop, color="0.3", ls=":", lw=0.7)
    ax[0].text(t_wall, ax[0].get_ylim()[1] * 0.85, " wall", color=WALL, fontsize=6)
    if t_drop is not None:
        ax[0].text(t_drop, ax[0].get_ylim()[1] * 0.85, " loss", color="0.3", fontsize=6)
    ax[0].text(t_red, ax[0].get_ylim()[1] * 0.85, " red", color=RED, fontsize=6)
    ax[3].set_xlim(t[0], t[-1])
    fig.align_ylabels(ax)
    fig.tight_layout(h_pad=0.3)
    fig.savefig(HERE / "fig_traj5_metrics.pdf")
    plt.close(fig)
    return t_wall, t_red, t_drop


# ============================================================
# Crazyflie flights
# ============================================================
def load_flight(path, chasers):
    df = pd.read_csv(path, dtype={"id": str, "flight_state": str})
    df = df[df["flight_state"].fillna("").str.strip() == "Formation"].copy()
    df["tr"] = df["time_s"].round(2)
    cols = ["opti_x", "opti_y", "opti_z", "sp_x", "sp_y", "sp_z", "target_x", "target_y", "target_z"]
    per = {i: df[df["id"] == i].groupby("tr")[cols].mean() for i in chasers}
    idx = None
    for i in chasers:
        idx = per[i].index if idx is None else idx.intersection(per[i].index)
    X = np.stack([per[i].loc[idx, cols[0:3]].to_numpy() for i in chasers])
    S = np.stack([per[i].loc[idx, cols[3:6]].to_numpy() for i in chasers])
    T = per[chasers[0]].loc[idx, cols[6:9]].to_numpy()
    t = np.asarray(idx) - idx[0]
    return t, X, S, T


def flight_metrics(t, X, S, T):
    D = np.linalg.norm(X[:, None] - X[None], axis=3)
    D[np.eye(X.shape[0], dtype=bool)] = np.inf
    r = np.linalg.norm(X - T[None], axis=2)
    e = np.linalg.norm(X - S, axis=2)
    return dict(duration=t[-1], min_sep=D.min(), min_target=r.min(), r_med=np.median(r),
                e_med=np.median(e), e_p95=np.percentile(e, 95), e_max=e.max())


def fig_arena():
    log_d = FLIGHTS / "swarm_MPC_voronoi_real_target_trajectory_20261026_145317" / "trajectory_20261026_145317.csv"
    fl_d = load_flight(log_d, ["1", "2", "3"])

    fig, ax = plt.subplots(1, 2, figsize=(PAGE_W, 2.5), gridspec_kw={"width_ratios": [1.45, 1.5]})
    # (a) top view of the flight with the Crazyflie target
    t, X, S, T = fl_d
    n = arena.TRAJ5_N_IN / np.linalg.norm(arena.TRAJ5_N_IN)
    tang = np.array([-n[1], n[0]])
    wall = arena.TRAJ5_START + 0.5 * n
    seg = np.array([wall - 4 * tang, wall + 4 * tang])
    ax[0].plot(seg[:, 0], seg[:, 1], "--", color=WALL, label="wall")
    ax[0].add_patch(plt.Rectangle(arena.ARENA_MIN[:2], *(arena.ARENA_MAX - arena.ARENA_MIN)[:2],
                                  fill=False, ec="0.5", lw=0.6, ls=":"))
    for i in range(X.shape[0]):
        ax[0].plot(X[i, :, 0], X[i, :, 1], color=plt.cm.tab10(i), lw=0.8, label=f"chaser {i + 1}")
        ax[0].plot(X[i, 0, 0], X[i, 0, 1], "o", color=plt.cm.tab10(i), ms=2.5)
    ax[0].plot(T[:, 0], T[:, 1], color=TARGET, lw=1.2, label="target (Crazyflie)")
    ax[0].annotate("", xy=T[-1, :2], xytext=T[-40, :2], arrowprops=dict(arrowstyle="->", lw=1))
    ax[0].set_xlim(-2.7, 2.7); ax[0].set_ylim(-2.7, 2.7); ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6, framealpha=0.9)
    ax[0].set_title("(a) Flight with a Crazyflie target", fontsize=8)

    # (b) distance of each chaser to the Crazyflie target
    for i in range(X.shape[0]):
        ax[1].plot(t, np.linalg.norm(X[i] - T, axis=1), color=plt.cm.tab10(i), label=f"chaser {i + 1}")
    ax[1].axhline(arena.TRAJ5_RADIUS, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel(r"$\|p_i-p_T\|$ [m]")
    ax[1].set_xlim(0, t[-1])
    ax[1].set_ylim(0.3, None)
    ax[1].legend(fontsize=6, ncol=3, loc="lower right")
    ax[1].set_title("(b) Distance to the target", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_arena.pdf")
    plt.close(fig)
    return {"drone_target": flight_metrics(*fl_d)}


def sim_metrics(name, dt, skip=0.0):
    d = load_pkl(name)
    P, V, Pd = d["position_history"], d["velocity_history"], d["p_d_hist"]
    T = d["target_true_hist"][0:3]
    est = d["perceived_target_hist"][:, 0:3]
    alive = alive_mask(d)
    k0 = int(skip / dt)
    e = np.linalg.norm(P - Pd, axis=1)
    return dict(n=P.shape[0], n_end=int(alive[:, -1].sum()), duration=(P.shape[2] - 1) * dt,
                min_sep=pairwise_min(P, alive).min(),
                min_sep_after=pairwise_min(P[:, :, k0:], alive[:, k0:]).min(),
                e_med=np.nanmedian(e), e_max=np.nanmax(e), v_max=np.linalg.norm(V, axis=1)[alive].max(),
                est_med=np.median(np.linalg.norm(est - T[None], axis=1)[alive]))


if __name__ == "__main__":
    sim = SimulationConfig()
    airspace = RestrictedAirspace(sim)
    fig_mesh()
    fig_airspace(airspace)
    fig_lisbon(airspace)
    t_wall, t_red, t_drop = fig_traj5_metrics(airspace, sim)
    print(f"trajectory 5: target crosses the wall at {t_wall:.2f} s, the red border at {t_red:.2f} s, "
          f"chasers lost at {t_drop} s")
    for name, dt in [(TRAJ5, sim.dTi), ("simulation_results_traj_6.pkl", sim.dTi),
                     ("arena5_3.pkl", None), ("arena5_4.pkl", None)]:
        if dt is None:
            dt = 0.01
        m = sim_metrics(name, dt, skip=2.0)
        print(name, {k: round(float(v), 4) for k, v in m.items()})
    # fig_arena() needs the flight logs (air-space-deterrent/Crazy_Flies), not included in V17
