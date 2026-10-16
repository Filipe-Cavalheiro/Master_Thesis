"""
Plot / animate multi-drone flight logs.

CSV columns (one row per drone per sample):
    time_s, flight_state, id,
    opti_x, opti_y, opti_z,            # OptiTrack (ground truth) position
    est_x, est_y, est_z,               # onboard estimator position
    est_vx, est_vy, est_vz,            # onboard estimator velocity
    roll_deg, pitch_deg, yaw_deg,      # attitude
    gyro_x_dps, gyro_y_dps, gyro_z_dps # angular rates

Usage:
    python plot_data.py                         # newest CSV in LOG_DIR
    python plot_data.py path/to/log.csv
    python plot_data.py path/to/log.csv --source est     # use estimator pos for 3D/XY/animation
    python plot_data.py path/to/log.csv --animate
    python plot_data.py path/to/log.csv --animate --save anim.gif
    python plot_data.py path/to/log.csv --save-figs out_dir
"""

import argparse
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOG_DIR = Path("Optitrack/logs")
LOG_GLOB = "*.csv"

REQUIRED_COLS = [
    "time_s", "flight_state", "id",
    "opti_x", "opti_y", "opti_z",
    "est_x", "est_y", "est_z",
    "est_vx", "est_vy", "est_vz",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
]
NUMERIC_COLS = [c for c in REQUIRED_COLS if c not in ("flight_state", "id")]

AXES = ("x", "y", "z")
OPTI_COLS = ["opti_x", "opti_y", "opti_z"]
EST_COLS = ["est_x", "est_y", "est_z"]
EST_VEL_COLS = ["est_vx", "est_vy", "est_vz"]
ATT_COLS = ["roll_deg", "pitch_deg", "yaw_deg"]
GYRO_COLS = ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]


# ============================================================
# DATA CONTAINER
# ============================================================

@dataclass
class FlightLog:
    t: np.ndarray                                    # (N,) seconds since first sample
    drone_ids: list = field(default_factory=list)    # sorted drone ids
    opti: dict = field(default_factory=dict)         # id -> (N, 3) OptiTrack position
    est: dict = field(default_factory=dict)          # id -> (N, 3) estimated position
    pos: dict = field(default_factory=dict)          # id -> (N, 3) position used for 3D/XY/animation
    vel_opti: dict = field(default_factory=dict)     # id -> (N, 3) finite-diff velocity of OptiTrack
    vel_est: dict = field(default_factory=dict)      # id -> (N, 3) estimator velocity
    att: dict = field(default_factory=dict)          # id -> (N, 3) roll, pitch, yaw [deg]
    gyro: dict = field(default_factory=dict)         # id -> (N, 3) gyro [deg/s]
    spans: list = field(default_factory=list)        # [(phase, t_start, t_end), ...]
    colors: dict = field(default_factory=dict)       # id -> color
    source: str = "opti"

    def label(self, oid):
        """CF01, CF02, ... (numeric id, or last two characters of a URI)."""
        s = str(oid).strip()
        if s.isdigit():
            return f"CF{int(s):02d}"
        return f"CF{s[-2:]}"


# ============================================================
# LOADING
# ============================================================

def _fill_nan(arr):
    """Linearly interpolate NaNs per column (used for derivatives only)."""
    return (
        pd.DataFrame(arr)
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )


def _extract(g, cols, t_abs, resample):
    """Return (N, len(cols)) array for one drone on the common time base."""
    if not resample:
        return g.set_index("time_s").reindex(t_abs)[cols].to_numpy(dtype=float)
    out = np.full((len(t_abs), len(cols)), np.nan)
    for k, c in enumerate(cols):
        ok = g[c].notna().to_numpy()
        if ok.sum() >= 2:
            out[:, k] = np.interp(
                t_abs, g["time_s"].to_numpy()[ok], g[c].to_numpy()[ok],
                left=np.nan, right=np.nan,
            )
    return out


def load_log(csv_path, source="opti", vel_smooth_window=5):
    """Load a log and return a FlightLog with one trajectory per drone id."""
    df = pd.read_csv(csv_path, dtype={"id": str, "flight_state": str})

    # in case no id is provided its assumed only one drone was recorded
    if "id" not in df.columns:
        df["id"] = "1" 
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns {missing}.")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["flight_state"] = df["flight_state"].fillna("").str.strip()
    df["id"] = df["id"].fillna("").str.strip()

    df = df[df["id"] != ""].dropna(subset=["time_s"])
    if df.empty:
        raise ValueError("No drone samples found in the CSV.")

    per_drone = {
        oid: g.drop_duplicates("time_s").sort_values("time_s")
        for oid, g in df.groupby("id")
    }

    # ---- common time base -----------------------------------
    # If all drones share the same timestamps, use them directly; otherwise
    # resample every drone onto a uniform grid.
    t_union = np.sort(df["time_s"].unique())
    resample = not all(len(g) == len(t_union) for g in per_drone.values())
    if resample:
        dts = [np.median(np.diff(g["time_s"])) for g in per_drone.values() if len(g) > 1]
        dt = float(np.median(dts)) if dts else 0.02
        t_abs = np.arange(t_union[0], t_union[-1] + 1e-9, dt)
        print(f"Note: drones have different timestamps; resampled to {1 / dt:.0f} Hz.")
    else:
        t_abs = t_union

    t0 = t_abs[0]
    log = FlightLog(t=t_abs - t0, source=source)

    # ---- per-drone signals ----------------------------------
    for oid, g in per_drone.items():
        log.opti[oid] = _extract(g, OPTI_COLS, t_abs, resample)
        log.est[oid] = _extract(g, EST_COLS, t_abs, resample)
        log.vel_est[oid] = _extract(g, EST_VEL_COLS, t_abs, resample)
        log.att[oid] = _extract(g, ATT_COLS, t_abs, resample)
        log.gyro[oid] = _extract(g, GYRO_COLS, t_abs, resample)

    log.drone_ids = sorted(log.opti)
    log.pos = log.est if source == "est" else log.opti

    cmap = plt.get_cmap("tab10")
    for i, oid in enumerate(log.drone_ids):
        log.colors[oid] = cmap(i % 10)

    # ---- OptiTrack velocity (finite difference, lightly smoothed)
    for oid in log.drone_ids:
        p = log.opti[oid]
        if np.isnan(p).all() or len(log.t) < 2:
            log.vel_opti[oid] = np.full_like(p, np.nan)
            continue
        v = np.gradient(_fill_nan(p), log.t, axis=0)
        log.vel_opti[oid] = (
            pd.DataFrame(v)
            .rolling(vel_smooth_window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    # ---- phase spans from flight_state ----------------------
    state = (
        df[df["flight_state"] != ""]
        .drop_duplicates("time_s")
        .set_index("time_s")["flight_state"]
        .sort_index()
    )
    phase_of = state.reindex(t_abs, method="ffill").fillna("").to_numpy()

    i = 0
    n = len(phase_of)
    while i < n:
        j = i
        while j + 1 < n and phase_of[j + 1] == phase_of[i]:
            j += 1
        if phase_of[i] != "":
            t_start = log.t[i]
            t_end = log.t[j + 1] if j + 1 < n else log.t[-1]
            if t_end > t_start:
                log.spans.append((str(phase_of[i]), t_start, t_end))
        i = j + 1

    return log


# ============================================================
# DERIVED QUANTITIES
# ============================================================

def controller_mask(log):
    """Boolean mask of samples inside a 'controller' phase (all True if none)."""
    mask = np.zeros_like(log.t, dtype=bool)
    found = False
    for name, t0, t1 in log.spans:
        if "controller" in name.lower():
            mask |= (log.t >= t0) & (log.t <= t1)
            found = True
    return mask if found else np.ones_like(log.t, dtype=bool)


def estimation_error(log, oid):
    """Estimator position minus OptiTrack position, (N, 3)."""
    return log.est[oid] - log.opti[oid]


def centroid(log):
    """Swarm centroid at every time step, (N, 3)."""
    stack = np.stack([log.pos[o] for o in log.drone_ids], axis=0)
    with np.errstate(all="ignore"):
        return np.nanmean(stack, axis=0)


def distance_to_centroid(log, oid):
    return np.linalg.norm(log.pos[oid] - centroid(log), axis=1)


def min_pairwise_distance(log):
    """Minimum inter-drone distance at every time step (None if < 2 drones)."""
    if len(log.drone_ids) < 2:
        return None
    dists = [
        np.linalg.norm(log.pos[a] - log.pos[b], axis=1)
        for a, b in combinations(log.drone_ids, 2)
    ]
    with np.errstate(all="ignore"):
        return np.nanmin(np.vstack(dists), axis=0)


def print_summary(log):
    dur = log.t[-1]
    print(f"{len(log.t)} samples over {dur:.1f} s (~{len(log.t) / max(dur, 1e-9):.0f} Hz)")
    print(f"Drones: {[log.label(d) for d in log.drone_ids]}")
    print(f"Phases: {[s[0] for s in log.spans]}")
    print(f"Position source for 3D/XY/animation: {log.source}")

    for oid in log.drone_ids:
        n_missing = int(np.isnan(log.opti[oid][:, 0]).sum())
        if n_missing:
            print(f"Note: {log.label(oid)} has {n_missing} samples without OptiTrack data.")

    print("\nEstimator error vs OptiTrack (position norm):")
    for oid in log.drone_ids:
        e = np.linalg.norm(estimation_error(log, oid), axis=1)
        if np.isfinite(e).any():
            rmse = np.sqrt(np.nanmean(e ** 2))
            print(f"  {log.label(oid)}: mean {np.nanmean(e):.3f} m, "
                  f"RMSE {rmse:.3f} m, max {np.nanmax(e):.3f} m")

    mask = controller_mask(log)
    if len(log.drone_ids) >= 2:
        print("\nDistance to swarm centroid during controller phase:")
        for oid in log.drone_ids:
            d = distance_to_centroid(log, oid)[mask]
            if np.isfinite(d).any():
                print(f"  {log.label(oid)}: mean {np.nanmean(d):.3f} m, "
                      f"std {np.nanstd(d):.3f} m")

    mpd = min_pairwise_distance(log)
    if mpd is not None and np.isfinite(mpd).any():
        i = int(np.nanargmin(mpd))
        print(f"\nClosest approach between any two drones: "
              f"{mpd[i]:.3f} m at t = {log.t[i]:.2f} s")


# ============================================================
# HELPERS
# ============================================================

def _phase_colors(spans):
    names = []
    for name, _, _ in spans:
        if name not in names:
            names.append(name)
    cmap = plt.get_cmap("Pastel1")
    return {n: cmap(i % 9) for i, n in enumerate(names)}


def shade_phases(axes, spans):
    """Shade flight phases on each axis; legend entries on the first axis only."""
    colors = _phase_colors(spans)
    seen = set()
    for ax in axes:
        for name, t0, t1 in spans:
            label = None
            if ax is axes[0] and name not in seen:
                label = name
                seen.add(name)
            ax.axvspan(t0, t1, color=colors[name], alpha=0.35, label=label)


def plot_stack(log, title, panels, xlabel="Time (s)"):
    """
    Vertical stack of time-series panels.

    panels: list of (ylabel, [(y_array, legend_label, color, plot_kwargs), ...],
                     [optional hlines: (value, label)])
    """
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(12, 2.8 * len(panels) + 1), sharex=True
    )
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        ylabel, series = panel[0], panel[1]
        hlines = panel[2] if len(panel) > 2 else []
        for y, label, color, kwargs in series:
            ax.plot(log.t, y, label=label, color=color, **kwargs)
        for value, label in hlines:
            ax.axhline(value, color="gray", linestyle=":", linewidth=1.2, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)

    shade_phases(axes, log.spans)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _equal_3d_axes(ax, points):
    """Equal scaling on all three axes. points: (M, 3) array (NaNs ignored)."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) / 2
    half = max((maxs - mins).max() / 2, 0.1) * 1.1

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _all_points(log):
    return np.vstack([log.pos[o] for o in log.drone_ids])


def _first_last_valid(arr):
    ok = np.isfinite(arr).all(axis=1)
    idx = np.flatnonzero(ok)
    return (idx[0], idx[-1]) if idx.size else (None, None)


def _per_drone_panels(log, data, ylabels, **kwargs):
    """One panel per component (0..2) with one line per drone."""
    panels = []
    for k, ylabel in enumerate(ylabels):
        series = [
            (data[o][:, k], log.label(o), log.colors[o], dict(linewidth=1.4, **kwargs))
            for o in log.drone_ids
        ]
        panels.append((ylabel, series))
    return panels


# ============================================================
# STATIC PLOTS
# ============================================================

def plot_3d(log):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for oid in log.drone_ids:
        p = log.pos[oid]
        ax.plot(p[:, 0], p[:, 1], p[:, 2],
                label=log.label(oid), color=log.colors[oid], linewidth=1.6)
        i0, i1 = _first_last_valid(p)
        if i0 is not None:
            ax.scatter(*p[i0], marker="o", s=45, color=log.colors[oid])
            ax.scatter(*p[i1], marker="X", s=60, color=log.colors[oid])

    _equal_3d_axes(ax, _all_points(log))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Trajectories [{log.source}]  (o = start, X = end)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_top_down(log):
    fig, ax = plt.subplots(figsize=(8, 7))

    for oid in log.drone_ids:
        p = log.pos[oid]
        ax.plot(p[:, 0], p[:, 1], label=log.label(oid),
                color=log.colors[oid], linewidth=1.6)
        i0, i1 = _first_last_valid(p)
        if i0 is not None:
            ax.scatter(p[i0, 0], p[i0, 1], marker="o", s=55,
                       color=log.colors[oid], zorder=5)
            ax.scatter(p[i1, 0], p[i1, 1], marker="X", s=70,
                       color=log.colors[oid], zorder=5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Top-Down Trajectories (X-Y) [{log.source}]  (o = start, X = end)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_position(log):
    """OptiTrack (solid) vs estimator (dashed) position per axis."""
    panels = []
    for k, a in enumerate(AXES):
        series = []
        for o in log.drone_ids:
            series.append((log.opti[o][:, k], f"{log.label(o)} opti",
                           log.colors[o], {"linewidth": 1.6}))
            series.append((log.est[o][:, k], f"{log.label(o)} est",
                           log.colors[o], {"linewidth": 1.0, "linestyle": "--"}))
        panels.append((f"{a.upper()} (m)", series))
    return plot_stack(log, "Position vs Time (solid = OptiTrack, dashed = estimator)", panels)


def plot_velocity(log):
    """Estimator velocity (solid) vs finite-difference OptiTrack velocity (dotted)."""
    panels = []
    for k, a in enumerate(AXES):
        series = []
        for o in log.drone_ids:
            series.append((log.vel_est[o][:, k], f"{log.label(o)} est",
                           log.colors[o], {"linewidth": 1.4}))
            series.append((log.vel_opti[o][:, k], f"{log.label(o)} opti diff",
                           log.colors[o], {"linewidth": 1.0, "linestyle": ":"}))
        panels.append((f"V{a.upper()} (m/s)", series))

    series = [
        (np.linalg.norm(log.vel_est[o], axis=1), log.label(o), log.colors[o],
         {"linewidth": 1.4})
        for o in log.drone_ids
    ]
    panels.append(("|v| est (m/s)", series))
    return plot_stack(
        log, "Velocity (solid = estimator, dotted = finite difference of OptiTrack)", panels
    )


def plot_attitude(log):
    return plot_stack(
        log, "Attitude",
        _per_drone_panels(log, log.att, ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]),
    )


def plot_gyro(log):
    return plot_stack(
        log, "Gyroscope",
        _per_drone_panels(log, log.gyro, ["Gyro X (deg/s)", "Gyro Y (deg/s)", "Gyro Z (deg/s)"]),
    )


def plot_estimation_error(log):
    """Estimator position minus OptiTrack position."""
    errs = {o: estimation_error(log, o) for o in log.drone_ids}
    panels = _per_drone_panels(
        log, errs, [f"Err {a.upper()} (m)" for a in AXES]
    )
    series = [
        (np.linalg.norm(errs[o], axis=1), log.label(o), log.colors[o], {"linewidth": 1.4})
        for o in log.drone_ids
    ]
    panels.append(("|err| (m)", series))
    return plot_stack(log, "Estimator error (estimator - OptiTrack)", panels)


def plot_distances(log):
    """Distance of each drone to the swarm centroid, and closest drone-drone distance."""
    panels = []

    if len(log.drone_ids) >= 2:
        series = [
            (distance_to_centroid(log, o), log.label(o), log.colors[o], {"linewidth": 1.5})
            for o in log.drone_ids
        ]
        panels.append(("Dist. to centroid (m)", series))

    mpd = min_pairwise_distance(log)
    if mpd is not None:
        panels.append((
            "Min drone-drone (m)",
            [(mpd, "closest pair", "tab:red", {"linewidth": 1.5})],
        ))

    if not panels:
        return None
    return plot_stack(log, "Swarm Distances", panels)


def plot_flight_log(csv_path, source="opti"):
    """Create every static figure. Returns (log, {name: figure})."""
    log = load_log(csv_path, source=source)

    figs = {
        "trajectory_3d": plot_3d(log),
        "trajectory_xy": plot_top_down(log),
        "position": plot_position(log),
        "velocity": plot_velocity(log),
        "attitude": plot_attitude(log),
        "gyro": plot_gyro(log),
        "estimation_error": plot_estimation_error(log),
    }
    dist_fig = plot_distances(log)
    if dist_fig is not None:
        figs["distances"] = dist_fig
    return log, figs


# ============================================================
# ANIMATION
# ============================================================

def animate_flight(
    log,
    trail_length=100,
    playback_speed=1.0,
    stride=1,
    save_path=None,
):
    """
    Animate all drones in 3D.

    trail_length   : samples of history to draw (None = full trail)
    playback_speed : 1.0 = real time
    stride         : use every Nth sample (raise this if playback is slow)
    """
    if playback_speed <= 0:
        raise ValueError("playback_speed must be greater than zero.")
    stride = max(1, int(stride))

    t = log.t
    frames = np.arange(0, len(t), stride)
    median_dt = np.median(np.diff(t)) if len(t) > 1 else 0.02
    interval_ms = max(1, int(median_dt * stride * 1000 / playback_speed))

    # phase name per sample, for the on-screen caption
    phase_of = np.full(len(t), "", dtype=object)
    for name, t0, t1 in log.spans:
        phase_of[(t >= t0) & (t <= t1)] = name

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    _equal_3d_axes(ax, _all_points(log))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Animated Flight [{log.source}]")

    trails, dots = {}, {}
    for oid in log.drone_ids:
        c = log.colors[oid]
        trails[oid], = ax.plot([], [], [], color=c, linewidth=1.2,
                               label=log.label(oid))
        dots[oid], = ax.plot([], [], [], "o", markersize=8, color=c)
    ax.legend(loc="upper right")

    info = fig.text(0.5, 0.02, "", ha="center")

    def set_3d(line, x, y, z):
        line.set_data(x, y)
        line.set_3d_properties(z)

    def update(i):
        start = 0 if trail_length is None else max(0, i - trail_length)
        sl = slice(start, i + 1)

        for oid in log.drone_ids:
            p = log.pos[oid]
            set_3d(trails[oid], p[sl, 0], p[sl, 1], p[sl, 2])
            set_3d(dots[oid], [p[i, 0]], [p[i, 1]], [p[i, 2]])

        info.set_text(
            f"t = {t[i]:.2f} s | phase: {phase_of[i] or '-'} | "
            f"playback {playback_speed:g}x"
        )
        return (*trails.values(), *dots.values(), info)

    anim = FuncAnimation(
        fig, update, frames=frames, interval=interval_ms,
        blit=False, repeat=False,
    )

    if save_path:
        save_path = str(save_path)
        fps = max(1, round(1000 / interval_ms))
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end in .gif or .mp4")
        print(f"Animation saved to: {save_path}")

    return fig, anim


# ============================================================
# ENTRY POINT
# ============================================================

def newest_log(log_dir=LOG_DIR):
    files = sorted(Path(log_dir).glob(LOG_GLOB), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No {LOG_GLOB} files in {log_dir}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="Plot multi-drone Crazyflie logs.")
    parser.add_argument("csv", nargs="?", help="Log CSV (default: newest)")
    parser.add_argument("--source", choices=("opti", "est"), default="opti",
                        help="Position source for 3D/XY plots and animation (default opti)")
    parser.add_argument("--animate", action="store_true",
                        help="Also show the 3D animation")
    parser.add_argument("--save", metavar="FILE",
                        help="Save the animation (.gif or .mp4)")
    parser.add_argument("--save-figs", metavar="DIR",
                        help="Save static figures as PNG into DIR")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Animation playback speed (default 1.0)")
    parser.add_argument("--trail", type=int, default=100,
                        help="Animation trail length in samples")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth sample in the animation")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else newest_log()
    print(f"Loading {csv_path}")

    log, figs = plot_flight_log(csv_path, source=args.source)
    print_summary(log)

    if args.save_figs:
        out = Path(args.save_figs)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out / f"{csv_path.stem}_{name}.png", dpi=150)
        print(f"Figures saved to {out}")

    anim = None
    if args.animate or args.save:
        _, anim = animate_flight(
            log,
            trail_length=args.trail,
            playback_speed=args.speed,
            stride=args.stride,
            save_path=args.save,
        )

    plt.show()
    return anim


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nProgram terminated.")