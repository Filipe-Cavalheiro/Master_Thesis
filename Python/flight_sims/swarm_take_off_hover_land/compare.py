"""
Real vs simulated take off / hover / return / land flight: metrics and figures.

    python swarm_take_off_hover_land/compare.py                  # newest run in Results/ and its _sim.csv
    python swarm_take_off_hover_land/compare.py --log LOG.csv [--sim SIM.csv] [--no-show]

SIM defaults to <log>_sim.csv (log_sim.py). The real setpoints are rebuilt with
the flight code's ramps from the real measured positions (the log of this
flight code has no setpoint columns), the simulated ones are logged by the
simulation.

Metrics (per drone; also saved to <run folder of the log>/comparison/metrics.json):
  * position difference sim - real: RMS per axis, 3D RMS and max, per phase
  * take off: lift-off time (z > floor + 2 cm), time to 90 % of the climb,
    overshoot above the hover height
  * hover (last 5 s before the return): mean height, height error to the setpoint (bias),
    horizontal and vertical jitter (std), horizontal drift from home
  * landing: touchdown time (z < LAND_HEIGHT + 1 cm: the drones settle on the
    landing setpoint, motors on), horizontal offset at touchdown, slide on the
    floor after it
  * setpoint tracking error |p - sp| of the real and the simulated drone
Figures (PDF + PNG): position.pdf, difference.pdf, takeoff_landing.pdf, summary.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from log_sim import (FLIGHT_CODE, LAND_HEIGHT, PAUSE_BEFORE_RETURN, PHASES, RETURN,  # noqa: E402
                     TAKEOFF_HEIGHT, load_log, newest_log, real_setpoints, setpoint_schedule)

PHASE_COLORS = {"Take off": "#cfe8ff", "Hover": "#d9f2d0", RETURN: "#f3e1f7", "Land": "#ffe0c7"}
SHORT = {"Take off": "Take off", "Hover": "Hover", RETURN: "Return", "Land": "Land"}


def interp_rows(t_src, X, t):
    return np.column_stack([np.interp(t, t_src, X[:, a], left=np.nan, right=np.nan) for a in range(X.shape[1])])


def load_sim(sim_path, ids):
    df = pd.read_csv(sim_path, dtype={"id": str, "flight_state": str})
    df["id"] = df["id"].str.replace(r"\.0$", "", regex=True)
    out = {}
    for i in ids:
        g = df[df["id"] == i].sort_values("time_s")
        out[i] = {"t": g["time_s"].to_numpy(float), "p": g[["opti_x", "opti_y", "opti_z"]].to_numpy(float),
                  "sp": g[["sp_x", "sp_y", "sp_z"]].to_numpy(float)}
    return out


def first_time(t, cond):
    k = np.flatnonzero(cond)
    return float(t[k[0]]) if len(k) else float("nan")


def drone_metrics(t, p, sp, phases, floor, home):
    """Event / phase metrics of one trajectory (real or sim)."""
    t_to, t_h, t_l, t_e = phases["Take off"], phases.get("Hover", np.nan), phases.get("Land", np.nan), phases["EOF"]
    m = {}
    z = p[:, 2]
    after = t >= t_to
    m["liftoff_s"] = first_time(t, after & (z > floor + 0.02)) - t_to
    m["t90_climb_s"] = first_time(t, after & (z > floor + 0.9 * (TAKEOFF_HEIGHT - floor))) - t_to
    if np.isfinite(t_h):
        w = (t >= t_h) & (t <= t_h + 3.0)
        m["overshoot_m"] = float(np.nanmax(z[w]) - TAKEOFF_HEIGHT) if w.any() else float("nan")
    t_he = phases.get(RETURN, t_l)       # end of the hover (+ pause)
    if np.isfinite(t_he) and np.isfinite(t_h):
        w = (t >= max(t_h, t_he - PAUSE_BEFORE_RETURN - 5.0)) & (t <= t_he - PAUSE_BEFORE_RETURN)
        if w.any():
            m["hover_mean_z_m"] = float(np.nanmean(z[w]))
            m["hover_z_bias_m"] = float(np.nanmean(z[w] - sp[w, 2]))
            m["hover_z_std_mm"] = float(1e3 * np.nanstd(z[w]))
            xy = p[w, :2]
            m["hover_xy_std_mm"] = float(1e3 * np.sqrt(np.nanmean(np.sum((xy - np.nanmean(xy, 0)) ** 2, 1))))
            m["hover_xy_drift_mm"] = float(1e3 * np.linalg.norm(np.nanmean(xy, 0) - home[:2]))
    if np.isfinite(t_l):
        # the drones settle on the LAND_HEIGHT setpoint (motors still on until EOF), not on the floor
        t_td = first_time(t, (t >= t_l) & (z < LAND_HEIGHT + 0.01))
        m["touchdown_s"] = t_td - t_l
        if np.isfinite(t_td):
            q = p[np.searchsorted(t, t_td)]
            m["landing_xy_offset_mm"] = float(1e3 * np.linalg.norm(q[:2] - home[:2]))
        # after touchdown (motors on, on the floor): sliding / tipping
        w = t >= (t_td if np.isfinite(t_td) else t_e)
        if w.sum() > 1:
            pw = p[w][np.all(np.isfinite(p[w]), 1)]
            m["slide_after_touchdown_mm"] = float(1e3 * np.max(np.linalg.norm(pw[:, :2] - pw[0, :2], axis=1)))
    for ph in PHASES:
        w = phase_mask(t, phases, ph)
        e = np.linalg.norm(p[w] - sp[w], axis=1)
        e = e[np.isfinite(e)]
        if len(e):
            m[f"tracking_rms_{SHORT[ph]}_mm"] = float(1e3 * np.sqrt(np.mean(e ** 2)))
    return m


def phase_mask(t, phases, ph):
    order = [p for p in PHASES if p in phases]
    if ph not in order:
        return np.zeros(len(t), bool)
    k = order.index(ph)
    t1 = phases[order[k + 1]] if k + 1 < len(order) else phases["EOF"]
    return (t >= phases[ph]) & (t < t1)


def shade(ax, phases):
    order = [p for p in PHASES if p in phases]
    for k, ph in enumerate(order):
        t1 = phases[order[k + 1]] if k + 1 < len(order) else phases["EOF"]
        ax.axvspan(phases[ph], t1, color=PHASE_COLORS[ph], alpha=0.6, lw=0, label=SHORT[ph])


def compare(log_path, sim_path=None, show=True, out_dir=None):
    log_path = Path(log_path)
    sim_path = Path(sim_path) if sim_path else log_path.with_name(log_path.stem + "_sim.csv")
    ids, real, phases, t_end = load_log(log_path)
    sim = load_sim(sim_path, ids)
    sched = setpoint_schedule(phases, real)
    out_dir = Path(out_dir) if out_dir else log_path.parent / "comparison"     # in the run folder
    out_dir.mkdir(parents=True, exist_ok=True)

    t_from, t_to = phases["Take off"], phases["EOF"] + 0.5
    res = {"log": str(log_path), "sim": str(sim_path), "phases_s": phases, "drones": {}}
    data = {}
    times = {i: real[i]["t"][(real[i]["t"] >= t_from - 0.5) & (real[i]["t"] <= t_to)] for i in ids}
    sp_real = real_setpoints(real, sched, times)
    for i in ids:
        r = real[i]
        w = (r["t"] >= t_from - 0.5) & (r["t"] <= t_to)
        t, p = r["t"][w], r["p"][w]
        home, floor = r["p"][0], float(r["p"][0, 2])
        sp_r = sp_real[i]
        p_s = interp_rows(sim[i]["t"], sim[i]["p"], t)
        sp_s = interp_rows(sim[i]["t"], sim[i]["sp"], t)
        data[i] = dict(t=t, p=p, sp=sp_r, ps=p_s, sps=sp_s)
        d = {"home": home.tolist()}
        e = p_s - p
        for ph in ("all", *PHASES):
            m = (t >= t_from) & (t <= phases["EOF"]) if ph == "all" else phase_mask(t, phases, ph)
            ee = e[m]
            ee = ee[np.all(np.isfinite(ee), 1)]
            if len(ee):
                d[f"diff_{SHORT.get(ph, ph)}"] = {"rms_xyz_mm": (1e3 * np.sqrt(np.mean(ee ** 2, 0))).round(1).tolist(),
                                   "rms_3d_mm": round(float(1e3 * np.sqrt(np.mean(np.sum(ee ** 2, 1)))), 1),
                                   "max_3d_mm": round(float(1e3 * np.max(np.linalg.norm(ee, axis=1))), 1)}
        d["real"] = drone_metrics(t, p, sp_r, phases, floor, home)
        d["sim"] = drone_metrics(t, p_s, sp_s, phases, floor, home)
        res["drones"][i] = d

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print_report(res)
    plot(ids, data, phases, res, out_dir, log_path.stem, show)
    print(f"Metrics and figures saved to {out_dir}")
    return res


def print_report(res):
    print("\nSimulation vs real flight: position difference (sim - real)")
    print(f"  {'drone':6s} {'phase':9s} {'RMS x/y/z [mm]':>22s} {'RMS 3D':>8s} {'max 3D':>8s}")
    for i, d in res["drones"].items():
        for ph in ("all", *(SHORT[q] for q in PHASES)):
            if f"diff_{ph}" in d:
                q = d[f"diff_{ph}"]
                x, y, z = q["rms_xyz_mm"]
                print(f"  CF{int(i):02d}   {ph:9s} {x:7.1f} {y:7.1f} {z:7.1f}  {q['rms_3d_mm']:8.1f} {q['max_3d_mm']:8.1f}")
    keys = [("liftoff_s", "lift-off after the takeoff start [s]", 2), ("t90_climb_s", "90 % of the climb [s]", 2),
            ("overshoot_m", "overshoot above 0.6 m [mm]", -3), ("hover_mean_z_m", "hover height [m]", 3),
            ("hover_z_bias_m", "hover height - setpoint [mm]", -3), ("hover_z_std_mm", "hover z jitter (std) [mm]", 1),
            ("hover_xy_std_mm", "hover xy jitter (std) [mm]", 1), ("hover_xy_drift_mm", "hover xy offset from home [mm]", 1),
            ("touchdown_s", "down to 6 cm after the landing start [s]", 2),
            ("landing_xy_offset_mm", "xy offset from home at touchdown [mm]", 1),
            ("slide_after_touchdown_mm", "slide on the floor after touchdown [mm]", 1),
            ("tracking_rms_Take off_mm", "setpoint tracking RMS, take off [mm]", 1),
            ("tracking_rms_Hover_mm", "setpoint tracking RMS, hover [mm]", 1),
            ("tracking_rms_Return_mm", "setpoint tracking RMS, return [mm]", 1),
            ("tracking_rms_Land_mm", "setpoint tracking RMS, landing [mm]", 1)]
    print("\n  real / sim" + "".join(f"{'CF%02d' % int(i):>20s}" for i in res["drones"]))
    for k, label, nd in keys:
        row = f"  {label:42s}"
        for d in res["drones"].values():
            a, b = d["real"].get(k, np.nan), d["sim"].get(k, np.nan)
            if nd < 0:      # metres shown in mm
                a, b, nd2 = 1e3 * a, 1e3 * b, 1
            else:
                nd2 = nd
            row += f"{a:9.{nd2}f} / {b:<8.{nd2}f}"
        print(row)


def plot(ids, data, phases, res, out_dir, title, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    figs = {}

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    for a, ax in enumerate(axes):
        shade(ax, phases)
        for k, i in enumerate(ids):
            d = data[i]
            ax.plot(d["t"], d["p"][:, a], color=colors(k), lw=1.8, label=f"CF{int(i):02d} real")
            ax.plot(d["t"], d["ps"][:, a], color=colors(k), lw=1.3, ls="--", label=f"CF{int(i):02d} sim")
            ax.plot(d["t"], d["sp"][:, a], color=colors(k), lw=0.8, ls=":", alpha=0.8,
                    label=f"CF{int(i):02d} setpoint" if a == 0 else None)
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(h, lab, ncol=4, fontsize=7, loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title}: position, real (solid) vs simulation (dashed), setpoint (dotted)")
    fig.tight_layout()
    figs["position"] = fig

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    for ax in axes:
        shade(ax, phases)
    for k, i in enumerate(ids):
        d = data[i]
        e = d["ps"] - d["p"]
        axes[0].plot(d["t"], 1e3 * np.linalg.norm(e, axis=1), color=colors(k), lw=1.5, label=f"CF{int(i):02d} |sim - real|")
        axes[1].plot(d["t"], 1e3 * np.linalg.norm(e[:, :2], axis=1), color=colors(k), lw=1.3, label=f"CF{int(i):02d} horizontal")
        axes[1].plot(d["t"], 1e3 * e[:, 2], color=colors(k), lw=1.3, ls="--", label=f"CF{int(i):02d} vertical (z)")
    axes[0].set_ylabel("3D difference [mm]")
    axes[1].set_ylabel("difference [mm]")
    axes[1].axhline(0, color="k", lw=0.6)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title}: simulation - real")
    fig.tight_layout()
    figs["difference"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    wins = [(phases["Take off"] - 0.3, phases.get("Hover", phases["Take off"] + 3) + 2.5, "take off"),
            (phases.get("Land", phases["EOF"] - 4) - 0.8, phases["EOF"] + 0.5, "landing")]
    for ax, (a, b, name) in zip(axes, wins):
        shade(ax, phases)
        for k, i in enumerate(ids):
            d = data[i]
            m = (d["t"] >= a) & (d["t"] <= b)
            ax.plot(d["t"][m], d["p"][m, 2], color=colors(k), lw=1.8, label=f"CF{int(i):02d} real")
            ax.plot(d["t"][m], d["ps"][m, 2], color=colors(k), lw=1.3, ls="--", label=f"CF{int(i):02d} sim")
            ax.plot(d["t"][m], d["sp"][m, 2], color=colors(k), lw=0.8, ls=":")
        ax.set_xlim(a, b)
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    figs["takeoff_landing"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["all", *(SHORT[p] for p in PHASES if p in phases)]
    width = 0.8 / len(ids)
    for k, i in enumerate(ids):
        d = res["drones"][i]
        vals = [d.get(f"diff_{ph}", {}).get("rms_3d_mm", np.nan) for ph in labels]
        ax.bar(np.arange(len(labels)) + (k - (len(ids) - 1) / 2) * width, vals, width, color=colors(k),
               label=f"CF{int(i):02d}")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("RMS 3D difference sim - real [mm]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title(f"{title}: simulation vs real, per phase")
    fig.tight_layout()
    figs["summary"] = fig

    for name, f in figs.items():
        for ext in ("pdf", "png"):
            f.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help=f"flight log (default: newest {FLIGHT_CODE} run in Results/)")
    ap.add_argument("--sim", help="simulated flight (default: <log>_sim.csv)")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    args = ap.parse_args()
    log_path = Path(args.log).resolve() if args.log else newest_log()
    print(f"Log: {log_path}")
    compare(log_path, args.sim, show=not args.no_show)


if __name__ == "__main__":
    main()
