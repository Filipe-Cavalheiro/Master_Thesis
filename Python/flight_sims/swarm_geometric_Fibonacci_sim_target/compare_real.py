"""
Real vs simulated swarm geometric Fibonacci flight: metrics and figures
(called by main.py --IRL_log; same role as compare_real.py of swarm_take_off_hover_land and 8traj).

Metrics (per drone; saved to <run folder of the log>/comparison/metrics.json):
  * position difference sim - real: RMS per axis, 3D RMS and max, per phase
  * setpoint tracking RMS |p - sp| of the real and the simulated drone, per phase
  * encirclement, last 10 s of the Main Controller: distance to the target, its
    error to sphere_radius (mean, std)
  * minimum distance between two drones in the Main Controller
Figures (PDF + PNG): position, top_view, encirclement, difference, summary.
The real setpoints are the logged ones (<log>_setpoints.csv, from 12 Nov),
else rebuilt (ramps from the real measured positions).
"""

import json
from pathlib import Path

import numpy as np

from flight_code_funcs import (MAIN, PAUSE_BEFORE_RETURN, PHASE_COLORS, PHASES, POS_COLS, SHORT, flight_target_traj,
                               homes_of, load_log, position_at, ramp_goal, ramp_setpoint, setpoint_events)

SETTLED_WINDOW = 10.0       # s, end of the Main Controller used for the encirclement metrics


def phase_end(phases, ph):
    order = [p for p in PHASES if p in phases]
    k = order.index(ph)
    return phases[order[k + 1]] if k + 1 < len(order) else phases["EOF"]


def phase_mask(t, phases, ph):
    if ph not in phases:
        return np.zeros(len(t), bool)
    return (t >= phases[ph]) & (t < phase_end(phases, ph))


def shade(ax, phases):
    for ph in (p for p in PHASES if p in phases):
        ax.axvspan(phases[ph], phase_end(phases, ph), color=PHASE_COLORS[ph], alpha=0.6, lw=0, label=SHORT[ph])


def min_pair_distance(P):
    """Minimum distance between two drones at every sample: P (n_drones, n_samples, 3)."""
    n = len(P)
    if n < 2:
        return np.full(P.shape[1], np.nan)
    return np.min([np.linalg.norm(P[a] - P[b], axis=1) for a in range(n) for b in range(a + 1, n)], axis=0)


def interp_rows(t_src, X, t):
    return np.column_stack([np.interp(t, t_src, X[:, a], left=np.nan, right=np.nan) for a in range(X.shape[1])])


def step_values(et, vals, t, t_stop):
    j = np.searchsorted(et, t, side="right") - 1
    s = np.where((j >= 0)[:, None], vals[np.clip(j, 0, None)], np.nan)
    s[t >= t_stop] = np.nan                          # motors off
    return s


def real_setpoints(c, ids, drones, sched, stream, phases, times, sp_log=None):
    """
    Setpoints the flight code sent to the real drones, at `times[id]`: the
    logged ones if there are, else rebuilt: the Main Controller stream
    (state-independent: the one the simulation computed) and the ramps from
    the real measured positions.
    """
    sp = {}
    if sp_log is not None and len(sp_log):
        for i in ids:
            g = sp_log[sp_log["id"] == i]
            sp[i] = step_values(g["time_s"].to_numpy(float), g[["sp_x", "sp_y", "sp_z"]].to_numpy(float),
                                times[i], phases["EOF"])
        return sp
    homes = homes_of(drones)
    starts = {name: {i: position_at(drones[i], t0) for i in ids} for name, t0, _, _ in sched}
    events = setpoint_events(sched, stream)
    for i in ids:
        ev = [e for e in events if e[4] in (None, i)]
        vals = np.array([stream[i][1][k] if name == MAIN else
                         ramp_setpoint(starts[name][i], ramp_goal(c, name, homes[i]), k, n)
                         for _, name, k, n, _ in ev])
        sp[i] = step_values(np.array([e[0] for e in ev]), vals, times[i], phases["EOF"])
    return sp


def target_series(c, t_start, t):
    """The virtual target at the times t: it starts with the Main Controller (t_start) and runs to the end."""
    out = np.full((len(t), 3), np.nan)
    if t_start is None:
        return out
    traj = flight_target_traj(c)["pos"]
    m = t >= t_start
    if m.any():
        out[m] = np.array([traj(tt - t_start).ravel() for tt in t[m]])
    return out


def plot_flight(ids, t, data, target, phases, c, out_dir, title):
    """
    data[i]: "real" and "sim" positions and "sp" (real) setpoints at the
    common times t; target: the virtual target at t. Figures position,
    top_view, encirclement, difference.
    """
    real = True
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    figs = {}
    sim_ls = "--" if real else "-"

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    for a, ax in enumerate(axes):
        shade(ax, phases)
        ax.plot(t, target[:, a], color="k", lw=1.0, ls="-.", label="target" if a == 0 else None)
        for k, i in enumerate(ids):
            d = data[i]
            if real:
                ax.plot(t, d["real"][:, a], color=colors(k), lw=1.8, label=f"CF{int(i):02d} real")
            ax.plot(t, d["sim"][:, a], color=colors(k), lw=1.3, ls=sim_ls, label=f"CF{int(i):02d} sim")
            ax.plot(t, d["sp"][:, a], color=colors(k), lw=0.8, ls=":", alpha=0.8,
                    label=f"CF{int(i):02d} setpoint" if a == 0 else None)
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, fontsize=7, loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title}: position" + (", real (solid) vs simulation (dashed)" if real else ", simulation (solid)")
                 + ", setpoint (dotted)")
    fig.tight_layout()
    figs["position"] = fig

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(target[:, 0], target[:, 1], color="k", lw=1.0, ls="-.", label="target")
    for k, i in enumerate(ids):
        d = data[i]
        for key, ls, lw in (("real", "-", 1.6), ("sim", sim_ls, 1.3)):
            if key not in d:
                continue
            P = d[key]
            ok = np.all(np.isfinite(P), 1)
            ax.plot(P[ok, 0], P[ok, 1], color=colors(k), lw=lw, ls=ls, label=f"CF{int(i):02d} {key}")
            ax.plot(P[ok, 0][0], P[ok, 1][0], "o", color=colors(k), ms=6)
            ax.plot(P[ok, 0][-1], P[ok, 1][-1], "x", color=colors(k), ms=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"{title}: top view (o start, x end)")
    fig.tight_layout()
    figs["top_view"] = fig

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    m = phase_mask(t, phases, MAIN)
    for ax in axes:
        shade(ax, phases)
        if MAIN in phases:
            ax.set_xlim(phases[MAIN] - 0.5, phase_end(phases, MAIN) + 0.5)
    axes[0].axhline(c.sphere_radius, color="k", lw=0.8, ls="-.", label=f"sphere radius {c.sphere_radius} m")
    for k, i in enumerate(ids):
        d = data[i]
        for key, ls, lw in (("real", "-", 1.6), ("sim", sim_ls, 1.3)):
            if key in d:
                axes[0].plot(t[m], np.linalg.norm(d[key][m] - target[m], axis=1), color=colors(k), ls=ls, lw=lw,
                             label=f"CF{int(i):02d} {key}")
    for key, ls in (("real", "-"), ("sim", sim_ls)):
        if all(key in data[i] for i in ids):
            dmin = min_pair_distance(np.array([data[i][key][m] for i in ids]))
            axes[1].plot(t[m], dmin, color="k" if key == "real" else "tab:orange", ls=ls, lw=1.4, label=key)
    axes[0].set_ylabel("distance to the target [m]")
    axes[1].set_ylabel("min distance between drones [m]")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=4)
    fig.suptitle(f"{title}: encirclement (Main Controller)")
    fig.tight_layout()
    figs["encirclement"] = fig

    if real:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
        for ax in axes:
            shade(ax, phases)
        for k, i in enumerate(ids):
            e = data[i]["sim"] - data[i]["real"]
            axes[0].plot(t, 1e3 * np.linalg.norm(e, axis=1), color=colors(k), lw=1.5, label=f"CF{int(i):02d} |sim - real|")
            axes[1].plot(t, 1e3 * np.linalg.norm(e[:, :2], axis=1), color=colors(k), lw=1.3, label=f"CF{int(i):02d} horizontal")
            axes[1].plot(t, 1e3 * e[:, 2], color=colors(k), lw=1.3, ls="--", label=f"CF{int(i):02d} vertical (z)")
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

    for name, f in figs.items():
        for ext in ("pdf", "png"):
            f.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    return figs


def compare(c, log_path, sim_df, sched, stream, ctrl_period, out_dir, meta=None, sp_log=None):
    import matplotlib.pyplot as plt

    ids, real, phases, _ = load_log(log_path)
    sim = {}
    for i in ids:
        g = sim_df[sim_df["id"].astype(str) == i].sort_values("time_s")
        sim[i] = {"t": g["time_s"].to_numpy(float), "p": g[POS_COLS].to_numpy(float),
                  "sp": g[["sp_x", "sp_y", "sp_z"]].to_numpy(float)}
    out_dir.mkdir(parents=True, exist_ok=True)

    # common time base: the samples of the first drone of the log
    t_from, t_to = phases["Take off"] - 0.5, phases["EOF"] + 0.5
    t = real[ids[0]]["t"]
    t = t[(t >= t_from) & (t <= t_to)]
    sp_real = real_setpoints(c, ids, real, sched, stream, phases, {i: t for i in ids}, sp_log)
    t_main = min((ts[0] for ts, _ in stream.values() if len(ts)), default=None)
    target = target_series(c, t_main, t)
    data = {}
    for i in ids:
        data[i] = {"real": interp_rows(real[i]["t"], real[i]["p"], t),
                   "sim": interp_rows(sim[i]["t"], sim[i]["p"], t),
                   "sp": sp_real[i],
                   "sp_sim": interp_rows(sim[i]["t"], sim[i]["sp"], t)}

    res = {"log": str(log_path), "phases_s": phases, "constants": vars(c), "controller_period_s": ctrl_period,
           "real_setpoints": "logged (<log>_setpoints.csv)" if sp_log is not None else "rebuilt",
           "flight_meta": "<log>_meta.json" if meta else None, "drones": {}}
    m_main = phase_mask(t, phases, MAIN)
    m_settled = m_main & (t >= phase_end(phases, MAIN) - PAUSE_BEFORE_RETURN - SETTLED_WINDOW) \
        if MAIN in phases else m_main
    for i in ids:
        d = data[i]
        r = {"home": real[i]["p"][0].tolist()}
        e = d["sim"] - d["real"]
        for ph in ("all", *PHASES):
            m = (t >= phases["Take off"]) & (t <= phases["EOF"]) if ph == "all" else phase_mask(t, phases, ph)
            ee = e[m]
            ee = ee[np.all(np.isfinite(ee), 1)]
            if not len(ee):
                continue
            q = {"rms_xyz_mm": (1e3 * np.sqrt(np.mean(ee ** 2, 0))).round(1).tolist(),
                 "rms_3d_mm": round(float(1e3 * np.sqrt(np.mean(np.sum(ee ** 2, 1)))), 1),
                 "max_3d_mm": round(float(1e3 * np.max(np.linalg.norm(ee, axis=1))), 1)}
            for key, P, S in (("real", d["real"], d["sp"]), ("sim", d["sim"], d["sp_sim"])):
                tr = np.linalg.norm(P[m] - S[m], axis=1)
                tr = tr[np.isfinite(tr)]
                if len(tr):
                    q[f"tracking_rms_{key}_mm"] = round(float(1e3 * np.sqrt(np.mean(tr ** 2))), 1)
            r[f"diff_{SHORT.get(ph, ph)}"] = q
        for key in ("real", "sim"):
            dist = np.linalg.norm(d[key][m_settled] - target[m_settled], axis=1)
            dist = dist[np.isfinite(dist)]
            if len(dist):
                r[f"encirclement_{key}"] = {"distance_to_target_mean_m": round(float(dist.mean()), 4),
                                            "radius_error_mean_m": round(float(dist.mean() - c.sphere_radius), 4),
                                            "radius_error_std_m": round(float(dist.std()), 4)}
        res["drones"][i] = r
    for key in ("real", "sim"):
        dmin = min_pair_distance(np.array([data[i][key][m_main] for i in ids]))
        if np.any(np.isfinite(dmin)):
            res[f"min_distance_between_drones_main_{key}_m"] = round(float(np.nanmin(dmin)), 4)

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print_report(res)
    figs = plot_flight(ids, t, data, target, phases, c, out_dir, log_path.stem)

    colors = plt.get_cmap("tab10")
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
    ax.set_title(f"{log_path.stem}: simulation vs real, per phase")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"summary.{ext}", bbox_inches="tight", dpi=150)
    figs["summary"] = fig
    print(f"Metrics and figures saved to {out_dir}")
    return res


def print_report(res):
    print("\nSimulation vs real flight: position difference (sim - real), setpoint tracking RMS")
    print(f"  {'drone':6s} {'phase':9s} {'RMS x/y/z [mm]':>22s} {'RMS 3D':>8s} {'max 3D':>8s} {'track real/sim':>16s}")
    for i, d in res["drones"].items():
        for ph in ("all", *(SHORT[q] for q in PHASES)):
            q = d.get(f"diff_{ph}")
            if q:
                x, y, z = q["rms_xyz_mm"]
                tr = f"{q.get('tracking_rms_real_mm', np.nan):7.1f}/{q.get('tracking_rms_sim_mm', np.nan):<7.1f}"
                print(f"  CF{int(i):02d}   {ph:9s} {x:7.1f} {y:7.1f} {z:7.1f}  {q['rms_3d_mm']:8.1f} "
                      f"{q['max_3d_mm']:8.1f}  {tr}")
    print(f"\n  encirclement, last {SETTLED_WINDOW:.0f} s of the Main Controller: distance to the target - sphere radius")
    for i, d in res["drones"].items():
        row = f"  CF{int(i):02d}"
        for key in ("real", "sim"):
            e = d.get(f"encirclement_{key}")
            if e:
                row += f"   {key}: {1e3 * e['radius_error_mean_m']:7.1f} mm (std {1e3 * e['radius_error_std_m']:.1f})"
        print(row)
    for key in ("real", "sim"):
        v = res.get(f"min_distance_between_drones_main_{key}_m")
        if v is not None:
            print(f"  min distance between drones, Main Controller, {key}: {v:.3f} m")
