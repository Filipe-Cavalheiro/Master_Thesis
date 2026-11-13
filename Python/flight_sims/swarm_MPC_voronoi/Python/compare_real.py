"""
Real vs simulated swarm_MPC_voronoi flight: metrics and figures (called by
flight_replay.replay, main.py --IRL_log; same role and figures as
compare_real.py of swarm_geometric_Fibonacci_sim_target).

Metrics (per drone; saved to <run folder of the log>/comparison/metrics.json):
  * position difference sim - real: RMS per axis, 3D RMS and max, whole
    flight and per phase
  * setpoint tracking RMS |p - sp| of the real and of the simulated drone,
    per phase (real: the setpoints logged in the flight)
  * encirclement, last SETTLED_WINDOW s of the Formation phase (chasers): distance
    to the target, its error to the sphere radius (mean, std)
  * minimum distance between two drones in the Formation phase
Figures (PDF + PNG), the flight phases as coloured backgrounds: position,
top_view, encirclement, difference, summary.
"""

import json

import numpy as np

PHASES = ("Take off", "Approach", "Formation", "Return", "Land")       # logger.mark_state of the flight code
FORMATION = "Formation"
# as swarm_geometric_Fibonacci_sim_target (Take off, Main Controller -> Formation, Return, Land), plus Approach
PHASE_COLORS = {"Take off": "#cfe8ff", "Approach": "#fff3c4", FORMATION: "#d9f2d0",
                "Return": "#f3e1f7", "Land": "#ffe0c7"}
SETTLED_WINDOW = 10.0       # s, end of the Formation phase used for the encirclement metrics


def phase_end(phases, ph):
    order = sorted((p for p in PHASES if p in phases), key=lambda p: phases[p])
    k = order.index(ph)
    return phases[order[k + 1]] if k + 1 < len(order) else phases["EOF"]


def phase_mask(t, phases, ph):
    if ph not in phases:
        return np.zeros(len(t), bool)
    return (t >= phases[ph]) & (t < phase_end(phases, ph))


def shade(ax, phases, label=True):
    """The flight phases as coloured backgrounds (legend entries if label)."""
    for ph in (p for p in PHASES if p in phases):
        ax.axvspan(phases[ph], phase_end(phases, ph), color=PHASE_COLORS[ph], alpha=0.6, lw=0, zorder=0,
                   label=ph if label else None)


def min_pair_distance(P):
    """Minimum distance between two drones at every sample: P (n_drones, n_samples, 3)."""
    n = len(P)
    if n < 2:
        return np.full(P.shape[1], np.nan)
    return np.min([np.linalg.norm(P[a] - P[b], axis=1) for a in range(n) for b in range(a + 1, n)], axis=0)


def interp_rows(t_src, X, t):
    ok = np.all(np.isfinite(X), axis=1)
    return np.column_stack([np.interp(t, t_src[ok], X[ok, a], left=np.nan, right=np.nan) for a in range(X.shape[1])])


def plot_flight(ids, t, data, target, phases, sphere_radius, chasers, out_dir, title):
    """
    data[i]: "real" and "sim" positions and "sp" (real) setpoints at the
    common times t; target: the logged target at t. Figures position,
    top_view, encirclement, difference.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    figs = {}
    name = {i: f"CF{int(i):02d}" + ("" if i in chasers else " (target)") for i in ids}

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    for a, ax in enumerate(axes):
        shade(ax, phases, label=a == 0)
        ax.plot(t, target[:, a], color="k", lw=1.0, ls="-.", label="target" if a == 0 else None)
        for k, i in enumerate(ids):
            d = data[i]
            ax.plot(t, d["real"][:, a], color=colors(k), lw=1.8, label=f"{name[i]} real" if a == 0 else None)
            ax.plot(t, d["sim"][:, a], color=colors(k), lw=1.3, ls="--", label=f"{name[i]} sim" if a == 0 else None)
            ax.plot(t, d["sp"][:, a], color=colors(k), lw=0.8, ls=":", alpha=0.8,
                    label=f"{name[i]} setpoint" if a == 0 else None)
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=5, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.45))
    axes[-1].set_xlabel("Time from the take off [s]")
    fig.suptitle(f"{title}: position, real (solid) vs simulation (dashed), setpoint (dotted)")
    fig.tight_layout()
    figs["position"] = fig

    fig, ax = plt.subplots(figsize=(7, 6))
    ok = np.all(np.isfinite(target), 1)
    ax.plot(target[ok, 0], target[ok, 1], color="k", lw=1.0, ls="-.", label="target")
    for k, i in enumerate(ids):
        for key, ls, lw in (("real", "-", 1.6), ("sim", "--", 1.3)):
            P = data[i][key]
            ok = np.all(np.isfinite(P), 1)
            if not ok.any():
                continue
            ax.plot(P[ok, 0], P[ok, 1], color=colors(k), lw=lw, ls=ls, label=f"{name[i]} {key}")
            ax.plot(P[ok, 0][0], P[ok, 1][0], "o", color=colors(k), ms=6)
            ax.plot(P[ok, 0][-1], P[ok, 1][-1], "x", color=colors(k), ms=8)
    ax.add_patch(plt.Rectangle((-2.5, -2.5), 5, 5, fill=False, ls="--", color="gray", lw=0.8))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"{title}: top view (o start, x end, dashed box: drone area)")
    fig.tight_layout()
    figs["top_view"] = fig

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    m = phase_mask(t, phases, FORMATION)
    for ax in axes:
        shade(ax, phases)
        if FORMATION in phases:
            ax.set_xlim(phases[FORMATION] - 0.5, phase_end(phases, FORMATION) + 0.5)
    axes[0].axhline(sphere_radius, color="k", lw=0.8, ls="-.", label=f"sphere radius {sphere_radius} m")
    for k, i in enumerate(ids):
        if i not in chasers:
            continue
        for key, ls, lw in (("real", "-", 1.6), ("sim", "--", 1.3)):
            axes[0].plot(t[m], np.linalg.norm(data[i][key][m] - target[m], axis=1), color=colors(k), ls=ls, lw=lw,
                         label=f"{name[i]} {key}")
    for key, ls in (("real", "-"), ("sim", "--")):
        dmin = min_pair_distance(np.array([data[i][key][m] for i in ids]))
        axes[1].plot(t[m], dmin, color="k" if key == "real" else "tab:orange", ls=ls, lw=1.4, label=key)
    axes[0].set_ylabel("distance to the target [m]")
    axes[1].set_ylabel("min distance between drones [m]")
    axes[1].set_xlabel("Time from the take off [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=4)
    fig.suptitle(f"{title}: encirclement (Formation phase)")
    fig.tight_layout()
    figs["encirclement"] = fig

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    for ax in axes:
        shade(ax, phases)
    for k, i in enumerate(ids):
        e = data[i]["sim"] - data[i]["real"]
        axes[0].plot(t, 1e3 * np.linalg.norm(e, axis=1), color=colors(k), lw=1.5, label=f"{name[i]} |sim - real|")
        axes[1].plot(t, 1e3 * np.linalg.norm(e[:, :2], axis=1), color=colors(k), lw=1.3, label=f"{name[i]} horizontal")
        axes[1].plot(t, 1e3 * e[:, 2], color=colors(k), lw=1.3, ls="--", label=f"{name[i]} vertical (z)")
    axes[0].set_ylabel("3D difference [mm]")
    axes[1].set_ylabel("difference [mm]")
    axes[1].axhline(0, color="k", lw=0.6)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Time from the take off [s]")
    fig.suptitle(f"{title}: simulation - real")
    fig.tight_layout()
    figs["difference"] = fig

    for fname, f in figs.items():
        for ext in ("pdf", "png"):
            f.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight", dpi=150)
    return figs


def compare(ids, real, sim_df, phases, out_dir, title, sphere_radius, target_id=None, meta=None, show=True,
            plot=True):
    """
    ids: drones of the log; real[i]: the logged flight of drone i (t from the
    take off, p, sp, target; flight_replay.load_flight); sim_df: the
    simulated flight (flight_replay.simulate_flight, same times); phases: the
    phase start times; target_id: the target drone (None: virtual target).
    plot False: metrics only (no figures).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, sp_cols, tg_cols = ["opti_x", "opti_y", "opti_z"], ["sp_x", "sp_y", "sp_z"], ["target_x", "target_y", "target_z"]
    chasers = [i for i in ids if i != target_id]
    sim = {}
    for i in ids:
        g = sim_df[sim_df["id"].astype(str) == i].sort_values("time_s")
        sim[i] = {"t": g["time_s"].to_numpy(float), "p": g[pos].to_numpy(float), "sp": g[sp_cols].to_numpy(float)}
    out_dir.mkdir(parents=True, exist_ok=True)

    # common time base: the samples of the first drone of the log, take off to the end
    t = real[ids[0]]["t"]
    t = t[(t >= phases["Take off"] - 0.5) & (t <= phases["EOF"] + 0.5)]
    r0 = real[ids[0]]
    target = interp_rows(r0["t"], r0["target"], t)
    data = {}
    for i in ids:
        data[i] = {"real": interp_rows(real[i]["t"], real[i]["p"], t),
                   "sim": interp_rows(sim[i]["t"], sim[i]["p"], t),
                   "sp": interp_rows(real[i]["t"], real[i]["sp"], t),
                   "sp_sim": interp_rows(sim[i]["t"], sim[i]["sp"], t)}

    res = {"title": title, "phases_s": phases, "sphere_radius_m": sphere_radius,
           "target": "virtual" if target_id is None else f"drone {target_id}",
           "flight_meta": "<log>_meta.json" if meta else None, "drones": {}}
    m_form = phase_mask(t, phases, FORMATION)
    m_settled = m_form & (t >= phase_end(phases, FORMATION) - SETTLED_WINDOW) if FORMATION in phases else m_form
    for i in ids:
        d = data[i]
        r = {"role": "chaser" if i in chasers else "target", "home": real[i]["p"][0].tolist()}
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
            r[f"diff_{ph}"] = q
        if i in chasers:
            for key in ("real", "sim"):
                dist = np.linalg.norm(d[key][m_settled] - target[m_settled], axis=1)
                dist = dist[np.isfinite(dist)]
                if len(dist):
                    r[f"encirclement_{key}"] = {"distance_to_target_mean_m": round(float(dist.mean()), 4),
                                                "radius_error_mean_m": round(float(dist.mean() - sphere_radius), 4),
                                                "radius_error_std_m": round(float(dist.std()), 4)}
        res["drones"][i] = r
    for key in ("real", "sim"):
        dmin = min_pair_distance(np.array([data[i][key][m_form] for i in ids]))
        if np.any(np.isfinite(dmin)):
            res[f"min_distance_between_drones_formation_{key}_m"] = round(float(np.nanmin(dmin)), 4)

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print_report(res)
    if not plot:
        print(f"Metrics saved to {out_dir / 'metrics.json'}")
        return res
    figs = plot_flight(ids, t, data, target, phases, sphere_radius, chasers, out_dir, title)

    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["all", *(p for p in PHASES if p in phases)]
    width = 0.8 / len(ids)
    for k, i in enumerate(ids):
        d = res["drones"][i]
        vals = [d.get(f"diff_{ph}", {}).get("rms_3d_mm", np.nan) for ph in labels]
        ax.bar(np.arange(len(labels)) + (k - (len(ids) - 1) / 2) * width, vals, width, color=colors(k),
               label=f"CF{int(i):02d}" + ("" if i in chasers else " (target)"))
    for x, ph in enumerate(labels):
        if ph in PHASE_COLORS:
            ax.axvspan(x - 0.45, x + 0.45, color=PHASE_COLORS[ph], alpha=0.6, lw=0, zorder=0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("RMS 3D difference sim - real [mm]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title(f"{title}: simulation vs real, per phase")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"summary.{ext}", bbox_inches="tight", dpi=150)
    figs["summary"] = fig
    print(f"Metrics and figures saved to {out_dir}")
    if show:
        plt.show()
    return res


def print_report(res):
    print("\nSimulation vs real flight: position difference (sim - real), setpoint tracking RMS")
    print(f"  {'drone':6s} {'phase':9s} {'RMS x/y/z [mm]':>22s} {'RMS 3D':>8s} {'max 3D':>8s} {'track real/sim':>16s}")
    for i, d in res["drones"].items():
        for ph in ("all", *PHASES):
            q = d.get(f"diff_{ph}")
            if q:
                x, y, z = q["rms_xyz_mm"]
                tr = f"{q.get('tracking_rms_real_mm', np.nan):7.1f}/{q.get('tracking_rms_sim_mm', np.nan):<7.1f}"
                print(f"  CF{int(i):02d}   {ph:9s} {x:7.1f} {y:7.1f} {z:7.1f}  {q['rms_3d_mm']:8.1f} "
                      f"{q['max_3d_mm']:8.1f}  {tr}")
    print(f"\n  encirclement, last {SETTLED_WINDOW:.0f} s of the Formation phase: distance to the target - sphere radius")
    for i, d in res["drones"].items():
        row = f"  CF{int(i):02d}"
        for key in ("real", "sim"):
            e = d.get(f"encirclement_{key}")
            if e:
                row += f"   {key}: {1e3 * e['radius_error_mean_m']:7.1f} mm (std {1e3 * e['radius_error_std_m']:.1f})"
        print(row)
    for key in ("real", "sim"):
        v = res.get(f"min_distance_between_drones_formation_{key}_m")
        if v is not None:
            print(f"  min distance between drones, Formation phase, {key}: {v:.3f} m")
