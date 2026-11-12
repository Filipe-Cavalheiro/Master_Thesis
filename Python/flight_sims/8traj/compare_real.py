"""
Real vs simulated figure-8 flight, over the whole flight (called by main.py --IRL_log):
position.pdf (x, y, z real / sim / setpoint), top_view.pdf, difference.pdf
(sim - real per axis and |e|) and metrics.json (RMS per phase, max, tracking
error) in <run folder>/comparison/. The Main Controller comparison of the
thesis simulation (real_sim_comparison_*) is drawn by Animation_display.create_figures.
"""

import json

import numpy as np

from flight_code_funcs import PHASES, POS_COLS


PHASE_COLORS = {"Take off": "#cfe8ff", "Main Controller": "#d9f2d0", "Return to origin": "#f3e1f7",
                "Land": "#ffe0c7"}


def compare(log_path, sim_df, events, phases, drone, out_dir):
    """events: the setpoints of the real drone (resolve() with its logged return start)."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t = drone["t"]
    P = drone["p"]
    S = np.column_stack([np.interp(t, sim_df["time_s"], sim_df[c], left=np.nan, right=np.nan) for c in POS_COLS])
    et = np.array([e[0] for e in events])
    ep = np.array([e[2] for e in events])
    k = np.searchsorted(et, t, side="right") - 1
    SP = np.where((k >= 0)[:, None], ep[np.clip(k, 0, None)], np.nan)
    E = S - P

    order = [p for p in PHASES if p in phases]
    ends = {p: (phases[order[i + 1]] if i + 1 < len(order) else t[-1]) for i, p in enumerate(order)}

    def stats(m):
        e = E[m]
        e = e[np.all(np.isfinite(e), 1)]
        if not len(e):
            return None
        n = np.linalg.norm(e, axis=1)
        tr_r = np.linalg.norm((P - SP)[m], axis=1)
        tr_s = np.linalg.norm((S - SP)[m], axis=1)
        return {"rms_xyz_m": np.sqrt(np.mean(e ** 2, 0)).round(4).tolist(), "rms_3d_m": round(float(np.sqrt(np.mean(n ** 2))), 4),
                "mean_3d_m": round(float(n.mean()), 4), "p95_3d_m": round(float(np.percentile(n, 95)), 4),
                "max_3d_m": round(float(n.max()), 4),
                "tracking_rms_real_m": round(float(np.sqrt(np.nanmean(tr_r ** 2))), 4),
                "tracking_rms_sim_m": round(float(np.sqrt(np.nanmean(tr_s ** 2))), 4)}

    metrics = {"log": str(log_path), "sim_minus_real": {"whole flight": stats(t >= phases["Take off"])}}
    for p in order:
        s = stats((t >= phases[p]) & (t < ends[p]))
        if s:
            metrics["sim_minus_real"][p] = s
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def shade(ax):
        for p in order:
            ax.axvspan(phases[p], ends[p], color=PHASE_COLORS[p], lw=0, zorder=0)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=150)

    C_R, C_S = (60 / 255, 160 / 255, 80 / 255), (255 / 255, 140 / 255, 30 / 255)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for a, ax in enumerate(axs):
        shade(ax)
        ax.plot(t, SP[:, a], "--", color="0.4", lw=1.2, label="setpoint")
        ax.plot(t, P[:, a], color=C_R, lw=1.8, label="real")
        ax.plot(t, S[:, a], color=C_S, lw=1.5, label="sim")
        ax.set_ylabel(f"{'xyz'[a]} [m]")
        ax.grid(True, alpha=0.4)
    axs[0].legend(loc="upper right", ncol=3)
    axs[-1].set_xlabel("Time [s]")
    save(fig, "position")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    m = (t >= phases.get("Main Controller", phases["Take off"]))
    ax.plot(SP[m, 0], SP[m, 1], "--", color="0.4", lw=1.2, label="setpoint")
    ax.plot(P[m, 0], P[m, 1], color=C_R, lw=1.6, label="real")
    ax.plot(S[m, 0], S[m, 1], color=C_S, lw=1.4, label="sim")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    save(fig, "top_view")

    fig, ax = plt.subplots(figsize=(10, 4))
    shade(ax)
    for a, col in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        ax.plot(t, 100 * E[:, a], color=col, lw=1.2, label=f"$e_{'xyz'[a]}$")
    ax.plot(t, 100 * np.linalg.norm(E, axis=1), "k", lw=1.8, label=r"$\|e\|$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("sim - real [cm]")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", ncol=4)
    save(fig, "difference")

    w = metrics["sim_minus_real"]
    print("sim - real (3D RMS / max):")
    for p, s in w.items():
        print(f"  {p:18s} {100 * s['rms_3d_m']:6.1f} / {100 * s['max_3d_m']:6.1f} cm   "
              f"tracking RMS real {100 * s['tracking_rms_real_m']:5.1f} cm, sim {100 * s['tracking_rms_sim_m']:5.1f} cm")
    print(f"Comparison saved to {out_dir}")
