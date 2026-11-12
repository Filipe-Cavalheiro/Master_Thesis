"""
"Motion through time" (stroboscopic) composite + real-vs-sim error.

From a STATIC camera clip (tripod; the camera rigid body is still used for
its pose):
  1. background = per-pixel median of frames spread over the clip (the moving
     drones vanish),
  2. at N instants, the real drones are cut out of the frame by background
     difference (|frame - background| > thr, cleaned, gated to discs around
     the projected mocap positions so people/shadows are ignored) and pasted
     on the background in time order,
  3. the simulated drones of the same instants are drawn as ghost markers
     (orange diamonds) with their full trails, the mocap trails in green,
     instants numbered with their time.
Outputs (SESSION/strobe/ or --out-dir):
  strobe.png           annotated composite (paper figure / slide)
  strobe_clean.png     composite only (annotate it yourself)
  strobe_overlay.png   composite + trails/ghosts but no legend / labels
  error.pdf / .png     3D distance real-sim per drone over the Formation phase
  metrics.json         RMS / mean / max / per axis per drone and overall,
                       and the pixel error at the strobe instants

    python stroboscopic.py SESSION                      # Formation phase, 8 instants
    python stroboscopic.py SESSION --n 10 --t0 30 --t1 50   # video times
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import (COLOR_REAL, COLOR_SIM, Intrinsics, Session, draw_polyline, load_extrinsics,  # noqa: E402
                         marker_radius, project, put_label, video_frames, video_info)
from render_overlay import legend  # noqa: E402

# categorical slots (dataviz reference palette, light mode) for the per-drone error plot
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def error_metrics(real, sim, t0, t1, dt=0.02):
    """3D position error real - sim per drone over mocap times [t0, t1]."""
    ts = np.arange(t0, t1, dt)
    out, series = {}, {}
    all_e = []
    for i in real.ids:
        if i not in sim.data:
            continue
        pr, ps = real.sample(i, ts), sim.sample(i, ts)
        ok = np.all(np.isfinite(pr) & np.isfinite(ps), axis=1)
        if ok.sum() < 2:
            continue
        d = ps[ok] - pr[ok]
        e = np.linalg.norm(d, axis=1)
        all_e.append(e)
        series[i] = (ts[ok] - t0, e)
        out[i] = {"rms_m": float(np.sqrt(np.mean(e ** 2))), "mean_m": float(e.mean()), "max_m": float(e.max()),
                  "p95_m": float(np.percentile(e, 95)),
                  "rms_xyz_m": np.sqrt(np.mean(d ** 2, axis=0)).round(4).tolist(), "n": int(ok.sum())}
    e = np.concatenate(all_e)
    out["all"] = {"rms_m": float(np.sqrt(np.mean(e ** 2))), "mean_m": float(e.mean()), "max_m": float(e.max()),
                  "p95_m": float(np.percentile(e, 95)), "duration_s": float(t1 - t0)}
    return out, series


def plot_error(series, metrics, path_stem, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for k, (i, (t, e)) in enumerate(series.items()):
        m = metrics[i]
        ax.plot(t, 100 * e, color=SERIES[k % len(SERIES)], lw=1.5,
                label=f"CF{int(i):02d}  RMS {100 * m['rms_m']:.1f} cm, max {100 * m['max_m']:.1f} cm"
                if i.isdigit() else i)
    a = metrics["all"]
    ax.axhline(100 * a["rms_m"], color="#52514e", lw=1, ls="--", label=f"all drones  RMS {100 * a['rms_m']:.1f} cm")
    ax.set_xlabel("time since the start of the Formation phase [s]")
    ax.set_ylabel("|p_real - p_sim| [cm]")
    ax.set_ylim(0, 100 * max(e.max() for _, e in series.values()) * 1.45)   # headroom for the legend
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def median_background(video, t0, t1, n=60):
    info = video_info(video)
    fps = info["fps"] or 30.0
    want = set(np.linspace(t0 * fps, t1 * fps, n).round().astype(int))
    frames = [fr for k, _, fr in video_frames(video, t0, t1 + 0.1) if k in want]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--t0", type=float, help="video start of the composite [s] (default: Formation phase)")
    ap.add_argument("--t1", type=float, help="video end [s]")
    ap.add_argument("--n", type=int, default=8, help="number of instants")
    ap.add_argument("--thr", type=int, default=30, help="background difference threshold (0-255)")
    ap.add_argument("--gate", type=float, default=2.5, help="cut-out radius / drone radius (0: no gating)")
    ap.add_argument("--bg-margin", type=float, default=5.0,
                    help="background frames also taken this long [s] before/after the composite window")
    ap.add_argument("--legend", choices=["tl", "bl"], default="bl", help="legend corner (top/bottom left)")
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    s = Session(args.session)
    video = s.path("video")
    info = video_info(video)
    intr = Intrinsics.load(s.path("intrinsics")).scaled_to(info["width"], info["height"])
    T_bc = load_extrinsics(s.path("extrinsics"))
    scale, offset = s.sync()
    cam = s.camera_track()
    real = s.real_tracks()
    res, sinfo = s.sim_tracks()
    sim = res[0] if res is not None else None
    out_dir = Path(args.out_dir) if args.out_dir else s.root / "strobe"
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = info["frames"] / (info["fps"] or 30.0)

    to_video = lambda tm: (tm - offset) / scale   # noqa: E731
    if args.t0 is None or args.t1 is None:
        if sim is None:
            sys.exit("No simulation: give --t0 and --t1")
        t0 = to_video(sinfo["formation_start_mocap_s"]) if args.t0 is None else args.t0
        t1 = to_video(sinfo["formation_end_mocap_s"]) if args.t1 is None else args.t1
    else:
        t0, t1 = args.t0, args.t1
    t0, t1 = max(t0, 0.0), min(t1, duration - 0.05)
    tm0, tm1 = scale * t0 + offset, scale * t1 + offset

    dp, dr = cam.motion(tm0, tm1)
    print(f"Composite window: video {t0:.2f}-{t1:.2f} s, camera motion {dp * 100:.1f} cm / {dr:.2f} deg")
    if dp > 0.02 or dr > 0.5:
        print("WARNING: the camera is not static over the window: the composite will be blurred")
    T_wc = cam.T_wb(0.5 * (tm0 + tm1)) @ T_bc

    bg = median_background(video, max(0.0, t0 - args.bg_margin), min(duration - 0.05, t1 + args.bg_margin))
    fps = info["fps"] or 30.0
    instants = np.linspace(t0, t1, args.n)
    want = {int(round(t * fps)): t for t in instants}
    frames = {k: (t, fr) for k, t, fr in video_frames(video, t0 - 0.1, t1 + 0.1) if k in want}

    comp = bg.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    strobe_pts = []      # (instant number, t_m, id, uv_real, uv_sim, depth)
    for n, k in enumerate(sorted(frames)):
        t, fr = frames[k]
        tm = scale * t + offset
        diff = np.abs(fr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
        mask = (diff > args.thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        gate = np.zeros_like(mask)
        pr, ps = real.at(tm), (sim.at(tm) if sim is not None else {})
        for i, p in pr.items():
            if not np.all(np.isfinite(p)):
                continue
            uv, ok, z = project(p[None], T_wc, intr, margin=0.0)
            if not ok[0]:
                continue
            r = marker_radius(intr, z[0])
            cv2.circle(gate, tuple(np.round(uv[0]).astype(int)), int(args.gate * r), 1, -1)
            q = np.full(2, np.nan)
            if i in ps and np.all(np.isfinite(ps[i])):
                qs, oks, _ = project(ps[i][None], T_wc, intr, margin=0.0)
                if oks[0]:
                    q = qs[0]
            strobe_pts.append((n, tm, i, uv[0], q, z[0]))
        if args.gate > 0:
            mask &= gate
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
        comp = (soft * fr + (1 - soft) * comp).astype(np.uint8)
    cv2.imwrite(str(out_dir / "strobe_clean.png"), comp)

    # trails over the window + ghosts
    ann = comp.copy()
    ts = np.arange(tm0, tm1, 0.02)
    for tracks, color in ((sim, COLOR_SIM), (real, COLOR_REAL)):
        if tracks is None:
            continue
        for i in tracks.ids:
            P = tracks.sample(i, ts)
            okp = np.all(np.isfinite(P), axis=1)
            uv = np.full((len(ts), 2), np.nan)
            if okp.any():
                q, valid, _ = project(P[okp], T_wc, intr)
                q[~valid] = np.nan
                uv[okp] = q
            draw_polyline(ann, uv, color, 2, fade=False)
    for n, tm, i, uvr, uvs, z in strobe_pts:
        r = marker_radius(intr, z)
        if np.all(np.isfinite(uvs)):
            u, v = np.round(uvs).astype(int)
            pts = np.array([(u, v - r), (u + r, v), (u, v + r), (u - r, v)], np.int32)
            over = ann.copy()
            cv2.fillPoly(over, [pts], COLOR_SIM, cv2.LINE_AA)
            cv2.addWeighted(over, 0.35, ann, 0.65, 0, ann)
            cv2.polylines(ann, [pts], True, COLOR_SIM, 2, cv2.LINE_AA)
        cv2.circle(ann, tuple(np.round(uvr).astype(int)), r, COLOR_REAL, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "strobe_overlay.png"), ann)
    # one time label per instant, above its highest drone, skipped where it would collide
    placed = []
    for n in sorted({p[0] for p in strobe_pts}):
        same = [p for p in strobe_pts if p[0] == n]
        top = min(same, key=lambda p: p[3][1])
        org = np.array([top[3][0] - 18, top[3][1] - marker_radius(intr, top[5]) - 8])
        if all(np.linalg.norm(org - q) > 55 for q in placed):
            put_label(ann, f"{top[1] - tm0:.1f}s", tuple(org.astype(int)), (255, 255, 255), 0.5)
            placed.append(org)
    lines = [("Motion through time: real vs simulation", None),
             ("real drone (video cut-out, OptiTrack trail)", COLOR_REAL)]
    if sim is not None:
        lines.append(("simulated drone (ghost, trail)", COLOR_SIM))
    lines.append((f"{args.n} instants over {t1 - t0:.1f} s", None))
    legend(ann, lines, corner=args.legend)
    cv2.imwrite(str(out_dir / "strobe.png"), ann)

    result = {"window_video_s": [t0, t1], "window_mocap_s": [tm0, tm1], "camera_motion_cm_deg": [dp * 100, dr],
              "instants_video_s": instants.round(3).tolist()}
    if sim is not None:
        m, series = error_metrics(real, sim, sinfo["formation_start_mocap_s"], sinfo["formation_end_mocap_s"])
        px = [float(np.linalg.norm(uvr - uvs)) for _, _, _, uvr, uvs, _ in strobe_pts if np.all(np.isfinite(uvs))]
        m["strobe_pixel_error"] = {"mean_px": float(np.mean(px)) if px else None,
                                   "max_px": float(np.max(px)) if px else None}
        result["error_formation_phase"] = m
        result["sim_alignment"] = sinfo
        plot_error(series, m, out_dir / "error", "Real (OptiTrack) vs simulated position, Formation phase")
        a = m["all"]
        print(f"Real vs sim over the Formation phase ({a['duration_s']:.1f} s): RMS {100 * a['rms_m']:.1f} cm, "
              f"mean {100 * a['mean_m']:.1f} cm, max {100 * a['max_m']:.1f} cm")
        for i in series:
            print(f"  CF{i}: RMS {100 * m[i]['rms_m']:.1f} cm, max {100 * m[i]['max_m']:.1f} cm, "
                  f"RMS xyz {np.round(100 * np.array(m[i]['rms_xyz_m']), 1).tolist()} cm")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"Composite and metrics -> {out_dir}")


if __name__ == "__main__":
    main()
