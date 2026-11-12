"""
Virtual camera overlay: draws the REAL (OptiTrack) and the SIMULATED drone
trajectories over the video, as seen by a virtual camera that follows the
tracked camera (pose T_wc(t) = T_wb(t) T_bc, intrinsics K + distortion).

For each video frame (video time t_v -> mocap time t_m = scale t_v + offset):
  * real drones (green): current position (ring sized like a 12 cm drone) and
    the trail of the last --trail seconds,
  * simulated drones (orange, Crazy_Flies/flight_sim.py on the same flight:
    same start state, same flight code, Crazyflie plant), same drawing, plus
    a thin line real -> sim (the instantaneous difference),
  * the virtual target (magenta) and, with --grid, the arena floor grid
    (a direct visual check of the calibration: it must stick to the floor),
  * a legend / clock / current error box.
Also writes a per-frame CSV (3D and pixel error real vs sim).

    python render_overlay.py SESSION                       # -> SESSION/overlay.mp4, overlay_metrics.csv
    python render_overlay.py SESSION --t0 20 --t1 60 --grid --trail 4 --out talk_clip.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import (COLOR_REAL, COLOR_SIM, COLOR_TARGET, Intrinsics, Session, VideoWriter,  # noqa: E402
                         draw_polyline, draw_world_segments, floor_grid_segments, load_extrinsics, marker_radius,
                         project, put_label, video_frames, video_info)


def arena_box_segments(lo=(-2.5, -2.5, 0.0), hi=(2.5, 2.5, 2.5)):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    segs = [((*c[k], z0), (*c[(k + 1) % 4], z0)) for k in range(4)]
    segs += [((*c[k], z1), (*c[(k + 1) % 4], z1)) for k in range(4)]
    segs += [((*c[k], z0), (*c[k], z1)) for k in range(4)]
    return segs


def draw_drone(img, uv, depth, intr, color, style, label=None):
    u, v = int(round(uv[0])), int(round(uv[1]))
    r = marker_radius(intr, depth)
    if style == "ring":
        cv2.circle(img, (u, v), r, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(img, (u, v), r, color, 2, cv2.LINE_AA)
    else:   # diamond (simulation)
        pts = np.array([(u, v - r), (u + r, v), (u, v + r), (u - r, v)], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        cv2.circle(img, (u, v), 2, color, -1, cv2.LINE_AA)
    if label:
        put_label(img, label, (u + r + 3, v - r), color, 0.5)


def legend(img, lines, scale=0.6, corner="tl"):
    """Semi-transparent legend box; corner: tl (top left) or bl (bottom left)."""
    h = 26 * len(lines) + 12
    w = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t, _ in lines)) + 50
    y0 = 10 if corner == "tl" else img.shape[0] - 10 - h
    over = img.copy()
    cv2.rectangle(over, (10, y0), (10 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.6, img, 0.4, 0, img)
    for k, (text, color) in enumerate(lines):
        y = y0 + 24 + 26 * k
        if color is not None:
            cv2.line(img, (20, y - 6), (40, y - 6), color, 3, cv2.LINE_AA)
        cv2.putText(img, text, (48 if color is not None else 20, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


class OverlayRenderer:
    """Draws the real/sim drones of mocap time t_m into a frame seen from T_wc."""

    def __init__(self, intr, real, sim=None, target=None, trail=3.0, grid=False, labels=True):
        self.intr, self.real, self.sim, self.target = intr, real, sim, target
        self.trail, self.grid, self.labels = trail, grid, labels
        self.segs = floor_grid_segments() + arena_box_segments() if grid else []

    def trail_uv(self, tracks, i, t_m, T_wc):
        ts = np.arange(t_m - self.trail, t_m + 1e-9, 0.04)
        P = tracks.sample(i, ts)
        ok = np.all(np.isfinite(P), axis=1)
        uv = np.full((len(ts), 2), np.nan)
        if ok.any():
            q, valid, _ = project(P[ok], T_wc, self.intr)
            q[~valid] = np.nan
            uv[ok] = q
        return uv

    def draw(self, img, t_m, T_wc):
        if self.segs:
            draw_world_segments(img, self.segs, T_wc, self.intr, (230, 230, 230), 1)
        real_now = self.real.at(t_m)
        sim_now = self.sim.at(t_m) if self.sim is not None else {}
        if self.target is not None:
            draw_polyline(img, self.trail_uv(self.target, "target", t_m, T_wc), COLOR_TARGET, 1)
            p = self.target.at(t_m)["target"]
            if np.all(np.isfinite(p)):
                uv, ok, _ = project(p[None], T_wc, self.intr, margin=0.0)
                if ok[0]:
                    cv2.drawMarker(img, tuple(np.round(uv[0]).astype(int)), COLOR_TARGET, cv2.MARKER_TILTED_CROSS,
                                   18, 2, cv2.LINE_AA)
        rows = {}
        for tracks, now, color, style in ((self.sim, sim_now, COLOR_SIM, "diamond"),
                                          (self.real, real_now, COLOR_REAL, "ring")):
            if tracks is None:
                continue
            for i in tracks.ids:
                draw_polyline(img, self.trail_uv(tracks, i, t_m, T_wc), color, 2)
        # current positions, far to near
        items = []
        for tracks, now, color, style, kind in ((self.sim, sim_now, COLOR_SIM, "diamond", "sim"),
                                                (self.real, real_now, COLOR_REAL, "ring", "real")):
            for i, p in now.items():
                if np.all(np.isfinite(p)):
                    uv, ok, z = project(p[None], T_wc, self.intr, margin=0.0)
                    if ok[0]:
                        items.append((z[0], uv[0], color, style, kind, i))
        for z, uv, color, style, kind, i in sorted(items, key=lambda it: -it[0]):
            draw_drone(img, uv, z, self.intr, color, style,
                       f"CF{int(i):02d}" if (self.labels and kind == "real" and i.isdigit()) else None)
        # real -> sim difference
        pix = {(k, i): uv for z, uv, c, s, k, i in items}
        for i in real_now:
            if ("real", i) in pix and ("sim", i) in pix:
                cv2.line(img, tuple(np.round(pix[("real", i)]).astype(int)),
                         tuple(np.round(pix[("sim", i)]).astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
            e3 = np.linalg.norm(real_now[i] - sim_now[i]) if i in sim_now else np.nan
            epx = (np.linalg.norm(pix[("real", i)] - pix[("sim", i)])
                   if ("real", i) in pix and ("sim", i) in pix else np.nan)
            rows[i] = (e3, epx)
        return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--out", help="output mp4 (default SESSION/overlay.mp4)")
    ap.add_argument("--t0", type=float, default=0.0, help="video start [s]")
    ap.add_argument("--t1", type=float, default=None, help="video end [s]")
    ap.add_argument("--trail", type=float, default=3.0, help="trail length [s]")
    ap.add_argument("--grid", action="store_true", help="draw the floor grid and the 5x5x2.5 m flight volume")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--width", type=int, help="output width [px] (default: video width)")
    ap.add_argument("--title", default="Real (OptiTrack) vs simulation")
    args = ap.parse_args()

    s = Session(args.session)
    video = s.path("video")
    info = video_info(video)
    intr = Intrinsics.load(s.path("intrinsics")).scaled_to(info["width"], info["height"])
    T_bc = load_extrinsics(s.path("extrinsics"))
    scale, offset = s.sync()
    cam = s.camera_track()
    real = s.real_tracks()
    sim = target = None
    if not args.no_sim:
        res, sinfo = s.sim_tracks()
        if res is not None:
            sim, target = res
            print(f"Simulation placed in mocap time: flight log offset {sinfo['log_offset_s']:.3f} s "
                  f"(match residual {sinfo['log_offset_residual_m'] * 100:.1f} cm), Formation phase "
                  f"{sinfo['formation_start_mocap_s']:.2f}-{sinfo['formation_end_mocap_s']:.2f} s mocap time")
        else:
            print("No simulation (session.json: flight_log / sim): real trajectories only")

    out = Path(args.out) if args.out else s.root / "overlay.mp4"
    if not out.is_absolute() and args.out:
        out = s.root / out
    size = (info["width"], info["height"])
    out_size = size if not args.width else (args.width, int(round(info["height"] * args.width / info["width"])))
    writer = VideoWriter(out, info["fps"] or 30.0, out_size)
    rend = OverlayRenderer(intr, real, sim, target, args.trail, args.grid)
    metrics = []
    n = 0
    for k, t, frame in video_frames(video, args.t0, args.t1):
        t_m = scale * t + offset
        T_wb = cam.T_wb(t_m)
        img = frame.copy()
        if T_wb is None:
            put_label(img, "no camera pose", (20, info["height"] - 20), (0, 0, 255), 0.7, 2)
        else:
            rows = rend.draw(img, t_m, T_wb @ T_bc)
            for i, (e3, epx) in rows.items():
                metrics.append([k, t, t_m, i, e3, epx])
            errs = [e for e, _ in rows.values() if np.isfinite(e)]
        lines = [(args.title, None), ("real (OptiTrack)", COLOR_REAL)]
        if sim is not None:
            lines.append(("simulation", COLOR_SIM))
        if target is not None:
            lines.append(("virtual target", COLOR_TARGET))
        txt = f"t = {t_m - real.span()[0]:5.1f} s"
        if T_wb is not None and sim is not None and errs:
            txt += f"   |real - sim| = {100 * np.mean(errs):4.1f} cm"
        lines.append((txt, None))
        legend(img, lines)
        if out_size != size:
            img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
        writer.write(img)
        n += 1
    writer.close()
    mpath = out.with_name(out.stem + "_metrics.csv")
    pd.DataFrame(metrics, columns=["frame", "video_time_s", "mocap_time_s", "id", "err3d_m", "err_px"]).to_csv(
        mpath, index=False)
    print(f"Overlay: {n} frames -> {out}\nPer-frame errors -> {mpath}")


if __name__ == "__main__":
    main()
