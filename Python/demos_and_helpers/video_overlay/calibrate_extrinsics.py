"""
Hand-eye offset T_bc between the camera rigid body (Motive marker
constellation) and the camera optical centre (OpenCV frame).

Motive gives T_wb (world <- rigid body); the renderer needs T_wc = T_wb T_bc.
T_bc is fixed (markers glued to the camera / its cage) and found from video
frames that see points of KNOWN world position (surveyed points):
  * loose markers / Crazyflies placed in the arena (floor AND on boxes at
    different heights; positions read in Motive or from the mocap log),
  * the drones themselves (tracked rigid bodies: --drones uses their logged
    positions at the frame time; their LED / body centre is the image point).
For each frame with >= 4 points, solvePnP gives T_wc,k and T_bc,k =
T_wb,k^-1 T_wc,k; the T_bc,k are averaged, then refined jointly over all the
correspondences (reprojection error, robust loss). Use frames from >= 3
clearly different camera positions/orientations.

Steps (SESSION needs intrinsics.json, sync.json, the mocap logs, the video):

  1. click: show frames, click the listed points in order (right click = not visible)
       python calibrate_extrinsics.py click SESSION --times 4.0 9.5 15.2 --points points.csv
       python calibrate_extrinsics.py click SESSION --times 4.0 9.5 --drones        # the drones on the floor
  2. solve: first estimate from the clicks
       python calibrate_extrinsics.py solve SESSION
  3. auto (optional, more correspondences): detects the blob nearest to each
     predicted point (bright: markers under a light / LEDs) in many frames, then solve again
       python calibrate_extrinsics.py auto SESSION --points points.csv --t0 2 --t1 30 --step 0.5
       python calibrate_extrinsics.py solve SESSION

points.csv: name,x,y,z (world Z-up frame, metres). Correspondences are
appended to SESSION/correspondences.csv (video_time_s,name,x,y,z,u,v,source).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import (Intrinsics, Session, inv_T, load_extrinsics, make_T, project,  # noqa: E402
                         save_extrinsics, video_frames, video_info)

CORR_COLUMNS = ["video_time_s", "name", "x", "y", "z", "u", "v", "source"]


def load_points(session: Session, points_csv, use_drones, t_video):
    """Surveyed points visible at video time t: [(name, xyz)]."""
    pts = []
    if points_csv:
        df = pd.read_csv(points_csv)
        pts += [(str(r["name"]), np.array([r["x"], r["y"], r["z"]], float)) for _, r in df.iterrows()]
    if use_drones:
        scale, offset = session.sync()
        pos = session.real_tracks().at(scale * t_video + offset)
        pts += [(f"drone{i}", p) for i, p in pos.items() if np.all(np.isfinite(p))]
    return pts


def frame_at(video, t):
    for _, tt, fr in video_frames(video, start_s=max(0.0, t - 0.02)):
        return tt, fr
    raise ValueError(f"No frame at {t} s")


def append_corr(session: Session, rows):
    path = session.root / "correspondences.csv"
    df = pd.DataFrame(rows, columns=CORR_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"{len(df)} correspondences appended to {path}")


# ============================================================
# CLICK
# ============================================================

def cmd_click(args):
    import matplotlib.pyplot as plt

    s = Session(args.session)
    intr = Intrinsics.load(s.path("intrinsics"))
    have_guess = s.path("extrinsics").exists()
    rows = []
    for t in args.times:
        tt, fr = frame_at(s.path("video"), t)
        pts = load_points(s, args.points, args.drones, tt)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if have_guess:   # current estimate, to help find the points
            scale, offset = s.sync()
            T_wb = s.camera_track().T_wb(scale * tt + offset)
            if T_wb is not None:
                uv, valid, _ = project(np.array([p for _, p in pts]), T_wb @ load_extrinsics(s.path("extrinsics")),
                                       intr.scaled_to(fr.shape[1], fr.shape[0]))
                for (name, _), q, ok in zip(pts, uv, valid):
                    if ok:
                        ax.plot(*q, "c+", ms=12)
                        ax.text(q[0] + 5, q[1] - 5, name, color="c")
        for name, p in pts:
            ax.set_title(f"t = {tt:.2f} s: click '{name}' at {np.round(p, 3)}  (right click: not visible)")
            fig.canvas.draw()
            c = plt.ginput(1, timeout=0, mouse_add=1, mouse_pop=None, mouse_stop=3)
            if c:
                rows.append([tt, name, *p, c[0][0], c[0][1], "click"])
                ax.plot(*c[0], "r+", ms=14)
        plt.close(fig)
    append_corr(s, rows)


# ============================================================
# AUTO
# ============================================================

def detect_near(gray, pred, radius, polarity="bright"):
    """Centroid of the blob nearest to pred inside a window of `radius` px, or None."""
    h, w = gray.shape
    u, v = int(round(pred[0])), int(round(pred[1]))
    x0, x1, y0, y1 = max(0, u - radius), min(w, u + radius + 1), max(0, v - radius), min(h, v + radius + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    win = gray[y0:y1, x0:x1].astype(np.float32)
    if polarity == "dark":
        win = 255 - win
    thr = max(np.percentile(win, 98), np.median(win) + 40)
    n, lab, stats, cents = cv2.connectedComponentsWithStats((win >= thr).astype(np.uint8))
    best, bd = None, np.inf
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 2:
            continue
        d = np.hypot(*(cents[k] + (x0, y0) - pred))
        if d < bd:
            best, bd = k, d
    if best is None or bd > radius:
        return None
    # intensity-weighted centroid of that component (sub-pixel)
    ys, xs = np.nonzero(lab == best)
    wgt = win[ys, xs] - thr + 1.0
    return np.array([x0 + np.sum(xs * wgt) / wgt.sum(), y0 + np.sum(ys * wgt) / wgt.sum()])


def cmd_auto(args):
    s = Session(args.session)
    info = video_info(s.path("video"))
    intr = Intrinsics.load(s.path("intrinsics")).scaled_to(info["width"], info["height"])
    T_bc = load_extrinsics(s.path("extrinsics"))
    scale, offset = s.sync()
    cam = s.camera_track()
    rows = []
    times = np.arange(args.t0, args.t1 + 1e-9, args.step)
    k_next = 0
    for _, t, fr in video_frames(s.path("video"), args.t0, args.t1 + 0.1):
        if k_next >= len(times) or t + 1e-6 < times[k_next]:
            continue
        k_next += 1
        T_wb = cam.T_wb(scale * t + offset)
        if T_wb is None:
            continue
        pts = load_points(s, args.points, args.drones, t)
        uv, valid, _ = project(np.array([p for _, p in pts]), T_wb @ T_bc, intr, margin=0.0)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for (name, p), q, ok in zip(pts, uv, valid):
            if not ok:
                continue
            c = detect_near(gray, q, args.radius, args.polarity)
            if c is not None:
                rows.append([t, name, *p, c[0], c[1], "auto"])
    print(f"Auto detection: {len(rows)} correspondences in {len(times)} frames")
    append_corr(s, rows)


# ============================================================
# SOLVE
# ============================================================

def solve(corr: pd.DataFrame, cam, scale, offset, intr: Intrinsics, T_init=None, verbose=True):
    """T_bc from the correspondences (PnP per frame, average, joint robust refinement)."""
    groups = []
    for t, g in corr.groupby("video_time_s"):
        T_wb = cam.T_wb(scale * t + offset)
        if T_wb is None:
            continue
        groups.append((t, T_wb, g[["x", "y", "z"]].to_numpy(float), g[["u", "v"]].to_numpy(float)))
    if not groups:
        raise RuntimeError("No correspondence at a time with a valid camera pose")

    # per-frame PnP
    ests = []
    for t, T_wb, P, uv in groups:
        if len(P) < 4:
            continue
        ok, rv, tv = cv2.solvePnP(P, uv, intr.K, intr.dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(P, uv, intr.K, intr.dist, rv, tv)
        T_cw = make_T(cv2.Rodrigues(rv)[0], tv)
        ests.append(inv_T(T_wb) @ inv_T(T_cw))
    if ests:
        R0 = Rotation.from_matrix(np.array([T[:3, :3] for T in ests])).mean().as_matrix()
        t0 = np.median(np.array([T[:3, 3] for T in ests]), axis=0)
        T0 = make_T(R0, t0)
        spread = np.std(np.array([T[:3, 3] for T in ests]), axis=0)
        if verbose:
            print(f"PnP on {len(ests)} frames: T_bc translation {np.round(t0, 4)} m (std {np.round(spread, 4)})")
    elif T_init is not None:
        T0 = T_init
    else:
        raise RuntimeError("No frame with >= 4 correspondences and no initial extrinsics")

    P_all = np.vstack([P for _, _, P, _ in groups])
    uv_all = np.vstack([uv for _, _, _, uv in groups])
    T_wb_all = [T for _, T, P, _ in groups for _ in range(len(P))]
    R_wb = np.array([T[:3, :3] for T in T_wb_all])
    t_wb = np.array([T[:3, 3] for T in T_wb_all])
    # points in the rigid-body frame (fixed during the optimization)
    P_b = np.einsum("nji,nj->ni", R_wb, P_all - t_wb)

    def residual(x):
        R_bc = Rotation.from_rotvec(x[:3]).as_matrix()
        pc = (P_b - x[3:]) @ R_bc            # R_bc^T (p_b - t_bc), row form
        img, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intr.K, intr.dist)
        r = (img.reshape(-1, 2) - uv_all)
        r[pc[:, 2] <= 0] = 1e3
        return r.ravel()

    x0 = np.r_[Rotation.from_matrix(T0[:3, :3]).as_rotvec(), T0[:3, 3]]
    sol = least_squares(residual, x0, loss="soft_l1", f_scale=2.0)
    T_bc = make_T(Rotation.from_rotvec(sol.x[:3]).as_matrix(), sol.x[3:])
    r = residual(sol.x).reshape(-1, 2)
    e = np.linalg.norm(r, axis=1)
    inl = e < max(5.0, 3 * np.median(e))
    stats = {"n_correspondences": int(len(e)), "n_frames": len(groups), "n_pnp_frames": len(ests),
             "rms_px": float(np.sqrt(np.mean(e[inl] ** 2))), "median_px": float(np.median(e)),
             "inlier_fraction": float(inl.mean())}
    # uncertainty from the Jacobian (inliers, pixel noise = rms)
    try:
        J = sol.jac
        cov = np.linalg.inv(J.T @ J) * stats["rms_px"] ** 2
        sd = np.sqrt(np.diag(cov))
        stats["std_rot_deg"] = np.degrees(sd[:3]).round(4).tolist()
        stats["std_trans_mm"] = (1e3 * sd[3:]).round(3).tolist()
    except np.linalg.LinAlgError:
        pass
    if verbose:
        print(f"Joint refinement: {stats['n_correspondences']} correspondences, {stats['n_frames']} frames, "
              f"reprojection RMS {stats['rms_px']:.2f} px (median {stats['median_px']:.2f}), "
              f"inliers {100 * stats['inlier_fraction']:.0f} %")
        print(f"T_bc: t = {np.round(T_bc[:3, 3] * 1e3, 1)} mm, "
              f"R = rotvec {np.round(np.degrees(sol.x[:3]), 2)} deg")
    return T_bc, stats


def cmd_solve(args):
    s = Session(args.session)
    info = video_info(s.path("video"))
    intr = Intrinsics.load(s.path("intrinsics")).scaled_to(info["width"], info["height"])
    corr = pd.read_csv(s.root / "correspondences.csv")
    if args.source:
        corr = corr[corr["source"].isin(args.source)]
    scale, offset = s.sync()
    T_init = load_extrinsics(s.path("extrinsics")) if s.path("extrinsics").exists() else None
    T_bc, stats = solve(corr, s.camera_track(), scale, offset, intr, T_init)
    save_extrinsics(s.path("extrinsics"), T_bc, **stats)
    print(f"Saved to {s.path('extrinsics')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("click")
    c.add_argument("session")
    c.add_argument("--times", type=float, nargs="+", required=True, help="video times [s] of the frames")
    c.add_argument("--points", help="points.csv (name,x,y,z)")
    c.add_argument("--drones", action="store_true", help="also the drones (logged positions)")
    a = sub.add_parser("auto")
    a.add_argument("session")
    a.add_argument("--points")
    a.add_argument("--drones", action="store_true")
    a.add_argument("--t0", type=float, required=True)
    a.add_argument("--t1", type=float, required=True)
    a.add_argument("--step", type=float, default=0.5)
    a.add_argument("--radius", type=int, default=25, help="search radius [px] around the predicted point")
    a.add_argument("--polarity", choices=["bright", "dark"], default="bright")
    so = sub.add_parser("solve")
    so.add_argument("session")
    so.add_argument("--source", nargs="*", help="only these correspondence sources (click / auto)")
    args = ap.parse_args()
    {"click": cmd_click, "auto": cmd_auto, "solve": cmd_solve}[args.cmd](args)


if __name__ == "__main__":
    main()
