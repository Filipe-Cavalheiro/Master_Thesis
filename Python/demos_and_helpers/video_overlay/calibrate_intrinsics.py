"""
Camera intrinsics (pinhole K + radial/tangential distortion) from a
checkerboard, with OpenCV.

Film / photograph a printed checkerboard (flat, rigid: glued on cardboard or
shown on a laptop/tablet screen) from 20-40 views: all over the image
(corners included), tilted up to ~45 deg, at several distances. Use EXACTLY
the video mode of the flight (resolution, fps, lens, stabilization OFF,
fixed zoom/focus): the intrinsics depend on it.

    python calibrate_intrinsics.py --video calib.mp4 --board 9 6 --square 0.025 --out SESSION/intrinsics.json
    python calibrate_intrinsics.py --images "calib/*.jpg" --board 9 6 --square 0.025 --out SESSION/intrinsics.json

--board: inner corners per row and per column (a 10 x 7 squares board has 9 x 6).
--square: square side [m] (only scales the board poses; K does not depend on it).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import Intrinsics, video_frames  # noqa: E402


def board_points(cols, rows, square):
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def find_corners(gray, board):
    ok, corners = cv2.findChessboardCornersSB(gray, board, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                                              | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    return ok, corners


def iter_images(args):
    if args.video:
        info_step = args.every
        last = None
        for k, t, frame in video_frames(args.video, step=info_step):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # skip blurred frames (motion blur ruins the corners)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            if last is not None and sharp < 0.3 * last:
                continue
            last = sharp if last is None else 0.9 * last + 0.1 * sharp
            yield f"{Path(args.video).name}@{t:.2f}s", gray
    else:
        for p in sorted(glob.glob(args.images)):
            img = cv2.imread(p)
            if img is not None:
                yield Path(p).name, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def select_diverse(views, max_views):
    """At most max_views views, spread over the image (greedy farthest point on the board centre / size)."""
    if len(views) <= max_views:
        return views
    feats = np.array([[*c.reshape(-1, 2).mean(0), np.ptp(c.reshape(-1, 2), axis=0).mean()] for _, c in views])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    chosen = [0]
    d = np.linalg.norm(feats - feats[0], axis=1)
    while len(chosen) < max_views:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(feats - feats[k], axis=1))
    return [views[k] for k in sorted(chosen)]


def calibrate(views, size, board, square, model="standard"):
    obj = board_points(*board, square)
    flags = 0
    if model == "rational":
        flags |= cv2.CALIB_RATIONAL_MODEL
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(views), [c for _, c in views], size, None, None,
                                                     flags=flags)
    per_view = []
    for (name, c), rv, tv in zip(views, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        per_view.append((name, float(np.sqrt(np.mean(np.sum((proj - c) ** 2, axis=2))))))
    return rms, K, dist, per_view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images", help="glob of images")
    ap.add_argument("--board", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"), help="inner corners")
    ap.add_argument("--square", type=float, required=True, help="square side [m]")
    ap.add_argument("--every", type=int, default=10, help="video: use every N-th frame")
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--model", choices=["standard", "rational"], default="standard",
                    help="standard: k1 k2 p1 p2 k3; rational: 8 coefficients (wide-angle / GoPro linear off)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", help="write an undistorted preview image here")
    args = ap.parse_args()

    board = tuple(args.board)
    views, size, last_gray = [], None, None
    n_seen = 0
    for name, gray in iter_images(args):
        n_seen += 1
        size = gray.shape[::-1]
        ok, corners = find_corners(gray, board)
        if ok:
            views.append((name, corners.astype(np.float32)))
            last_gray = gray
    print(f"Checkerboard found in {len(views)} / {n_seen} images")
    if len(views) < 8:
        sys.exit("Need at least 8 views of the checkerboard")
    views = select_diverse(views, args.max_views)

    rms, K, dist, per_view = calibrate(views, size, board, args.square, args.model)
    # drop outlier views (> 3x median error) and recalibrate once
    errs = np.array([e for _, e in per_view])
    keep = errs < 3 * np.median(errs)
    if not np.all(keep) and keep.sum() >= 8:
        print(f"Dropping {np.sum(~keep)} outlier views")
        views = [v for v, k in zip(views, keep) if k]
        rms, K, dist, per_view = calibrate(views, size, board, args.square, args.model)

    intr = Intrinsics(K, dist.ravel(), size[0], size[1], float(rms))
    intr.save(args.out, n_views=len(views), board=list(board), square_m=args.square, model=args.model)
    fov = np.degrees(2 * np.arctan(np.array(size) / (2 * np.diag(K)[:2])))
    print(f"RMS reprojection error {rms:.3f} px over {len(views)} views (good: < 0.5 px)")
    print(f"fx fy = {K[0, 0]:.1f} {K[1, 1]:.1f}, cx cy = {K[0, 2]:.1f} {K[1, 2]:.1f}, "
          f"FOV {fov[0]:.1f} x {fov[1]:.1f} deg, dist = {np.round(dist.ravel(), 4).tolist()}")
    print(f"Saved to {args.out}")
    if args.preview and last_gray is not None:
        und = cv2.undistort(last_gray, K, dist)
        cv2.imwrite(args.preview, np.hstack([last_gray, und]))


if __name__ == "__main__":
    main()
