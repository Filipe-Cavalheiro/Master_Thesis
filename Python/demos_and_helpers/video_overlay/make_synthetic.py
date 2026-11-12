"""
Synthetic filming session(s) with known ground truth, to test the whole
pipeline without real footage.

A real flight log (default: swarm_MPC_voronoi_sim_target, 26 Oct, 3 drones,
with its simulation <log>_sim.csv from Crazy_Flies/flight_sim.py) is the
ground truth motion. A ray-cast arena (6 x 6 x 3 m room, textured floor with
tape grid, netted walls) is filmed by a virtual camera with lens distortion,
sensor noise and finite exposure, plus:
  * the drones (body, arms, rotors, LED) at their logged positions,
  * 8 bright calibration markers at surveyed positions (points.csv),
  * screen flashes (a white screen held in the lower-right of the view) at
    the mocap times logged in mocap_events.csv,
  * a checkerboard calibration image set (calib/).
The recorder files are written as record_mocap.py would: the drones and the
camera RIGID BODY (camera pose composed with the inverse of a known hand-eye
offset, with mocap noise, 120 Hz) in the recorder time base, which differs
from the flight log time base (log offset) and from the video time
(offset + clock-rate scale).

Two sessions: `moving` (handheld camera walking along the side of the arena)
and `static` (tripod: stroboscopic composite).

    python make_synthetic.py --out test_output/synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import (CAMERA_COLUMNS, EVENT_COLUMNS, REPO, TRAJ_COLUMNS, Tracks, VideoWriter, inv_T,  # noqa: E402
                         look_at, make_T, read_traj)

FLIGHT_LOG = (next(p for p in HERE.parents if (p / ".git").exists()) / "Results"
              / "swarm_MPC_voronoi_sim_target_trajectory_20261026_141807" / "trajectory_20261026_141807.csv")

# ---------------- ground truth ----------------
W, H, FPS = 960, 540, 30.0
K_TRUE = np.array([[650.0, 0.0, 483.0], [0.0, 652.0, 268.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.11, 0.06, 0.0008, -0.0006, 0.0])
# rigid body <- camera: the marker constellation's frame is rotated and offset from the optical centre
R_BC_TRUE = Rotation.from_euler("xyz", [-87.0, 3.0, -92.0], degrees=True).as_matrix()
T_BC_TRUE = make_T(R_BC_TRUE, [0.032, -0.048, 0.021])
LOG_OFFSET = 7.3            # recorder time = flight log time + LOG_OFFSET
VIDEO_OFFSET = 1.0137       # t_mocap = VIDEO_SCALE * t_video + VIDEO_OFFSET
VIDEO_SCALE = 1.00003       # phone clock 30 ppm fast
FLASH_TIMES = [1.8, 3.9, 5.25, 50.6]   # mocap time of the screen flashes
FLASH_LEN = 0.12
MOCAP_RATE = 120.0
CAMERA_ID = 10
MARGIN = 120                # px of ideal (undistorted) canvas around the image
POINTS = [("m1", -1.5, -1.2, 0.0), ("m2", -1.2, 1.8, 0.0), ("m3", 1.4, -1.0, 0.0), ("m4", 1.5, 1.6, 0.0),
          ("m5", -0.6, -1.3, 0.75), ("m6", 0.5, 2.0, 1.1), ("m7", 0.9, -0.3, 1.6), ("m8", -1.0, 0.6, 0.45)]
ROOM = 3.0                  # half side [m]
ROOM_H = 3.0


# ============================================================
# SCENE
# ============================================================

def make_atlas(seed=0, ppm=200):
    """Textures of the 6 room faces in one atlas; returns atlas, face table (axis, value, u axis, v axis, offset)."""
    rng = np.random.default_rng(seed)
    n6, n3 = int(6 * ppm), int(3 * ppm)

    def noise(h, w, base, amp, blur):
        t = rng.normal(0, 1, (h, w)).astype(np.float32)
        t = cv2.GaussianBlur(t, (0, 0), blur)
        t = base + amp * t / (t.std() + 1e-6)
        return t

    floor = np.dstack([noise(n6, n6, b, 10, 3) for b in (118, 120, 124)])
    floor += noise(n6, n6, 0, 6, 25)[..., None]
    for k in range(0, 7):           # tape grid every 1 m, white
        p = int(k * ppm)
        cv2.line(floor, (p, 0), (p, n6 - 1), (225, 225, 225), 6)
        cv2.line(floor, (0, p), (n6 - 1, p), (225, 225, 225), 6)
    cv2.rectangle(floor, (int(1.4 * ppm), int(2.6 * ppm)), (int(2.2 * ppm), int(3.4 * ppm)), (40, 40, 60), -1)   # mat
    cv2.circle(floor, (n6 // 2, n6 // 2), int(0.15 * ppm), (30, 30, 200), 8)                                    # origin

    def wall(color, seed_shift):
        w = np.dstack([noise(n3, n6, c, 8, 2) for c in color])
        for k in range(0, n6, int(0.25 * ppm)):    # safety net
            cv2.line(w, (k, 0), (k, n3 - 1), (25, 25, 25), 2)
        for k in range(0, n3, int(0.25 * ppm)):
            cv2.line(w, (0, k), (n6 - 1, k), (25, 25, 25), 2)
        r = np.random.default_rng(seed_shift)
        for _ in range(4):                           # posters / boxes
            x, y = r.integers(0, n6 - 300), r.integers(int(0.3 * ppm), n3 - 250)
            c = tuple(int(v) for v in r.integers(40, 220, 3))
            cv2.rectangle(w, (x, y), (x + r.integers(120, 300), y + r.integers(100, 240)), c, -1)
        return w

    walls = [wall((70, 72, 78), 1), wall((78, 74, 70), 2), wall((72, 78, 72), 3), wall((74, 70, 80), 4)]
    ceil = np.dstack([noise(n6, n6, b, 6, 4) for b in (55, 55, 58)])
    atlas = np.zeros((2 * n6, 2 * n6, 3), np.float32)
    atlas[:n6, :n6] = floor
    atlas[:n6, n6:] = ceil
    atlas[n6:n6 + n3, :n6] = walls[0]
    atlas[n6:n6 + n3, n6:] = walls[1]
    atlas[n6 + n3:, :n6] = walls[2]
    atlas[n6 + n3:, n6:] = walls[3]
    atlas = np.clip(atlas, 0, 255).astype(np.uint8)
    # face: (normal axis, plane value, u axis, u origin, v axis, v origin, v sign, atlas x0, atlas y0)
    faces = [
        (2, 0.0, 0, -ROOM, 1, -ROOM, 1, 0, 0),             # floor: u = x, v = y
        (2, ROOM_H, 0, -ROOM, 1, -ROOM, 1, n6, 0),         # ceiling
        (0, -ROOM, 1, -ROOM, 2, ROOM_H, -1, 0, n6),        # wall x = -3: u = y, v = top-down z
        (0, ROOM, 1, -ROOM, 2, ROOM_H, -1, n6, n6),
        (1, -ROOM, 0, -ROOM, 2, ROOM_H, -1, 0, n6 + n3),
        (1, ROOM, 0, -ROOM, 2, ROOM_H, -1, n6, n6 + n3),
    ]
    return atlas, faces, ppm


class Renderer:
    """Ray-cast room (exit face of the convex box) + drones and markers drawn on an undistorted canvas, then distorted."""

    def __init__(self):
        self.atlas, self.faces, self.ppm = make_atlas()
        self.Wc, self.Hc = W + 2 * MARGIN, H + 2 * MARGIN
        self.Kc = K_TRUE.copy()
        self.Kc[0, 2] += MARGIN
        self.Kc[1, 2] += MARGIN
        u, v = np.meshgrid(np.arange(self.Wc, dtype=np.float32), np.arange(self.Hc, dtype=np.float32))
        pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size, np.float32)])
        self.rays_c = (np.linalg.inv(self.Kc) @ pix).astype(np.float32)      # (3, N) camera-frame rays
        # distortion map: distorted output pixel -> undistorted canvas pixel
        ud, vd = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        pts = np.stack([ud.ravel(), vd.ravel()], 1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6)
        und = cv2.undistortPoints(pts, K_TRUE, DIST_TRUE, None, None, K_TRUE, crit).reshape(H, W, 2)
        self.map_x = (und[..., 0] + MARGIN).astype(np.float32)
        self.map_y = (und[..., 1] + MARGIN).astype(np.float32)
        self.rng = np.random.default_rng(7)

    def background(self, T_wc):
        R, c = T_wc[:3, :3].astype(np.float32), T_wc[:3, 3].astype(np.float32)
        d = R @ self.rays_c                              # world rays (3, N)
        best = np.full(d.shape[1], np.inf, np.float32)
        mx = np.zeros(d.shape[1], np.float32)
        my = np.zeros(d.shape[1], np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            for ax, val, ua, uo, va, vo, vs, x0, y0 in self.faces:
                s = (val - c[ax]) / d[ax]
                s[~(s > 0)] = np.inf
                closer = s < best
                if not closer.any():
                    continue
                best[closer] = s[closer]
                hu = c[ua] + s[closer] * d[ua, closer]
                hv = c[va] + s[closer] * d[va, closer]
                mx[closer] = x0 + (hu - uo) * self.ppm
                my[closer] = y0 + (vs * (hv - vo)) * self.ppm
        return cv2.remap(self.atlas, mx.reshape(self.Hc, self.Wc), my.reshape(self.Hc, self.Wc), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def proj(self, P, T_wc):
        T_cw = inv_T(T_wc)
        pc = np.atleast_2d(P) @ T_cw[:3, :3].T + T_cw[:3, 3]
        uv = pc[:, :2] / pc[:, 2:3] * np.diag(self.Kc)[:2] + self.Kc[:2, 2]
        return uv, pc[:, 2]

    def objects(self, canvas, T_wc, drones, show_markers=True):
        """Drones (dict id -> position) and markers, far to near, with 4x supersampled drawing."""
        f = self.Kc[0, 0]
        items = []
        for p in drones.values():
            if np.all(np.isfinite(p)):
                items.append(("drone", np.asarray(p, float)))
        if show_markers:
            items += [("marker", np.array(q[1:], float)) for q in POINTS]
        depth = [self.proj(p, T_wc)[1][0] for _, p in items]
        S = 4   # cv2 shift bits: 1/16 px precision
        for (kind, p), z in sorted(zip(items, depth), key=lambda it: -it[1]):
            if z < 0.2:
                continue
            if kind == "marker":
                uv, _ = self.proj(p, T_wc)
                r = max(2.5, f * 0.014 / z)
                cv2.circle(canvas, tuple((uv[0] * 16).astype(int)), int(r * 16), (250, 250, 250), -1, cv2.LINE_AA, S)
                continue
            arms = p + np.array([[0.033, 0.033, 0], [-0.033, 0.033, 0], [-0.033, -0.033, 0], [0.033, -0.033, 0]])
            uv_c, _ = self.proj(p, T_wc)
            uv_r, zr = self.proj(arms, T_wc)
            c = tuple((uv_c[0] * 16).astype(int))
            for q in uv_r:
                cv2.line(canvas, c, tuple((q * 16).astype(int)), (35, 35, 35), max(1, int(f * 0.006 / z)), cv2.LINE_AA, S)
            for q, zz in zip(uv_r, zr):
                rr = max(2.0, f * 0.024 / zz)
                cv2.circle(canvas, tuple((q * 16).astype(int)), int(rr * 16), (175, 175, 180), -1, cv2.LINE_AA, S)
                cv2.circle(canvas, tuple((q * 16).astype(int)), int(rr * 16), (40, 40, 40), 1, cv2.LINE_AA, S)
            cv2.circle(canvas, c, int(max(2.0, f * 0.016 / z) * 16), (30, 30, 30), -1, cv2.LINE_AA, S)
            cv2.circle(canvas, c, int(max(1.0, f * 0.005 / z) * 16), (255, 160, 40), -1, cv2.LINE_AA, S)

    def finish(self, canvas, flash=0.0):
        img = cv2.remap(canvas, self.map_x, self.map_y, cv2.INTER_LINEAR)
        if flash > 0:   # a laptop screen held in the lower right, white with this exposure fraction
            x0, y0 = int(0.68 * W), int(0.62 * H)
            roi = img[y0:H - 15, x0:W - 15].astype(np.float32)
            img[y0:H - 15, x0:W - 15] = (roi + flash * (250 - roi)).astype(np.uint8)
        noise = self.rng.normal(0, 2.0, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ============================================================
# CAMERA PATHS
# ============================================================

def camera_pose(kind, t_m):
    """True camera optical pose T_wc at mocap time t_m."""
    if kind == "static":
        eye, tgt = np.array([-2.85, 0.25, 1.55]), np.array([0.0, 0.3, 0.95])
        return make_T(look_at(eye, tgt), eye)
    # handheld: slow walk along the side wall + small shake
    eye = np.array([-2.55 + 0.15 * np.sin(0.31 * t_m),
                    0.3 + 0.75 * np.sin(0.19 * t_m),
                    1.45 + 0.08 * np.sin(0.53 * t_m)])
    tgt = np.array([0.0, 0.3 + 0.35 * np.sin(0.15 * t_m + 1.0), 0.95 + 0.1 * np.sin(0.4 * t_m)])
    R = look_at(eye, tgt)
    shake = Rotation.from_rotvec(np.radians(0.4) * np.array([np.sin(2.1 * t_m), np.sin(2.7 * t_m + 1), np.sin(1.7 * t_m + 2)]))
    return make_T(R @ shake.as_matrix(), eye)


# ============================================================
# FILES
# ============================================================

def clamped(tracks: Tracks, i, times):
    """Positions of drone i at `times`, held at the first/last logged sample outside the log."""
    tf, pf = tracks.data[i]
    return np.column_stack([np.interp(times, tf, pf[:, a]) for a in range(3)])


def drones_at(tracks: Tracks, t):
    return {i: clamped(tracks, i, np.array([t]))[0] for i in tracks.ids}


def write_recorder_logs(out: Path, kind, flight: Tracks, t_end):
    rng = np.random.default_rng(3 if kind == "static" else 4)
    base = np.arange(0.0, t_end, 1.0 / MOCAP_RATE)
    ts = np.sort(base + rng.normal(0, 2e-4, len(base)))
    # drones: flight log shifted to recorder time, held still before/after the log (sitting on the floor)
    parts = []
    for i in flight.ids:
        P = clamped(flight, i, ts - LOG_OFFSET) + rng.normal(0, 3e-4, (len(ts), 3))
        parts.append(pd.DataFrame({"time_s": ts, "id": i, "opti_x": P[:, 0], "opti_y": P[:, 1], "opti_z": P[:, 2]}))
    df = pd.concat(parts).sort_values(["time_s", "id"], kind="stable")
    full = pd.DataFrame(columns=TRAJ_COLUMNS)
    for c in ["time_s", "id", "opti_x", "opti_y", "opti_z"]:
        full[c] = df[c].to_numpy()
    full["flight_state"] = ""
    full.to_csv(out / "mocap_trajectory.csv", index=False, float_format="%.6f")
    # camera rigid body: T_wb = T_wc T_bc^-1, with mocap noise
    cam = []
    T_cb = inv_T(T_BC_TRUE)
    for n, t in enumerate(ts):
        T_wb = camera_pose(kind, t) @ T_cb
        R = Rotation.from_matrix(T_wb[:3, :3]) * Rotation.from_rotvec(np.radians(0.04) * rng.normal(0, 1, 3))
        p = T_wb[:3, 3] + rng.normal(0, 3e-4, 3)
        cam.append([t, n, t + 1000.0, CAMERA_ID, 1, *p, *R.as_quat(), 0.0004])
    pd.DataFrame(cam, columns=CAMERA_COLUMNS).to_csv(out / "mocap_camera.csv", index=False, float_format="%.7f")
    pd.DataFrame([[t, "flash", str(k + 1)] for k, t in enumerate(FLASH_TIMES)], columns=EVENT_COLUMNS).to_csv(
        out / "mocap_events.csv", index=False)


def render_video(out: Path, kind, flight: Tracks, rend: Renderer, t_v_end, every=1):
    writer = VideoWriter(out / "video.mp4", FPS, (W, H))
    dt = 1.0 / FPS
    n_frames = int(t_v_end * FPS)
    static_bg = rend.background(camera_pose("static", 0.0)) if kind == "static" else None
    for k in range(n_frames):
        tv = k * dt
        t_m = VIDEO_SCALE * tv + VIDEO_OFFSET
        T_wc = camera_pose(kind, t_m)
        canvas = static_bg.copy() if static_bg is not None else rend.background(T_wc)
        rend.objects(canvas, T_wc, drones_at(flight, t_m - LOG_OFFSET))
        # exposure = frame period: lit fraction of the frame
        flash = 0.0
        for tf in FLASH_TIMES:
            a, b = (tf - VIDEO_OFFSET) / VIDEO_SCALE, (tf + FLASH_LEN - VIDEO_OFFSET) / VIDEO_SCALE
            flash = max(flash, max(0.0, min(b, tv + dt) - max(a, tv)) / dt)
        writer.write(rend.finish(canvas, flash))
        if k % 300 == 0:
            print(f"  {kind}: frame {k}/{n_frames}")
    writer.close()


def render_checkerboards(out: Path, rend: Renderer, n=30, board=(9, 6), square=0.04, seed=11):
    """Checkerboard photos with the true intrinsics / distortion (board on a grey background)."""
    rng = np.random.default_rng(seed)
    cols, rows = board
    sq_px = 40
    tex = np.full(((rows + 3) * sq_px, (cols + 3) * sq_px), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = (r + 1) * sq_px, (c + 1) * sq_px
                tex[y:y + sq_px, x:x + sq_px] = 0
    # board coordinates of the texture corners [m]: inner corner (0, 0) at texture pixel (2 sq, 2 sq)
    tw, th = tex.shape[1], tex.shape[0]
    corners_tex = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    corners_m = (corners_tex / sq_px - 2.0) * square
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    while k < n:
        z = rng.uniform(0.55, 1.3)
        ang = np.radians(rng.uniform(-40, 40, 3)) * np.array([1, 1, 0.5])
        R = Rotation.from_rotvec(ang).as_matrix()
        # board centre somewhere in the field of view (corners included)
        u = rng.uniform(0.15, 0.85) * W
        v = rng.uniform(0.15, 0.85) * H
        centre = z * np.linalg.inv(K_TRUE) @ np.array([u, v, 1.0])
        bc = np.array([cols - 1, rows - 1, 0]) * square / 2
        Pw = np.c_[corners_m, np.zeros(4)] - bc
        pc = Pw @ R.T + centre
        if np.any(pc[:, 2] < 0.2):
            continue
        uv = pc[:, :2] / pc[:, 2:3] * np.diag(rend.Kc)[:2] + rend.Kc[:2, 2]
        Hm = cv2.getPerspectiveTransform(corners_tex, uv.astype(np.float32))
        canvas = np.full((rend.Hc, rend.Wc), 140, np.uint8)
        warped = cv2.warpPerspective(tex, Hm, (rend.Wc, rend.Hc), flags=cv2.INTER_AREA, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(tex), Hm, (rend.Wc, rend.Hc), borderValue=0)
        canvas[mask > 0] = warped[mask > 0]
        img = rend.finish(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(out / f"calib_{k:02d}.png"), img)
        k += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "test_output/synthetic"))
    ap.add_argument("--flight-log", default=str(FLIGHT_LOG))
    ap.add_argument("--sessions", nargs="*", default=["moving", "static"])
    args = ap.parse_args()

    out = Path(args.out)
    flight_log = Path(args.flight_log)
    sim_log = flight_log.with_name(flight_log.stem + "_sim.csv")
    flight = Tracks.from_traj(read_traj(flight_log))
    t_mocap_end = FLASH_TIMES[-1] + 1.0
    t_video_end = (t_mocap_end - VIDEO_OFFSET) / VIDEO_SCALE
    rend = Renderer()

    print("Rendering checkerboard images...")
    render_checkerboards(out / "calib", rend)
    pd.DataFrame([(n, x, y, z) for n, x, y, z in POINTS], columns=["name", "x", "y", "z"]).to_csv(
        out / "points.csv", index=False)
    truth = {"K": K_TRUE.tolist(), "dist": DIST_TRUE.tolist(), "T_bc": T_BC_TRUE.tolist(),
             "video_offset": VIDEO_OFFSET, "video_scale": VIDEO_SCALE, "log_offset": LOG_OFFSET,
             "flash_times_mocap": FLASH_TIMES, "flight_log": str(flight_log), "sim": str(sim_log)}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))

    for kind in args.sessions:
        d = out / kind
        d.mkdir(parents=True, exist_ok=True)
        print(f"Session '{kind}': recorder logs...")
        write_recorder_logs(d, kind, flight, t_mocap_end + 1.0)
        (d / "session.json").write_text(json.dumps({
            "video": "video.mp4", "drones": "mocap_trajectory.csv", "camera": "mocap_camera.csv",
            "events": "mocap_events.csv", "camera_id": CAMERA_ID,
            "flight_log": str(flight_log), "sim": str(sim_log)}, indent=2))
        print(f"Session '{kind}': video ({t_video_end:.1f} s at {FPS:.0f} fps)...")
        render_video(d, kind, flight, rend, t_video_end)
    print(f"Synthetic sessions in {out}")


if __name__ == "__main__":
    main()
