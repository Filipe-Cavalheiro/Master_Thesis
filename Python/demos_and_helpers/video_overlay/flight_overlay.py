"""
Hand-held video of a flight vs its simulation, seen from the same camera.

Inputs (from a flight whose code logs the phone rigid body, CAMERA_RIGID_BODY_ID:
swarm_take_off_hover_land, Crazy_Flie/Optitrack/8traj.py):
    LOG.csv            flight log (OptiTrack positions of the real drones, flight_state phases)
    LOG_camera.csv     phone rigid-body poses (same NatNet frames, camera_logger.py)
    LOG_camera.json    time origin of the flight log on the same clock
    LOG_sim.csv        the simulation of the flight (main.py --IRL_log of Python/flight_sims/swarm_take_off_hover_land/
                       and Python/flight_sims/8traj/)
    the phone video, and the phone calibration (phone_align.py: phone_intrinsics.json
    + extrinsics.json, default Results/phone_calib)
All in the run folder of the flight, Results/<flight code>_trajectory_<date>/
(the video: video.mp4 there by default). The phases are those of the log.

Time sync video <-> flight log: automatic, from the motion of the hand-held
camera: the global image shift of the video (phase correlation, signed x/y)
against the shift predicted by the measured rotation of the phone rigid body;
or --offset (video t = flight t + offset). Pan / tilt the phone briskly a few
times while filming (before take off is fine): that makes the sync unambiguous.

The virtual camera of every video frame is T_wc = T_wb(t) T_bc (phone pose at
the frame time, hand-eye correction T_bc), with the phone intrinsics. Outputs
(in --out, default <run folder>/overlay/):
    sim_camera.mp4     the simulated camera video: the simulated drones (Crazyflie
                       wireframes with their simulated attitude) over a rendered
                       floor, as the phone would have seen them
    overlay.mp4        the real video with the simulated drones drawn over it
                       (orange) and the real OptiTrack positions (green rings)
    side_by_side.mp4   real | simulated camera
    dashboard.mp4      the overlay next to plots in the style of the V17
                       animation_display.py animations (flight_dashboard.py): 3D
                       scene with the camera, top view, z(t) and real - sim error
    stills/            real / sim / overlay frames at take off, hover, landing (PNG)
    overlay_metrics.csv  per frame: pixel and 3D distance real vs sim per drone
    sync.json          the time offset and its correlation

    python flight_overlay.py <repo>/Results/swarm_take_off_hover_land_trajectory_X/trajectory_X.csv
    python flight_overlay.py <repo>/Results/8traj_trajectory_X/trajectory_X.csv
    python flight_overlay.py LOG.csv --video flight.mp4 --offset 12.34 --grid
    python flight_overlay.py LOG.csv --video flight.mp4 --corrected   # Motive rigid body already corrected

Phone orientation: if the phone is held upside down compared with the
calibration session (landscape the other way round), the image is rotated by
180 deg about the optical axis; --flip auto (default) picks the orientation
whose predicted image motion matches the video (signed correlation).
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
# repository Results folder (one subfolder per run), wherever this folder is in the repository
LOGS_DIR = next((p / "Results" for p in HERE.parents if (p / ".git").exists()), HERE / "Results")
sys.path.insert(0, str(HERE))

from overlay_lib import (CameraTrack, Intrinsics, VideoWriter, load_extrinsics, put_label,  # noqa: E402
                         video_frames, video_info)
from flight_dashboard import Dashboard  # noqa: E402
import camera_recovery  # noqa: E402
from phone_align import (T_BC_NOMINAL, crazyflie_segments, draw_segs, floor_segments,  # noqa: E402
                         proj_w)

COL_SIM = (30, 140, 255)       # orange (BGR)
COL_REAL = (80, 210, 60)       # green
R_FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])   # 180 deg about the optical axis (phone upside down)


# ============================================================
# DATA
# ============================================================

def read_flight(path):
    df = pd.read_csv(path, dtype={"id": str, "flight_state": str})
    df["flight_state"] = df["flight_state"].fillna("").str.strip()
    if "id" not in df.columns:                  # single-drone logs before the id column (8traj)
        df["id"] = "1"
    df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def tracks(df, with_att=False):
    out = {}
    for i, g in df.groupby("id"):
        g = g.sort_values("time_s")
        P = g[["opti_x", "opti_y", "opti_z"]].to_numpy(float)
        ok = np.all(np.isfinite(P), 1)
        d = {"t": g["time_s"].to_numpy(float)[ok], "p": P[ok]}
        if with_att and {"roll_deg", "pitch_deg", "yaw_deg"} <= set(g.columns):
            d["rpy"] = g[["roll_deg", "pitch_deg", "yaw_deg"]].to_numpy(float)[ok]
        out[i] = d
    return dict(sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0])))


def at(track, t):
    ts = track["t"]
    if len(ts) == 0 or t < ts[0] or t > ts[-1]:
        return None
    return np.array([np.interp(t, ts, track["p"][:, a]) for a in range(3)])


def attitude_at(track, t):
    if "rpy" not in track:
        return np.eye(3)
    k = int(np.clip(np.searchsorted(track["t"], t), 0, len(track["t"]) - 1))
    return Rotation.from_euler("xyz", track["rpy"][k], degrees=True).as_matrix()


def phases_of(df):
    """Start time of every flight_state of the log, in time order, and EOF (end of the log)."""
    first = df[df["flight_state"] != ""].groupby("flight_state")["time_s"].min().sort_values()
    ph = {name: float(t) for name, t in first.items()}
    ph.setdefault("EOF", float(df["time_s"].max()))
    return ph


def flight_phases(ph):
    return [n for n in ph if n != "EOF"]


def phase_at(ph, t):
    cur = ""
    for name, t0 in ph.items():
        if t >= t0:
            cur = name
    return cur


def load_camera(log_path):
    """Phone rigid-body track in flight-log time (None if the rigid body was never tracked)."""
    p = Path(log_path)
    csv = p.with_name(p.stem + "_camera.csv")
    meta = json.loads(p.with_name(p.stem + "_camera.json").read_text())
    t0 = meta.get("flight_log_t0_monotonic")
    if t0 is None:
        raise SystemExit(f"{p.stem}_camera.json has no flight_log_t0_monotonic (the flight never started logging)")
    df = pd.read_csv(csv)
    df = df[df["valid"] == 1]
    if df.empty:
        return None                                   # never tracked: camera_recovery.recover_static
    t = df["mono_s"].to_numpy(float) - t0
    return CameraTrack(t, df[["x", "y", "z"]].to_numpy(float), df[["qx", "qy", "qz", "qw"]].to_numpy(float))


# ============================================================
# TIME SYNC FROM THE CAMERA MOTION
# ============================================================

def video_flow(video, width=320):
    """
    Global image shift between consecutive frames (phase correlation on
    downscaled, Hanning-windowed frames), in full-resolution px/s: (t, dx, dy).
    """
    ts, dx, dy, prev, win, scale, tp = [], [], [], None, None, 1.0, None
    for _, t, fr in video_frames(video):
        h = int(round(fr.shape[0] * width / fr.shape[1]))
        g = cv2.cvtColor(cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if win is None:
            win = cv2.createHanningWindow((width, h), cv2.CV_32F)
            scale = fr.shape[1] / width
        if prev is not None and t > tp:
            (sx, sy), _ = cv2.phaseCorrelate(prev, g, win)
            ts.append(0.5 * (t + tp))
            dx.append(sx * scale / (t - tp))
            dy.append(sy * scale / (t - tp))
        prev, tp = g, t
    return np.array(ts), np.array(dx), np.array(dy)


def camera_flow(cam: CameraTrack, T_bc, f_px):
    """
    Image shift predicted by the rotation of the camera [px/s]: angular
    velocity in the OpenCV camera frame, dx = -f w_y, dy = f w_x (a point on
    the optical axis moves by -w x p), on a 100 Hz grid.
    """
    dt = 0.01
    g = np.arange(cam.t[0], cam.t[-1] - dt, dt)
    R_bc = Rotation.from_matrix(T_bc[:3, :3])
    Rc = cam.slerp(g) * R_bc
    Rc2 = cam.slerp(g + dt) * R_bc
    w = (Rc.inv() * Rc2).as_rotvec() / dt
    dx, dy = -f_px * w[:, 1], f_px * w[:, 0]
    k = 20                                     # 0.2 s: the 360 Hz rigid-body rotation is noisy
    box = np.ones(k) / k
    return g + 0.5 * dt, np.convolve(dx, box, "same"), np.convolve(dy, box, "same")


def estimate_offset(vt, vdx, vdy, ct, cdx, cdy):
    """
    video_t = flight_t + offset maximizing the correlation of the measured
    and predicted image shifts (both axes, signed); coarse 0.05 s, fine 0.002 s.
    Returns (offset, correlation, best correlation more than 1 s away).
    """
    corr = flow_corr(vt, vdx, vdy, ct, cdx, cdy)
    coarse = np.arange(vt[0] - ct[-1], vt[-1] - ct[0], 0.05)
    c = np.array([corr(o) for o in coarse])
    o0 = coarse[int(np.argmax(c))]
    fine = np.arange(o0 - 0.1, o0 + 0.1, 0.002)
    cf = np.array([corr(o) for o in fine])
    far = np.abs(coarse - o0) > 1.0
    return float(fine[int(np.argmax(cf))]), float(cf.max()), float(c[far].max()) if far.any() else float("-inf")


def flow_corr(vt, vdx, vdy, ct, cdx, cdy):
    """corr(offset): signed correlation of the video and predicted image shifts (video smoothed over 0.2 s)."""
    n = max(1, int(round(0.2 / np.median(np.diff(vt)))))
    box = np.ones(n) / n
    clip = lambda x: np.clip(x, -np.percentile(np.abs(x), 99), np.percentile(np.abs(x), 99))  # noqa: E731
    vdx, vdy = np.convolve(clip(vdx), box, "same"), np.convolve(clip(vdy), box, "same")

    def corr(off):
        x = np.interp(vt - off, ct, cdx, left=np.nan, right=np.nan)
        y = np.interp(vt - off, ct, cdy, left=np.nan, right=np.nan)
        m = np.isfinite(x)
        if m.sum() < 0.5 * min(len(vt), (ct[-1] - ct[0]) * len(vt) / (vt[-1] - vt[0] + 1e-9)):
            return -np.inf
        a = np.r_[vdx[m], vdy[m]]
        b = np.r_[x[m], y[m]]
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    return corr


# ============================================================
# RENDERING
# ============================================================

def floor_polygon(img, T_wc, K, dist, half=3.0, color=(70, 72, 76)):
    """Filled floor (z = 0) of the simulated camera, clipped to the visible part."""
    n = 60
    edge = np.r_[np.linspace([-half, -half], [half, -half], n), np.linspace([half, -half], [half, half], n),
                 np.linspace([half, half], [-half, half], n), np.linspace([-half, half], [-half, -half], n)]
    P = np.column_stack([edge, np.zeros(len(edge))])
    uv, z = proj_w(P, T_wc, K, dist)
    ok = np.all(np.isfinite(uv), 1)
    if ok.sum() >= 3:
        hull = cv2.convexHull(np.round(uv[ok]).astype(np.int32))
        cv2.fillConvexPoly(img, hull, color, cv2.LINE_AA)


def draw_drone(img, p, R, T_wc, K, dist, color, thick, label=None):
    T_wo = np.eye(4)
    T_wo[:3, :3], T_wo[:3, 3] = R, p
    segs = crazyflie_segments()
    # thicker silhouette: draw the wireframe twice (dark outline, then colour)
    draw_segs(img, segs, T_wc, K, dist, (20, 20, 20), thick + 2, T_wo=T_wo, n=4)
    draw_segs(img, segs, T_wc, K, dist, color, thick, T_wo=T_wo, n=4)
    if label:
        uv, _ = proj_w(p + np.array([0, 0, 0.06]), T_wc, K, dist)
        if np.all(np.isfinite(uv[0])):
            put_label(img, label, tuple(np.round(uv[0] + (8, -4)).astype(int)), color, 0.45)


def draw_trail(img, track, t, T_wc, K, dist, color, length=2.0):
    m = (track["t"] >= t - length) & (track["t"] <= t)
    if m.sum() < 2:
        return
    uv, _ = proj_w(track["p"][m], T_wc, K, dist)
    n = len(uv)
    for k in range(1, n):
        a, b = uv[k - 1], uv[k]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            c = tuple(int((0.3 + 0.7 * k / n) * ch) for ch in color)
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), c, 2, cv2.LINE_AA)


def ring(img, p, T_wc, K, dist, color, thick):
    uv, z = proj_w(p, T_wc, K, dist)
    if np.all(np.isfinite(uv[0])) and z[0] > 0:
        r = int(np.clip(K[0, 0] * 0.07 / z[0], 5, 80))
        c = tuple(np.round(uv[0]).astype(int))
        cv2.circle(img, c, r, (20, 20, 20), thick + 2, cv2.LINE_AA)
        cv2.circle(img, c, r, color, thick, cv2.LINE_AA)
        return uv[0]
    return None


class Renderer:
    def __init__(self, intr: Intrinsics, real, sim, ph, grid=False, trail=2.0):
        self.intr, self.real, self.sim, self.ph, self.grid, self.trail = intr, real, sim, ph, grid, trail
        self.K, self.dist = intr.K, intr.dist
        self.th = max(1, int(round(2 * intr.width / 1280)))

    def sim_view(self, T_wc, t, shape):
        """The simulated camera image."""
        h, w = shape[:2]
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (46, 40, 36)                                     # background (walls / room)
        floor_polygon(img, T_wc, self.K, self.dist)
        draw_segs(img, floor_segments(3.0, 0.5), T_wc, self.K, self.dist, (110, 110, 110), 1)
        self._arena(img, T_wc)
        for i, tr in self.sim.items():
            p = at(tr, t)
            if p is None:
                continue
            draw_trail(img, tr, t, T_wc, self.K, self.dist, COL_SIM, self.trail)
            draw_drone(img, p, attitude_at(tr, t), T_wc, self.K, self.dist, COL_SIM, self.th + 1, f"CF{int(i):02d} sim")
        return img

    def _arena(self, img, T_wc, half=2.5, top=2.5):
        c = [(-half, -half), (half, -half), (half, half), (-half, half)]
        segs = [((*c[k], 0.0), (*c[(k + 1) % 4], 0.0)) for k in range(4)]
        segs += [((*c[k], top), (*c[(k + 1) % 4], top)) for k in range(4)]
        segs += [((*c[k], 0.0), (*c[k], top)) for k in range(4)]
        draw_segs(img, [(np.array(a), np.array(b)) for a, b in segs], T_wc, self.K, self.dist, (140, 140, 160), 1, n=30)

    def overlay(self, frame, T_wc, t, with_real=True):
        img = frame.copy()
        if self.grid:
            draw_segs(img, floor_segments(3.0, 0.5), T_wc, self.K, self.dist, (200, 200, 200), 1)
        rows = {}
        for i, tr in self.sim.items():
            p = at(tr, t)
            if p is None:
                continue
            draw_trail(img, tr, t, T_wc, self.K, self.dist, COL_SIM, self.trail)
            draw_drone(img, p, attitude_at(tr, t), T_wc, self.K, self.dist, COL_SIM, self.th + 1, f"CF{int(i):02d} sim")
            uv_s, _ = proj_w(p, T_wc, self.K, self.dist)
            rows[i] = {"sim": p, "uv_sim": uv_s[0]}
        if with_real:
            for i, tr in self.real.items():
                p = at(tr, t)
                if p is None:
                    continue
                uv = ring(img, p, T_wc, self.K, self.dist, COL_REAL, self.th)
                rows.setdefault(i, {}).update({"real": p, "uv_real": uv})
        return img, rows

    def hud(self, img, t, offset):
        ph = phase_at(self.ph, t)
        t0 = self.ph[flight_phases(self.ph)[0]] if flight_phases(self.ph) else 0.0
        put_label(img, f"t = {t - t0:5.2f} s   {ph if ph != 'EOF' else 'landed'}", (12, 26), (255, 255, 255), 0.6)
        put_label(img, "simulation", (12, img.shape[0] - 34), COL_SIM, 0.55)
        put_label(img, "real (OptiTrack)", (12, img.shape[0] - 12), COL_REAL, 0.55)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="flight log (Results/<flight code>_trajectory_*/trajectory_*.csv)")
    ap.add_argument("--video", help="phone video (default video.mp4 in the run folder of the log)")
    ap.add_argument("--sim", help="simulation CSV (default <log>_sim.csv)")
    ap.add_argument("--calib", default=str(LOGS_DIR / "phone_calib"),
                    help="directory with phone_intrinsics.json and extrinsics.json (phone_align.py)")
    ap.add_argument("--corrected", action="store_true",
                    help="the phone rigid body is already corrected in Motive (T_bc = nominal camera)")
    ap.add_argument("--flip", choices=("auto", "on", "off"), default="auto",
                    help="phone upside down vs the calibration (180 deg about the optical axis); auto: from the sync")
    ap.add_argument("--offset", type=float, help="video t = flight t + offset [s] (default: from the camera motion)")
    ap.add_argument("--out", help="output directory (default <run folder>/overlay)")
    ap.add_argument("--margin", type=float, default=2.0, help="seconds rendered before take off / after EOF")
    ap.add_argument("--grid", action="store_true", help="also draw the floor grid on the overlay (calibration check)")
    ap.add_argument("--no-real", action="store_true", help="overlay: only the simulation")
    ap.add_argument("--trail", type=float, default=2.0, help="trail length [s]")
    ap.add_argument("--no-video", action="store_true", help="only sync, stills and metrics")
    ap.add_argument("--no-dashboard", action="store_true", help="do not render dashboard.mp4 (the slow part)")
    ap.add_argument("--dash-size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    ap.add_argument("--dash-trail", type=float, help="trail length in the dashboard 3D / top views [s] (default: all)")
    ap.add_argument("--recover", choices=("auto", "off"), default="auto",
                    help="camera pose where the rigid body was lost: rotation from the video, anchored on the "
                         "tracked poses around the gap, refined on the drones (camera_recovery.py)")
    ap.add_argument("--no-drone-refine", action="store_true", help="gaps: video + anchors only, no drone refinement")
    ap.add_argument("--static", action="store_true",
                    help="static camera (tripod) solved from the drones: pose, focal length, k1 and offset "
                         "(automatic when the rigid body was never tracked; needs a rough --offset, +- 1 s)")
    ap.add_argument("--keep-intrinsics", action="store_true",
                    help="--static: keep the intrinsics of --calib instead of solving f and k1 (same lens and zoom)")
    args = ap.parse_args()

    log = Path(args.log).resolve()
    sim_p = Path(args.sim) if args.sim else log.with_name(log.stem + "_sim.csv")
    out = Path(args.out) if args.out else log.parent / "overlay"
    if args.video is None:
        args.video = str(log.parent / "video.mp4")
        if not Path(args.video).exists():
            raise SystemExit(f"No --video and no {args.video}")
    (out / "stills").mkdir(parents=True, exist_ok=True)

    flight = read_flight(log)
    real = tracks(flight)
    sim = tracks(read_flight(sim_p), with_att=True)
    ph = phases_of(flight)
    cam = load_camera(log)
    names = flight_phases(ph)
    t_a, t_b = (ph[names[0]] if names else 0.0) - args.margin, ph["EOF"] + args.margin

    info = video_info(args.video)
    W, H = info["width"], info["height"]
    calib = Path(args.calib)
    static = args.static or cam is None
    if static:
        # --- camera never tracked (or forced static): solve it from the drones
        if args.offset is None:
            raise SystemExit("No valid camera pose in the camera log: static camera solve, give a rough --offset "
                             "(video t = flight t + offset, +- 1 s: take off in the video - take off in the log)")
        print("Static camera: pose" + ("" if args.keep_intrinsics else ", focal length, k1") +
              " and offset from the drones ...")
        keep = Intrinsics.load(calib / "phone_intrinsics.json") if args.keep_intrinsics else None
        T_wc, intr, offset, rep = camera_recovery.recover_static(args.video, real, t_a + args.margin,
                                                                 t_b - args.margin, args.offset, keep)
        intr.save(out / "static_intrinsics.json", note="solved from the drones (camera_recovery.recover_static)")
        (out / "static_camera.json").write_text(json.dumps({**rep, "T_wc": T_wc.tolist()}, indent=2))
        cam = camera_recovery.static_track(T_wc, np.eye(4), t_a, t_b)
        T_bc0 = np.eye(4)
        args.offset, args.flip = offset, "off"
        args.recover = "off"
    else:
        intr = Intrinsics.load(calib / "phone_intrinsics.json")
    if not static and (intr.width > intr.height) != (W > H):
        raise SystemExit("The video and the phone calibration have different orientations (portrait / landscape)")
    if not static and abs(intr.width / intr.height - W / H) > 0.01 * W / H:
        print(f"WARNING: video {W}x{H} and calibration {intr.width}x{intr.height} have different aspect ratios: "
              "another lens / zoom (e.g. 0.5x ultra wide)? The intrinsics do not apply; a static camera can be "
              "solved from the drones (--static --offset)")
    elif (intr.width, intr.height) != (W, H):
        print(f"Note: video {W}x{H}, calibration {intr.width}x{intr.height}: intrinsics scaled "
              "(valid only if the phone used the same field of view)")
    intr = intr.scaled_to(W, H)
    if not static:
        T_bc0 = T_BC_NOMINAL if args.corrected else load_extrinsics(calib / "extrinsics.json")
    cands = {"off": T_bc0, "on": T_bc0 @ R_FLIP}
    if args.flip != "auto":
        cands = {args.flip: cands[args.flip]}

    # --- time sync (and phone orientation)
    res = {}
    if args.offset is not None and len(cands) == 1:
        flip = next(iter(cands))
        offset, cc, second = args.offset, float("nan"), float("nan")
    else:
        print("Sync: image flow of the video vs rotation of the phone ...")
        vt, vdx, vdy = video_flow(args.video)
        for name, T in cands.items():
            ct, cdx, cdy = camera_flow(cam, T, intr.K[0, 0])
            if args.offset is not None:
                res[name] = (args.offset, flow_corr(vt, vdx, vdy, ct, cdx, cdy)(args.offset), float("nan"))
            else:
                res[name] = estimate_offset(vt, vdx, vdy, ct, cdx, cdy)
            print(f"  flip {name:3s}: video t = flight t + {res[name][0]:.3f} s "
                  f"(correlation {res[name][1]:.2f}, next best {res[name][2]:.2f})")
        flip = max(res, key=lambda k: res[k][1])
        offset, cc, second = res[flip]
        print(f"Sync: video t = flight t + {offset:.3f} s, phone {'upside down (flip on)' if flip == 'on' else 'as calibrated'}")
        if cc < 0.5 or second > 0.8 * cc:
            print("  WARNING: weak or ambiguous sync: pan / tilt the phone briskly a few times during the "
                  "recording (a sync gesture), or give --offset")
    T_bc = cands[flip]
    (out / "sync.json").write_text(json.dumps({"video_t_equals_flight_t_plus": offset, "correlation": cc,
                                               "next_best_correlation": second, "flip": flip,
                                               "candidates": {k: list(v) for k, v in res.items()}}, indent=2))

    # --- camera pose where the rigid body was lost
    if args.recover == "auto" and camera_recovery.find_gaps(cam, t_a, t_b):
        print("Camera pose gaps: recovering from the video and the drones ...")
        cam, rec_rep = camera_recovery.recover_gaps(args.video, offset, intr, T_bc, cam, real, t_a, t_b,
                                                    refine=not args.no_drone_refine)
        camera_recovery.save_track(cam, out / "camera_recovered.csv")
        camera_recovery.save_report(rec_rep, out / "camera_recovery.json")

    rend = Renderer(intr, real, sim, ph, grid=args.grid, trail=args.trail)
    v0, v1 = max(0.0, t_a + offset), t_b + offset

    # --- stills in the middle of every phase (the dashboard one: of the longest phase)
    bounds = {n: (ph[n], ph[names[k + 1]] if k + 1 < len(names) else ph["EOF"]) for k, n in enumerate(names)}
    still_t = {n.lower().replace(" ", "_"): 0.5 * (a + b) for n, (a, b) in bounds.items()}
    longest = max(bounds, key=lambda n: bounds[n][1] - bounds[n][0]) if names else None
    still_mid = 0.5 * sum(bounds[longest]) if longest else 0.5 * (t_a + t_b)
    writers = {}
    if not args.no_video:
        fps = info["fps"] or 30.0
        writers = {"overlay": VideoWriter(out / "overlay.mp4", fps, (W, H)),
                   "sim_camera": VideoWriter(out / "sim_camera.mp4", fps, (W, H)),
                   "side_by_side": VideoWriter(out / "side_by_side.mp4", fps, (2 * W, H))}
    dash = None
    if not args.no_video and not args.no_dashboard:
        sp = None
        sim_df = read_flight(sim_p)
        if {"sp_x", "sp_y", "sp_z"} <= set(sim_df.columns):
            sp = tracks(sim_df.drop(columns=["opti_x", "opti_y", "opti_z"]).rename(
                columns={"sp_x": "opti_x", "sp_y": "opti_y", "sp_z": "opti_z"}))
        dash = Dashboard(real, sim, ph, cam, intr.K, (W, H), (t_a, t_b), size=tuple(args.dash_size),
                         trail=args.dash_trail, sim_setpoint=sp)
        writers["dashboard"] = VideoWriter(out / "dashboard.mp4", fps, tuple(args.dash_size))
    metrics, n, n_nopose = [], 0, 0
    for _, tv, fr in video_frames(args.video, start_s=v0, end_s=v1):
        t = tv - offset
        T_wb = cam.T_wb(t)
        T_wc = None
        if T_wb is None:
            n_nopose += 1
            sim_img = np.zeros_like(fr)
            put_label(sim_img, "no camera pose", (12, 60), (0, 0, 255), 0.7)
            ov = fr.copy()
            rows = {}
        else:
            T_wc = T_wb @ T_bc
            sim_img = rend.sim_view(T_wc, t, fr.shape)
            ov, rows = rend.overlay(fr, T_wc, t, with_real=not args.no_real)
        rend.hud(ov, t, offset)
        rend.hud(sim_img, t, offset)
        for i, r in rows.items():
            if "sim" in r and "real" in r:
                e_px = (np.linalg.norm(r["uv_sim"] - r["uv_real"])
                        if r.get("uv_real") is not None and np.all(np.isfinite(r["uv_sim"])) else np.nan)
                metrics.append({"video_t": tv, "flight_t": t, "phase": phase_at(ph, t), "id": i,
                                "err_3d_m": float(np.linalg.norm(r["sim"] - r["real"])), "err_px": float(e_px)})
        for name, ts in list(still_t.items()):
            if abs(t - ts) < 0.5 / (info["fps"] or 30.0):
                strip = np.hstack([fr, sim_img, ov])
                cv2.imwrite(str(out / "stills" / f"{name}_strip.png"), strip)
                for lab, im in (("real", fr), ("sim", sim_img), ("overlay", ov)):
                    cv2.imwrite(str(out / "stills" / f"{name}_{lab}.png"), im)
                del still_t[name]
        if writers:
            writers["overlay"].write(ov)
            writers["sim_camera"].write(sim_img)
            writers["side_by_side"].write(np.hstack([fr, sim_img]))
        if dash is not None:
            dimg = dash.compose(ov, t, T_wc)
            writers["dashboard"].write(dimg)
            if n == 0 or abs(t - still_mid) < 0.5 / (info["fps"] or 30.0):
                cv2.imwrite(str(out / "stills" / f"dashboard_{'start' if n == 0 else 'mid'}.png"), dimg)
        if n % 100 == 0:
            print(f"  frame {n}  flight t = {t:6.2f} s", flush=True)
        n += 1
    for w in writers.values():
        w.close()
    if dash is not None:
        dash.close()

    m = pd.DataFrame(metrics)
    m.to_csv(out / "overlay_metrics.csv", index=False)
    print(f"{n} frames rendered ({n_nopose} without camera pose), video {v0:.2f}-{v1:.2f} s -> {out}")
    if len(m):
        print("Real vs sim in the image, per phase (median / 95th percentile pixel distance, 3D RMS):")
        for (i, p), g in m[m["phase"].isin(names)].groupby(["id", "phase"], sort=False):
            e = g["err_px"].dropna()
            print(f"  CF{int(i):02d} {p:24s} {np.median(e) if len(e) else np.nan:6.1f} / "
                  f"{np.percentile(e, 95) if len(e) else np.nan:6.1f} px   "
                  f"{1e3 * np.sqrt(np.mean(g['err_3d_m'] ** 2)):6.1f} mm")


if __name__ == "__main__":
    main()
