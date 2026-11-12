"""
Shared code of the video overlay pipeline: log loading, frames and poses,
camera model / projection, time alignment and video I/O.

Conventions
-----------
* World frame: the OptiTrack frame converted to Z-up exactly as the flight
  code does (Crazy_Flies/*/natnet_funcs.py): (x, y, z) = (Zm, Xm, Ym) and
  quaternion (x, y, z, w) = (qz_m, qx_m, qy_m, qw_m). Same frame as the
  `opti_*` columns of the flight logs and as the Python simulation.
* Camera optical frame: OpenCV (x right, y down, z forward).
* Poses are (R, t) with p_parent = R @ p_child + t:
    T_wb  world <- camera rigid body (logged by Motive)
    T_bc  rigid body <- camera optical centre (hand-eye, extrinsics.json)
    T_wc  = T_wb @ T_bc
* Time bases:
    mocap time  = `time_s` of the recorder logs (record_mocap.py)
    video time  = presentation time of a frame in the video file
    t_mocap = scale * t_video + offset         (sync.json)
    flight log time + log_offset = mocap time   (estimate_log_offset)
    sim time 0  = start of the "Formation" phase of the flight log
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Columns of Crazy_Flies/*/trajectory_logger.py (TrajectoryLogger.COLUMNS)
TRAJ_COLUMNS = [
    "time_s", "flight_state", "id",
    "opti_x", "opti_y", "opti_z",
    "est_x", "est_y", "est_z",
    "est_vx", "est_vy", "est_vz",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "sp_x", "sp_y", "sp_z",
    "target_x", "target_y", "target_z",
]
POS_COLS = ["opti_x", "opti_y", "opti_z"]
CAMERA_COLUMNS = ["time_s", "frame", "motive_time_s", "id", "valid",
                  "x", "y", "z", "qx", "qy", "qz", "qw", "marker_error"]
EVENT_COLUMNS = ["time_s", "kind", "label"]

# Drawing colours (BGR): real = mocap, sim = simulation
COLOR_REAL = (80, 200, 60)      # green
COLOR_SIM = (30, 140, 255)      # orange
COLOR_TARGET = (220, 80, 220)   # magenta


# ============================================================
# MOTIVE -> Z-UP
# ============================================================

def motive_to_zup(pos, quat):
    """Motive (Y-up) position (x, y, z) and quaternion (x, y, z, w) -> world Z-up (natnet_funcs.py)."""
    mx, my, mz = pos
    qx, qy, qz, qw = quat
    return (mz, mx, my), (qz, qx, qy, qw)


# ============================================================
# POSES
# ============================================================

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera optical frame (OpenCV) at `eye` looking at `target`: R_wc."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ============================================================
# CAMERA MODEL
# ============================================================

@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    rms_px: float = float("nan")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["K"], float), np.array(d["dist"], float).ravel(),
                   int(d["width"]), int(d["height"]), float(d.get("rms_px", "nan")))

    def save(self, path, **extra):
        d = {"K": self.K.tolist(), "dist": self.dist.ravel().tolist(),
             "width": self.width, "height": self.height, "rms_px": self.rms_px, **extra}
        Path(path).write_text(json.dumps(d, indent=2))

    def scaled_to(self, width, height):
        """Same camera at another resolution (video recorded at a different size than the calibration)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return Intrinsics(K, self.dist.copy(), width, height, self.rms_px)


def load_extrinsics(path):
    """T_bc (rigid body <- camera optical frame) from extrinsics.json."""
    d = json.loads(Path(path).read_text())
    return np.array(d["T_bc"], float)


def save_extrinsics(path, T_bc, **extra):
    Path(path).write_text(json.dumps({"T_bc": T_bc.tolist(), **extra}, indent=2))


def project(points_w, T_wc, intr: Intrinsics, margin=0.25):
    """
    World points (N, 3) -> pixels (N, 2) and a validity mask (in front of the
    camera and inside the image enlarged by `margin`; the distortion model is
    meaningless far outside the calibrated field of view). Also returns the
    depth (N,).
    """
    P = np.atleast_2d(np.asarray(points_w, float))
    T_cw = inv_T(T_wc)
    pc = P @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = pc[:, 2]
    uv = np.full((len(P), 2), np.nan)
    front = z > 0.05
    if not np.any(front):
        return uv, front, z
    # field-of-view gate on the undistorted normalized coordinates
    xn = pc[front, :2] / z[front, None]
    K = intr.K
    lo = (-margin * np.array([intr.width, intr.height]) - K[:2, 2]) / np.diag(K)[:2]
    hi = ((1 + margin) * np.array([intr.width, intr.height]) - K[:2, 2]) / np.diag(K)[:2]
    inside = np.all((xn > lo) & (xn < hi), axis=1)
    idx = np.flatnonzero(front)[inside]
    valid = np.zeros(len(P), bool)
    if len(idx):
        img, _ = cv2.projectPoints(pc[idx].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, intr.dist)
        uv[idx] = img.reshape(-1, 2)
        valid[idx] = True
    return uv, valid, z


# ============================================================
# LOGS
# ============================================================

def read_traj(path):
    """Flight/recorder log (TrajectoryLogger format) -> DataFrame with string ids and clean flight_state."""
    df = pd.read_csv(path, dtype={"id": str, "flight_state": str})
    df["flight_state"] = df["flight_state"].fillna("").str.strip()
    df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


class Tracks:
    """
    Positions of several drones over time, interpolated at any time:
    tracks.at(t) -> {id: (3,) position or NaN outside the logged interval}.
    """

    def __init__(self, data: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.data = {}
        for i, (t, p) in data.items():
            order = np.argsort(t)
            t, p = np.asarray(t, float)[order], np.asarray(p, float)[order]
            keep = np.r_[True, np.diff(t) > 0] & np.all(np.isfinite(p), axis=1)
            self.data[i] = (t[keep], p[keep])
        self.ids = sorted(self.data, key=lambda s: (len(s), s))

    @classmethod
    def from_traj(cls, df, time_shift=0.0, phase=None):
        if phase is not None:
            df = df[df["flight_state"] == phase]
        return cls({i: (g["time_s"].to_numpy(float) + time_shift, g[POS_COLS].to_numpy(float))
                    for i, g in df.groupby("id")})

    def shifted(self, dt):
        return Tracks({i: (t + dt, p) for i, (t, p) in self.data.items()})

    def at(self, t):
        out = {}
        for i, (ts, p) in self.data.items():
            if len(ts) == 0 or t < ts[0] or t > ts[-1]:
                out[i] = np.full(3, np.nan)
            else:
                out[i] = np.array([np.interp(t, ts, p[:, a]) for a in range(3)])
        return out

    def sample(self, i, times):
        ts, p = self.data[i]
        out = np.column_stack([np.interp(times, ts, p[:, a], left=np.nan, right=np.nan) for a in range(3)])
        return out

    def span(self):
        return (min(t[0] for t, _ in self.data.values()), max(t[-1] for t, _ in self.data.values()))


class CameraTrack:
    """Rigid-body pose of the camera over mocap time (camera log of record_mocap.py)."""

    def __init__(self, t, pos, quat):
        order = np.argsort(t)
        t, pos, quat = np.asarray(t, float)[order], np.asarray(pos, float)[order], np.asarray(quat, float)[order]
        keep = np.r_[True, np.diff(t) > 0]
        self.t, self.pos, self.quat = t[keep], pos[keep], quat[keep]
        self.slerp = Slerp(self.t, Rotation.from_quat(self.quat))

    @classmethod
    def load(cls, path, rb_id=None):
        df = pd.read_csv(path)
        if rb_id is not None:
            df = df[df["id"] == rb_id]
        df = df[df["valid"].astype(bool)]
        if df.empty:
            raise ValueError(f"No valid camera pose in {path}")
        return cls(df["time_s"].to_numpy(), df[["x", "y", "z"]].to_numpy(), df[["qx", "qy", "qz", "qw"]].to_numpy())

    def T_wb(self, t):
        """World <- rigid body at mocap time t (None outside the log or across a tracking gap > 0.1 s)."""
        if t < self.t[0] or t > self.t[-1]:
            return None
        k = np.searchsorted(self.t, t)
        if 0 < k < len(self.t) and self.t[k] - self.t[k - 1] > 0.1:
            return None
        p = np.array([np.interp(t, self.t, self.pos[:, a]) for a in range(3)])
        return make_T(self.slerp([t]).as_matrix()[0], p)

    def motion(self, t0, t1):
        """Max translation [m] and rotation [deg] of the rigid body between t0 and t1 (static-camera check)."""
        m = (self.t >= t0) & (self.t <= t1)
        if m.sum() < 2:
            return 0.0, 0.0
        dp = np.linalg.norm(self.pos[m] - self.pos[m].mean(0), axis=1).max()
        rots = Rotation.from_quat(self.quat[m])
        dr = np.degrees((rots.mean().inv() * rots).magnitude()).max()
        return float(dp), float(dr)


def read_events(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.read_csv(p)


def formation_start(df):
    """Start time of the Formation phase in a flight log (sim time 0, Crazy_Flies/flight_sim.py)."""
    f = df[df["flight_state"] == "Formation"]
    if f.empty:
        raise ValueError("No 'Formation' phase in the flight log")
    return float(f["time_s"].min())


def estimate_log_offset(flight: Tracks, mocap: Tracks, max_offset=600.0, step=0.02):
    """
    Offset d [s] such that flight(t) ~ mocap(t + d): the flight log (its own
    time base, started by logger.begin()) against the recorder log (mocap time),
    by minimizing the mean distance between the positions of the common drones.
    Coarse grid search over the overlap, then a fine search around the minimum.
    Returns (d, mean residual [m]).
    """
    ids = [i for i in flight.ids if i in mocap.data]
    if not ids:
        raise ValueError("No drone id common to the flight log and the mocap log")
    f0, f1 = flight.span()
    m0, m1 = mocap.span()
    tf = np.arange(f0, f1, 0.1)
    # use the moving part of the flight (takeoff, formation): more informative
    fp = {i: flight.sample(i, tf) for i in ids}

    def cost(d):
        errs = []
        for i in ids:
            pm = mocap.sample(i, tf + d)
            e = np.linalg.norm(pm - fp[i], axis=1)
            errs.append(e[np.isfinite(e)])
        e = np.concatenate(errs)
        if len(e) < 0.3 * len(tf) * len(ids):    # too little overlap
            return np.inf
        return float(np.mean(e))

    lo, hi = max(-max_offset, m0 - f1), min(max_offset, m1 - f0)
    grid = np.arange(lo, hi, 0.25)
    c = np.array([cost(d) for d in grid])
    d0 = grid[np.nanargmin(c)]
    fine = np.arange(d0 - 0.5, d0 + 0.5, step)
    cf = np.array([cost(d) for d in fine])
    k = int(np.nanargmin(cf))
    d = fine[k]
    if 0 < k < len(fine) - 1 and np.all(np.isfinite(cf[k - 1:k + 2])):   # parabolic refinement
        a, b, cc = cf[k - 1:k + 2]
        den = a - 2 * b + cc
        if den > 0:
            d += 0.5 * step * (a - cc) / den
    return float(d), float(cost(d))


# ============================================================
# SESSION
# ============================================================

class Session:
    """
    A filming session directory. session.json (written by record_mocap.py,
    editable) names its files; defaults:
        video.mp4                 the camera video
        mocap_trajectory.csv      drone poses (TrajectoryLogger columns), mocap time
        mocap_camera.csv          camera rigid-body poses, mocap time
        mocap_events.csv          sync events (flashes, key presses), mocap time
        intrinsics.json           calibrate_intrinsics.py
        extrinsics.json           calibrate_extrinsics.py
        sync.json                 sync.py
        flight_log / sim          flight log of the flight code and its
                                  simulation (Crazy_Flies/flight_sim.py, <log>_sim.csv)
    """

    DEFAULTS = {
        "video": "video.mp4",
        "drones": "mocap_trajectory.csv",
        "camera": "mocap_camera.csv",
        "events": "mocap_events.csv",
        "intrinsics": "intrinsics.json",
        "extrinsics": "extrinsics.json",
        "sync": "sync.json",
        "flight_log": None,
        "sim": None,
        "camera_id": None,
    }

    def __init__(self, root):
        self.root = Path(root)
        self.cfg = dict(self.DEFAULTS)
        f = self.root / "session.json"
        if f.exists():
            self.cfg.update(json.loads(f.read_text()))

    def path(self, key):
        v = self.cfg.get(key)
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else (self.root / p)

    def save(self):
        (self.root / "session.json").write_text(json.dumps(self.cfg, indent=2))

    def sync(self):
        d = json.loads(self.path("sync").read_text())
        return float(d.get("scale", 1.0)), float(d["offset"])

    def camera_track(self):
        return CameraTrack.load(self.path("camera"), self.cfg.get("camera_id"))

    def real_tracks(self):
        """Real drone tracks in mocap time: the recorder log, else the flight log (then assumed in mocap time)."""
        p = self.path("drones")
        if p is not None and p.exists():
            return Tracks.from_traj(read_traj(p))
        return Tracks.from_traj(read_traj(self.path("flight_log")))

    def sim_tracks(self):
        """
        Simulated drone tracks in mocap time, or None. Sim time 0 = start of
        the Formation phase of the flight log; the flight log is placed in
        mocap time by matching it with the recorder log (estimate_log_offset).
        Returns (tracks, info dict).
        """
        sim_p, flight_p = self.path("sim"), self.path("flight_log")
        if sim_p is None or flight_p is None or not sim_p.exists():
            return None, {}
        flight = read_traj(flight_p)
        t_form = formation_start(flight)
        drones_p = self.path("drones")
        if drones_p is not None and drones_p.exists() and drones_p.resolve() != flight_p.resolve():
            d, res = estimate_log_offset(Tracks.from_traj(flight), Tracks.from_traj(read_traj(drones_p)))
        else:
            d, res = 0.0, 0.0
        sim = Tracks.from_traj(read_traj(sim_p), time_shift=t_form + d)
        target = read_traj(sim_p)
        tgt = None
        if {"target_x", "target_y", "target_z"} <= set(target.columns):
            g = target[target["id"] == target["id"].iloc[0]]
            tgt = Tracks({"target": (g["time_s"].to_numpy(float) + t_form + d,
                                     g[["target_x", "target_y", "target_z"]].to_numpy(float))})
        info = {"log_offset_s": d, "log_offset_residual_m": res,
                "formation_start_mocap_s": t_form + d,
                "formation_end_mocap_s": sim.span()[1]}
        return (sim, tgt), info


# ============================================================
# VIDEO I/O
# ============================================================

def video_frames(path, start_s=0.0, end_s=None, step=1):
    """
    Yields (index, video time [s], BGR frame). Uses the container timestamps
    (CAP_PROP_POS_MSEC): phones often record variable frame rate.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1e3)
    k = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # FFmpeg backend: timestamp of the frame just decoded (0 for the first one)
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1e3 if (ms > 0 or k == 0) else k / fps
        if end_s is not None and t > end_s:
            break
        if (k % step) == 0:
            yield k, t, frame
        k += 1
    cap.release()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


class VideoWriter:
    """mp4 writer: OpenCV (mp4v), then re-encoded to H.264 with ffmpeg when available (YouTube/PowerPoint)."""

    def __init__(self, path, fps, size):
        import shutil
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = shutil.which("ffmpeg")
        self.tmp = self.path.with_suffix(".tmp.mp4") if self.ffmpeg else self.path
        self.w = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.w.isOpened():
            raise IOError(f"Cannot write {self.tmp}")

    def write(self, frame):
        self.w.write(frame)

    def close(self):
        import subprocess
        self.w.release()
        if self.ffmpeg:
            r = subprocess.run([self.ffmpeg, "-y", "-loglevel", "error", "-i", str(self.tmp), "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(self.path)])
            if r.returncode == 0:
                self.tmp.unlink()
            else:
                self.tmp.replace(self.path)


# ============================================================
# DRAWING
# ============================================================

def draw_polyline(img, uv, color, thickness=2, fade=True):
    """Trail through the valid pixels (NaN breaks the line), older points fainter."""
    n = len(uv)
    for k in range(1, n):
        a, b = uv[k - 1], uv[k]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        c = tuple(int((0.3 + 0.7 * k / n) * ch) for ch in color) if fade else color
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), c, thickness, cv2.LINE_AA)


def marker_radius(intr, depth, size_m=0.12, lo=4, hi=60):
    """On-screen radius [px] of a sphere of diameter size_m at `depth`."""
    if not np.isfinite(depth) or depth <= 0:
        return lo
    return int(np.clip(intr.K[0, 0] * size_m / 2 / depth, lo, hi))


def put_label(img, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def floor_grid_segments(half=2.5, step=0.5):
    segs = []
    for v in np.arange(-half, half + 1e-9, step):
        segs.append(((v, -half, 0.0), (v, half, 0.0)))
        segs.append(((-half, v, 0.0), (half, v, 0.0)))
    return segs


def draw_world_segments(img, segs, T_wc, intr, color=(200, 200, 200), thickness=1, n=20):
    """3D segments, subdivided so that the lens distortion bends them correctly."""
    for a, b in segs:
        pts = np.linspace(a, b, n)
        uv, valid, _ = project(pts, T_wc, intr, margin=0.1)
        uv[~valid] = np.nan
        draw_polyline(img, uv, color, thickness, fade=False)


def angle_deg(R):
    return math.degrees(Rotation.from_matrix(R).magnitude())
