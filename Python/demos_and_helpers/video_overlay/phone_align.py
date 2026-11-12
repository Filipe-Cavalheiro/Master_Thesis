"""
Step 2 of the phone-camera correction: align a simulated (Python, pinhole)
camera with the real phone video and tell how the phone rigid body must be
corrected in Motive so that, from then on, its streamed pose IS the camera.

Input: the capture of phone_capture.py (phone ID 0 + static object ID 1, a
few still "stations") and the phone video filmed during it.

For every station the virtual camera is placed at

    T_wc = T_wb @ T_bc        T_wb: phone rigid body (Motive, world Z-up)
                              T_bc: correction, rigid body -> camera (unknown)

and the simulated view (floor grid, world axes, the object as a Crazyflie
wireframe, its markers) is drawn over the real frame. T_bc is found by:
  * clicking, on the real frames, the object's markers (glowing with the
    phone torch) and/or the object's pivot; or "Snap": detect the bright
    marker blobs near the predicted ones automatically;
  * "Solve": least squares on all clicks of all stations (reprojection);
  * the six sliders: manual fine-tuning in the camera frame, if needed.

Result (also saved to <capture dir>/phone_correction.json):
  * the Motive correction of the phone rigid body pivot, in the rigid body's
    LOCAL Motive axes: translate the pivot by (X, Y, Z) mm, then rotate it.
    After it the rigid body frame is the camera frame with the convention
        Motive local:  +Z forward (optical axis), +Y up, +X left
        world Z-up:    +x forward, +y left, +z up   (identity = level, looking along +x)
    i.e. camera_T_wc() below turns the streamed pose straight into the camera.
  * extrinsics.json (T_bc, OpenCV optical frame) for the other video_overlay tools.

    python phone_align.py sessions/phone_calib --video phone.mp4 --first-time 5.2
    python phone_align.py sessions/phone_calib --video phone.mp4 --times 5.2 14.8 23.1 --intrinsics intrinsics.json
    python phone_align.py sessions/phone_calib --images st0.jpg st1.jpg st2.jpg --hfov 68
    python phone_align.py sessions/phone_calib --video phone.mp4 --solve      # no GUI: solve from the saved clicks
    python phone_align.py sessions/phone_calib2 --video v.mp4 --corrected     # check after the Motive change: ~0

--first-time: video time [s] of station 0; the other stations are predicted
from the mocap times (adjust each one with the "video t" slider if needed).

GUI keys: n / p  next / previous station        Tab  next point to click
          left click  place the current point   right click  delete the nearest click
          , / .  -/+ 1 frame    < / >  -/+ 1 s   v  view: overlay / sim only / real only
          s  solve    g  snap markers    b  bake sliders    w  save    h  help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import Intrinsics, inv_T, look_at, make_T, save_extrinsics, video_info  # noqa: E402

# ============================================================
# FRAMES / CONVENTIONS
# ============================================================

# Camera convention of the corrected rigid body: x forward, y left, z up
# ("fwd" frame). Columns = OpenCV optical axes (right, down, forward) in it.
R_FWD_CV = np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]])
# Motive (Y-up) -> world Z-up of the flight code: v_zup = P v_motive, (x, y, z) = (Zm, Xm, Ym)
P_M2Z = np.array([[0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
T_BC_NOMINAL = make_T(R_FWD_CV, np.zeros(3))   # rigid body already corrected: T_bc = this

COL_GRID = (150, 150, 150)
COL_DRONE = (30, 140, 255)     # orange (sim)
COL_MARKER = (255, 255, 0)     # cyan
COL_CLICK = (40, 40, 255)      # red
COL_RESID = (0, 230, 255)      # yellow


def camera_T_wc(pos_zup, quat_zup):
    """
    Camera pose (OpenCV optical frame) from the streamed pose of the CORRECTED
    phone rigid body (world Z-up, as natnet_funcs.py converts it). Use this
    for the virtual camera during the flights.
    """
    T_wb = make_T(Rotation.from_quat(quat_zup).as_matrix(), pos_zup)
    return T_wb @ T_BC_NOMINAL


def motive_correction(T_bc):
    """
    T_bc (rigid body <- OpenCV optical frame, world Z-up axes) -> the pivot
    change to make in Motive: translation [mm] and rotation in the rigid
    body's local Motive (Y-up) axes, so that new_pose = old_pose * correction
    is the camera with the convention above (+Z forward, +Y up, +X left).
    """
    R_bf = T_bc[:3, :3] @ R_FWD_CV.T          # rigid body <- fwd camera frame
    t_b = T_bc[:3, 3]
    R_m = P_M2Z.T @ R_bf @ P_M2Z
    t_m = P_M2Z.T @ t_b
    rot = Rotation.from_matrix(R_m)
    return {
        "translate_mm_local_XYZ": (1e3 * t_m).round(1).tolist(),
        "rotate_deg_intrinsic_XYZ": rot.as_euler("XYZ", degrees=True).round(2).tolist(),
        "rotate_deg_extrinsic_xyz": rot.as_euler("xyz", degrees=True).round(2).tolist(),
        "rotate_quat_xyzw": rot.as_quat().round(6).tolist(),
        "rotate_axis_angle": {"axis": (rot.as_rotvec() / max(rot.magnitude(), 1e-12)).round(4).tolist(),
                              "angle_deg": round(float(np.degrees(rot.magnitude())), 2)},
        "lens_distance_from_pivot_mm": round(float(1e3 * np.linalg.norm(t_m)), 1),
        "T_correction_motive_local": make_T(R_m, t_m).tolist(),
    }


def print_motive_correction(c, corrected=False):
    t, e = c["translate_mm_local_XYZ"], c["rotate_deg_intrinsic_XYZ"]
    print("\n" + "=" * 72)
    print(("RESIDUAL " if corrected else "") + "CORRECTION OF THE PHONE RIGID BODY IN MOTIVE "
          "(rigid body LOCAL axes, Motive Y-up)")
    print(f"  1. translate pivot  X {t[0]:+8.1f} mm   Y {t[1]:+8.1f} mm   Z {t[2]:+8.1f} mm")
    print(f"  2. then rotate      X {e[0]:+8.2f} deg, then Y {e[1]:+8.2f} deg, then Z {e[2]:+8.2f} deg"
          "  (about the rotating local axes)")
    a = c["rotate_axis_angle"]
    print(f"     = {a['angle_deg']:.2f} deg about local axis {a['axis']}, quaternion xyzw {c['rotate_quat_xyzw']}")
    print(f"  lens optical centre {c['lens_distance_from_pivot_mm']:.1f} mm from the current pivot")
    print("  Afterwards the rigid body is the camera: +Z forward (optical axis), +Y up, +X left.")
    print("  Check: re-run phone_capture.py, then phone_align.py --corrected: the residual must be ~0.")
    print("=" * 72)


# ============================================================
# CAPTURE
# ============================================================

def load_capture(path):
    """capture.json -> list of stations {T_wb, objects {id: T_wo}, points [(name, xyz)], t_mid}."""
    d = json.loads(Path(path).read_text())
    pid = str(d["phone_id"])
    out = []
    for st in d["stations"]:
        b = st["bodies"]
        T = lambda s: make_T(Rotation.from_quat(s["quat"]).as_matrix(), s["pos"])  # noqa: E731
        objs = {k: T(v) for k, v in b.items() if k != pid}
        pts = [(f"pivot{k}", T_o[:3, 3].copy()) for k, T_o in objs.items()]
        pts += [(name, np.array(m["pos"], float)) for name, m in st.get("markers", {}).items()]
        out.append({"index": st["index"], "T_wb": T(b[pid]), "objects": objs, "points": pts,
                    "t_mid": 0.5 * (st["t_start"] + st["t_end"])})
    if not out:
        raise SystemExit(f"No station in {path}")
    return out, d


# ============================================================
# SOLVER
# ============================================================

def project_body(P_b, T_bc, K, dist):
    """Points in the rigid-body frame -> pixels and depth."""
    pc = (P_b - T_bc[:3, 3]) @ T_bc[:3, :3]
    img, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return img.reshape(-1, 2), pc[:, 2]


def initial_guess(stations):
    """Camera at the rigid body, looking at the object(s), image up = world up: averaged over the stations."""
    Rs = []
    for st in stations:
        target = np.mean([p for _, p in st["points"]], axis=0)
        eye = st["T_wb"][:3, 3]
        if np.linalg.norm(target - eye) < 1e-3:
            continue
        Rs.append(st["T_wb"][:3, :3].T @ look_at(eye, target))
    R = Rotation.from_matrix(np.array(Rs)).mean().as_matrix()
    return make_T(R, np.zeros(3))


def collect(stations, clicks):
    """Clicks -> arrays (points in the rigid-body frame, pixels, station index, name)."""
    P_b, uv, ks, names = [], [], [], []
    for k, st in enumerate(stations):
        pts = dict(st["points"])
        T_bw = inv_T(st["T_wb"])
        for name, q in clicks.get(k, {}).items():
            if name in pts:
                P_b.append(T_bw[:3, :3] @ pts[name] + T_bw[:3, 3])
                uv.append(q)
                ks.append(k)
                names.append(name)
    return np.array(P_b, float).reshape(-1, 3), np.array(uv, float).reshape(-1, 2), np.array(ks, int), names


def solve(stations, clicks, intr: Intrinsics, T_init, fit_focal=False, sigma_px=1.5,
          prior_t=0.10, prior_rot_deg=30.0, prior_f=0.3):
    """
    T_bc (and a focal scale) from the clicks of all stations: robust least
    squares on the reprojection error, with weak priors (lens within ~10 cm
    of the pivot, rotation near the start value, focal near the given one)
    so that a single station still gives a usable (if partial) answer.
    Returns T_bc, focal scale, stats (incl. which DOF the data determine).
    """
    P_b, uv, ks, names = collect(stations, clicks)
    if len(P_b) == 0:
        raise ValueError("No clicks yet")
    R0 = Rotation.from_matrix(T_init[:3, :3])
    pr_r, pr_t = np.radians(prior_rot_deg), prior_t
    n_par = 7 if fit_focal else 6

    def unpack(x):
        T = make_T((R0 * Rotation.from_rotvec(x[:3])).as_matrix(), x[3:6])
        s = np.exp(x[6]) if fit_focal else 1.0
        return T, s

    def K_of(s):
        K = intr.K.copy()
        K[0, 0] *= s
        K[1, 1] *= s
        return K

    def residual(x, with_prior=True):
        T, s = unpack(x)
        img, z = project_body(P_b, T, K_of(s), intr.dist)
        r = (img - uv) / sigma_px
        r[z <= 0.02] = 1e3
        out = [r.ravel()]
        if with_prior:
            out += [x[:3] / pr_r, x[3:6] / pr_t]
            if fit_focal:
                out.append([x[6] / prior_f])
        return np.concatenate(out)

    x0 = np.zeros(n_par)            # rotation relative to T_init, translation absolute
    x0[3:6] = T_init[:3, 3]
    sol = least_squares(residual, x0, loss="soft_l1", f_scale=3.0, x_scale="jac")
    T_bc, s = unpack(sol.x)

    img, _ = project_body(P_b, T_bc, K_of(s), intr.dist)
    e = np.linalg.norm(img - uv, axis=1)
    stats = {"n_clicks": int(len(e)), "n_stations": int(len(set(ks.tolist()))),
             "rms_px": float(np.sqrt(np.mean(e ** 2))), "max_px": float(e.max()),
             "per_point_px": {f"{k}:{n}": round(float(v), 2) for k, n, v in zip(ks, names, e)},
             "focal_scale": float(s)}
    # posterior std (pixel noise = max(rms, 0.5 px)) and data-only observability
    J = sol.jac
    try:
        noise = max(stats["rms_px"], 0.5) / sigma_px
        cov = np.linalg.inv(J.T @ J) * noise ** 2
        sd = np.sqrt(np.diag(cov))
        stats["std_rot_deg"] = np.degrees(sd[:3]).round(3).tolist()
        stats["std_trans_mm"] = (1e3 * sd[3:6]).round(1).tolist()
        prior_sd = np.r_[[pr_r] * 3, [pr_t] * 3, [prior_f] if fit_focal else []]
        labels = ["rot x", "rot y", "rot z", "t x", "t y", "t z", "focal"][:n_par]
        stats["undetermined"] = [lab for lab, a, b in zip(labels, sd, prior_sd) if a > 0.3 * b]
    except np.linalg.LinAlgError:
        stats["undetermined"] = ["all (singular)"]
    return T_bc, s, stats


def print_stats(stats):
    print(f"Solve: {stats['n_clicks']} clicks on {stats['n_stations']} station(s): "
          f"RMS {stats['rms_px']:.2f} px, max {stats['max_px']:.2f} px, focal scale {stats['focal_scale']:.4f}")
    if "std_trans_mm" in stats:
        print(f"  1-sigma: rotation {stats['std_rot_deg']} deg (body axes), translation {stats['std_trans_mm']} mm")
    if stats.get("undetermined"):
        print(f"  NOT determined by the data (prior dominates): {stats['undetermined']} "
              "-> add stations at other positions/heights, or more points per station")


# ============================================================
# FRAMES
# ============================================================

class Frames:
    def __init__(self, video=None, images=None):
        self.video, self.images = video, images
        self.cache = {}
        if video:
            self.cap = cv2.VideoCapture(str(video))
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open {video}")
            info = video_info(video)
            self.fps = info["fps"] or 30.0
            self.duration = info["frames"] / self.fps if info["frames"] > 0 else 3600.0
        else:
            self.fps, self.duration = 1.0, 0.0

    def get(self, k, t):
        key = (k, None) if self.images else (None, round(t, 4))
        if key not in self.cache:
            if self.images:
                fr = cv2.imread(str(self.images[k]))
                if fr is None:
                    raise SystemExit(f"Cannot read {self.images[k]}")
            else:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1e3)
                ok, fr = self.cap.read()
                if not ok:
                    fr = np.zeros((int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720,
                                   int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280, 3), np.uint8)
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = fr
        return self.cache[key]

    def size(self):
        fr = self.get(0, 0.0)
        return fr.shape[1], fr.shape[0]


# ============================================================
# SIMULATED VIEW
# ============================================================

def crazyflie_segments():
    """Crazyflie 2.x wireframe in its body frame (x forward, z up): arms, rotor discs, deck."""
    segs = []
    a = 0.046 / np.sqrt(2)                       # 92 mm motor to motor
    motors = [np.array([sx * a, sy * a, 0.0]) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    for m in motors:
        segs.append((np.zeros(3), m))
        c = [m + np.array([0.0235 * np.cos(u), 0.0235 * np.sin(u), 0.02]) for u in np.linspace(0, 2 * np.pi, 17)]
        segs += list(zip(c[:-1], c[1:]))
    d = 0.015
    sq = [np.array(p) for p in ((d, d, 0.01), (d, -d, 0.01), (-d, -d, 0.01), (-d, d, 0.01), (d, d, 0.01))]
    segs += list(zip(sq[:-1], sq[1:]))
    segs.append((np.array([0, 0, 0.01]), np.array([0.03, 0, 0.01])))    # heading
    return segs


def floor_segments(half=3.0, step=0.5):
    segs = []
    for v in np.arange(-half, half + 1e-9, step):
        segs.append((np.array([v, -half, 0.0]), np.array([v, half, 0.0])))
        segs.append((np.array([-half, v, 0.0]), np.array([half, v, 0.0])))
    return segs


def proj_w(P_w, T_wc, K, dist):
    T_cw = inv_T(T_wc)
    pc = np.atleast_2d(P_w) @ T_cw[:3, :3].T + T_cw[:3, 3]
    uv = np.full((len(pc), 2), np.nan)
    ok = pc[:, 2] > 0.05
    if ok.any():
        xn = pc[ok, :2] / pc[ok, 2:3]
        good = np.all(np.abs(xn) < 3.0, axis=1)     # keep the distortion model sane
        idx = np.flatnonzero(ok)[good]
        if len(idx):
            img, _ = cv2.projectPoints(pc[idx].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
            uv[idx] = img.reshape(-1, 2)
    return uv, pc[:, 2]


def draw_segs(img, segs, T_wc, K, dist, color, thick, T_wo=None, n=12):
    for a, b in segs:
        P = np.linspace(a, b, n)
        if T_wo is not None:
            P = P @ T_wo[:3, :3].T + T_wo[:3, 3]
        uv, _ = proj_w(P, T_wc, K, dist)
        for p, q in zip(uv[:-1], uv[1:]):
            if np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.all(np.abs(np.r_[p, q]) < 2e4):
                cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, thick, cv2.LINE_AA)


def disp_K(K, s):
    """Intrinsics of the image resized by s (pixel-centre convention)."""
    Kd = K.copy()
    Kd[0, 0] *= s
    Kd[1, 1] *= s
    Kd[0, 2] = s * (K[0, 2] + 0.5) - 0.5
    Kd[1, 2] = s * (K[1, 2] + 0.5) - 0.5
    return Kd


def render(frame, st, T_bc, intr: Intrinsics, focal_scale=1.0, clicks=None, current=None,
           view="overlay", alpha=0.75, max_width=1600, object_model="crazyflie"):
    """Real frame + simulated camera view (or sim only / real only), resized to <= max_width."""
    h, w = frame.shape[:2]
    s = min(1.0, max_width / w)
    base = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
    if view == "real":
        return base, s
    img = (base.astype(np.float32) * (alpha if view == "overlay" else 0.0)).astype(np.uint8)
    K = intr.K.copy()
    K[0, 0] *= focal_scale
    K[1, 1] *= focal_scale
    K = disp_K(K, s)
    dist = intr.dist
    T_wc = st["T_wb"] @ T_bc
    th = max(1, int(round(2 * s * w / 1280)))
    draw_segs(img, floor_segments(), T_wc, K, dist, COL_GRID, 1)
    for ax, col in zip(np.eye(3), ((0, 0, 255), (0, 200, 0), (255, 80, 0))):
        draw_segs(img, [(np.zeros(3), 0.5 * ax)], T_wc, K, dist, col, th + 1)
    for T_wo in st["objects"].values():
        if object_model == "crazyflie":
            draw_segs(img, crazyflie_segments(), T_wc, K, dist, COL_DRONE, th, T_wo=T_wo, n=4)
        else:
            draw_segs(img, [(np.zeros(3), 0.05 * a) for a in np.eye(3)], T_wc, K, dist, COL_DRONE, th, T_wo=T_wo)
    names = [n for n, _ in st["points"]]
    P = np.array([p for _, p in st["points"]])
    uv, z = proj_w(P, T_wc, K, dist)
    for name, q, d in zip(names, uv, z):
        if not np.all(np.isfinite(q)):
            continue
        c = tuple(np.round(q).astype(int))
        r = max(3, int(K[0, 0] * 0.008 / max(d, 0.05)))
        if name.startswith("pivot"):
            cv2.drawMarker(img, c, COL_MARKER, cv2.MARKER_CROSS, 4 * r, th)
        else:
            cv2.circle(img, c, r, COL_MARKER, th, cv2.LINE_AA)
        if name == current:
            cv2.circle(img, c, 3 * r + 8, (255, 255, 255), th, cv2.LINE_AA)
    for name, q in (clicks or {}).items():
        qd = s * (np.asarray(q) + 0.5) - 0.5
        c = tuple(np.round(qd).astype(int))
        cv2.drawMarker(img, c, COL_CLICK, cv2.MARKER_TILTED_CROSS, 14, th)
        if name in names:
            p = uv[names.index(name)]
            if np.all(np.isfinite(p)):
                cv2.line(img, c, tuple(np.round(p).astype(int)), COL_RESID, 1, cv2.LINE_AA)
    return img, s


# ============================================================
# AUTO SNAP
# ============================================================

def snap(frames, stations, times, T_bc, intr, focal_scale, clicks, radius=30):
    """Bright blob nearest to every predicted marker (not the pivots); manual clicks are kept."""
    from calibrate_extrinsics import detect_near
    K = intr.K.copy()
    K[0, 0] *= focal_scale
    K[1, 1] *= focal_scale
    n = 0
    for k, st in enumerate(stations):
        gray = cv2.cvtColor(frames.get(k, times[k]), cv2.COLOR_BGR2GRAY)
        T_wc = st["T_wb"] @ T_bc
        for name, p in st["points"]:
            if name.startswith("pivot") or name in clicks.get(k, {}):
                continue
            q, _ = proj_w(p, T_wc, K, intr.dist)
            if not np.all(np.isfinite(q[0])):
                continue
            c = detect_near(gray, q[0], radius)
            if c is not None:
                clicks.setdefault(k, {})[name] = [float(c[0]), float(c[1])]
                n += 1
    print(f"Snap: {n} marker(s) detected")
    return n


# ============================================================
# STATE / OUTPUT
# ============================================================

def save_all(root, stations, frames, times, T_bc, focal_scale, intr, clicks, stats, corrected):
    root = Path(root)
    state = {"times": times, "T_bc": T_bc.tolist(), "focal_scale": focal_scale,
             "clicks": {str(k): v for k, v in clicks.items()}}
    (root / "align_state.json").write_text(json.dumps(state, indent=2))
    K = intr.K.copy()
    K[0, 0] *= focal_scale
    K[1, 1] *= focal_scale
    Intrinsics(K, intr.dist, intr.width, intr.height, intr.rms_px).save(root / "phone_intrinsics.json")
    save_extrinsics(root / "extrinsics.json", T_bc, **{k: v for k, v in (stats or {}).items() if k != "per_point_px"})
    corr = motive_correction(T_bc)
    out = {"motive_correction": corr, "corrected_input": corrected,
           "camera_convention_after_correction": {"motive_local": "+Z forward, +Y up, +X left",
                                                  "world_zup": "+x forward, +y left, +z up",
                                                  "T_bc_opencv_after_correction": T_BC_NOMINAL.tolist()},
           "T_bc_opencv": T_bc.tolist(), "focal_scale": focal_scale, "stats": stats, "video_times": times}
    (root / "phone_correction.json").write_text(json.dumps(out, indent=2))
    for k, st in enumerate(stations):
        img, _ = render(frames.get(k, times[k]), st, T_bc, intr, focal_scale, clicks.get(k), max_width=1e9)
        cv2.imwrite(str(root / f"align_station{k}.png"), img)
    print(f"Saved {root}/phone_correction.json, extrinsics.json, phone_intrinsics.json, align_station*.png")
    print_motive_correction(corr, corrected)
    return corr


# ============================================================
# GUI
# ============================================================

class App:
    SLIDERS = [("right mm", -100, 100), ("down mm", -100, 100), ("fwd mm", -100, 100),
               ("tilt deg", -15, 15), ("pan deg", -15, 15), ("roll deg", -15, 15)]

    def __init__(self, args, stations, frames, intr, times, T_bc, focal_scale, clicks):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        self.plt = plt
        self.args, self.st, self.frames, self.intr = args, stations, frames, intr
        self.times, self.T_base, self.f = times, T_bc, focal_scale
        self.clicks, self.k, self.pi, self.view, self.stats = clicks, 0, 0, "overlay", None
        self.time_set = [args.times is not None] * len(stations)

        self.fig = plt.figure(figsize=(15, 9.5))
        self.fig.canvas.manager.set_window_title("phone_align")
        self.ax = self.fig.add_axes([0.01, 0.30, 0.98, 0.66])
        self.ax.set_axis_off()
        W, H = intr.width, intr.height
        self.im = self.ax.imshow(np.zeros((10, 10, 3), np.uint8), extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        self.sl = []
        for j, (name, lo, hi) in enumerate(self.SLIDERS):
            a = self.fig.add_axes([0.07 + 0.47 * (j // 3), 0.21 - 0.04 * (j % 3), 0.36, 0.025])
            s = Slider(a, name, lo, hi, valinit=0.0)
            s.on_changed(lambda _: self.redraw())
            self.sl.append(s)
        a = self.fig.add_axes([0.07, 0.08, 0.36, 0.025])
        self.s_time = Slider(a, "video t", 0.0, max(frames.duration, 1.0), valinit=times[0])
        self.s_time.on_changed(self.on_time)
        a = self.fig.add_axes([0.54, 0.08, 0.36, 0.025])
        self.s_alpha = Slider(a, "real img", 0.0, 1.0, valinit=0.75)
        self.s_alpha.on_changed(lambda _: self.redraw())
        self.btn = []
        for j, (lab, fn) in enumerate([("< station", lambda _: self.step(-1)), ("station >", lambda _: self.step(1)),
                                       ("Snap (g)", lambda _: self.do_snap()), ("Solve (s)", lambda _: self.do_solve()),
                                       ("Bake (b)", lambda _: self.bake()), ("Save (w)", lambda _: self.save())]):
            b = Button(self.fig.add_axes([0.07 + 0.145 * j, 0.015, 0.13, 0.04]), lab)
            b.on_clicked(fn)
            self.btn.append(b)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()
        print(__doc__.split("GUI keys:")[1])
        plt.show()

    # --- state
    def T_bc(self):
        v = [s.val for s in self.sl]
        d = make_T(Rotation.from_euler("xyz", v[3:], degrees=True).as_matrix(), np.array(v[:3]) / 1e3)
        return self.T_base @ d

    def points(self):
        return [n for n, _ in self.st[self.k]["points"]]

    def current(self):
        p = self.points()
        return p[self.pi % len(p)] if p else None

    def redraw(self):
        st = self.st[self.k]
        img, _ = render(self.frames.get(self.k, self.times[self.k]), st, self.T_bc(), self.intr, self.f,
                        self.clicks.get(self.k), self.current(), self.view, self.s_alpha.val,
                        object_model=self.args.object_model)
        self.im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        n = len(self.clicks.get(self.k, {}))
        rms = f"   solve RMS {self.stats['rms_px']:.2f} px" if self.stats else ""
        self.ax.set_title(f"station {self.k + 1}/{len(self.st)}   video t = {self.times[self.k]:.3f} s   "
                          f"click: {self.current()}  ({n} clicked here)   view: {self.view}{rms}", fontsize=10)
        self.fig.canvas.draw_idle()

    # --- events
    def on_time(self, val):
        self.times[self.k] = float(val)
        self.time_set[self.k] = True
        if self.k == 0:     # predict the stations not set by hand from the mocap times
            for j in range(1, len(self.st)):
                if not self.time_set[j]:
                    self.times[j] = self.times[0] + self.st[j]["t_mid"] - self.st[0]["t_mid"]
        self.redraw()

    def set_time(self, t):
        self.s_time.set_val(float(np.clip(t, 0.0, self.s_time.valmax)))

    def step(self, d):
        self.k = (self.k + d) % len(self.st)
        self.pi = 0
        self.s_time.eventson = False
        self.s_time.set_val(self.times[self.k])
        self.s_time.eventson = True
        self.redraw()

    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return      # zoom / pan active
        c = self.clicks.setdefault(self.k, {})
        if ev.button == 1 and self.current():
            c[self.current()] = [float(ev.xdata), float(ev.ydata)]
            self.pi += 1
        elif ev.button == 3 and c:
            q = np.array([ev.xdata, ev.ydata])
            name = min(c, key=lambda n: np.hypot(*(np.array(c[n]) - q)))
            del c[name]
        self.redraw()

    def on_key(self, ev):
        k = ev.key
        dt = 1.0 / self.frames.fps
        acts = {"n": lambda: self.step(1), "p": lambda: self.step(-1),
                "tab": lambda: setattr(self, "pi", self.pi + 1),
                ",": lambda: self.set_time(self.times[self.k] - dt), ".": lambda: self.set_time(self.times[self.k] + dt),
                "<": lambda: self.set_time(self.times[self.k] - 1), ">": lambda: self.set_time(self.times[self.k] + 1),
                "v": lambda: setattr(self, "view", {"overlay": "sim", "sim": "real", "real": "overlay"}[self.view]),
                "s": self.do_solve, "g": self.do_snap, "b": self.bake, "w": self.save,
                "h": lambda: print(__doc__.split("GUI keys:")[1])}
        if k in acts:
            acts[k]()
            self.redraw()

    # --- actions
    def bake(self):
        self.T_base = self.T_bc()
        for s in self.sl:
            s.eventson = False
            s.set_val(0.0)
            s.eventson = True

    def do_solve(self):
        self.bake()
        try:
            self.T_base, self.f, self.stats = solve(self.st, self.clicks, self.intr, self.T_base,
                                                    fit_focal=self.args.fit_focal,
                                                    prior_t=1e-4 if self.args.fix_translation else 0.10)
        except ValueError as e:
            print(e)
            return
        print_stats(self.stats)
        print_motive_correction(motive_correction(self.T_base), self.args.corrected)

    def do_snap(self):
        snap(self.frames, self.st, self.times, self.T_bc(), self.intr, self.f, self.clicks, self.args.snap_radius)

    def save(self):
        self.bake()
        save_all(self.args.capture_dir, self.st, self.frames, self.times, self.T_base, self.f, self.intr,
                 self.clicks, self.stats, self.args.corrected)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir", help="directory of phone_capture.py (capture.json)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="phone video filmed during the capture")
    src.add_argument("--images", nargs="+", help="one image per station instead of a video")
    ap.add_argument("--times", type=float, nargs="+", help="video time [s] of every station")
    ap.add_argument("--first-time", type=float, help="video time [s] of station 0 (others predicted)")
    ap.add_argument("--intrinsics", help="intrinsics.json (calibrate_intrinsics.py), recommended")
    ap.add_argument("--hfov", type=float, default=68.0,
                    help="without --intrinsics: field of view [deg] across the LONG image side (pinhole, no distortion)")
    ap.add_argument("--fit-focal", action="store_true", help="also estimate the focal length (needs >= 3 good stations)")
    ap.add_argument("--fix-translation", action="store_true",
                    help="lens at the rigid-body pivot (solve rotation [+ focal] only): use when the stations "
                         "cannot determine the lens offset (one point per station, similar distances)")
    ap.add_argument("--corrected", action="store_true",
                    help="the Motive rigid body is already corrected: start from the nominal camera; the result is the residual")
    ap.add_argument("--solve", action="store_true", help="no GUI: (snap,) solve from the saved clicks and save")
    ap.add_argument("--snap", action="store_true", help="with --solve: snap the markers first")
    ap.add_argument("--snap-radius", type=int, default=30, help="snap search radius [px]")
    ap.add_argument("--object-model", choices=["crazyflie", "axes"], default="crazyflie")
    ap.add_argument("--fresh", action="store_true", help="ignore align_state.json (saved clicks / times / estimate)")
    args = ap.parse_args()

    root = Path(args.capture_dir)
    stations, _ = load_capture(root / "capture.json")
    frames = Frames(args.video, args.images)
    if args.images and len(args.images) != len(stations):
        raise SystemExit(f"{len(args.images)} images for {len(stations)} stations")
    W, H = frames.size()

    if args.intrinsics:
        intr = Intrinsics.load(args.intrinsics)
        if (intr.width > intr.height) != (W > H):
            raise SystemExit("intrinsics.json and the video have different orientations (portrait / landscape)")
        intr = intr.scaled_to(W, H)
    else:
        f = 0.5 * max(W, H) / np.tan(np.radians(args.hfov) / 2)
        intr = Intrinsics(np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]]), np.zeros(5), W, H)
        print(f"No --intrinsics: pinhole with {args.hfov:.0f} deg across the long side (f = {f:.0f} px). "
              "Use calibrate_intrinsics.py or --fit-focal for an accurate result.")

    state_p = root / "align_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() and not args.fresh else {}
    clicks = {int(k): v for k, v in state.get("clicks", {}).items()}
    if args.times:
        if len(args.times) != len(stations):
            raise SystemExit(f"--times: {len(args.times)} values for {len(stations)} stations")
        times = list(args.times)
    elif args.first_time is not None:
        times = [args.first_time + s["t_mid"] - stations[0]["t_mid"] for s in stations]
    elif "times" in state and len(state["times"]) == len(stations):
        times = state["times"]
    else:
        times = [s["t_mid"] - stations[0]["t_mid"] for s in stations]
        if args.video:
            print("No --times / --first-time: station 0 assumed at t = 0; set it with the 'video t' slider.")
    if "T_bc" in state:
        T_bc = np.array(state["T_bc"])
    elif args.corrected:
        T_bc = T_BC_NOMINAL.copy()
    else:
        T_bc = initial_guess(stations)
    focal = float(state.get("focal_scale", 1.0))

    print(f"{len(stations)} station(s); points per station: {[len(s['points']) for s in stations]}")
    if args.solve:
        if args.snap:
            snap(frames, stations, times, T_bc, intr, focal, clicks, args.snap_radius)
        T_bc, focal, stats = solve(stations, clicks, intr, T_bc, fit_focal=args.fit_focal,
                                   prior_t=1e-4 if args.fix_translation else 0.10)
        print_stats(stats)
        save_all(root, stations, frames, times, T_bc, focal, intr, clicks, stats, args.corrected)
        return
    App(args, stations, frames, intr, times, T_bc, focal, clicks)


if __name__ == "__main__":
    main()
