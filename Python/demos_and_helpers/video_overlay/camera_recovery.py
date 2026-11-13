"""
Camera pose where the OptiTrack lost the phone rigid body, from the video and the drones.
Used by flight_overlay.py (--recover auto, default; --static).

* Gaps (the camera was tracked before and/or after): for every video frame of a gap
  - rotation: frame-to-frame rotation from the background features (KLT, rotation-only
    fit, RANSAC), corrected every 0.1 s by the rotation of the essential matrix (not
    biased by the parallax of the translation), integrated forward from the last tracked
    pose and backward from the first recovered one, and blended so that it closes on both;
  - optical centre: translation directions of the essential geometry every 0.1 s, scaled
    so that the path closes on the second anchor (one anchor only: held at it);
  - refinement on the drones: the OptiTrack positions of the drones, projected, must fall
    on the drones in the video (moving blobs after warping the previous frame onto the
    current one with the background homography: the spinning propellers stay, the
    background parallax leaves thin lines). Smooth position / rotation corrections
    (knots every 0.5 s, zero at the anchors), robust least squares.
  Test (Fibonacci flight of 12 Nov, a tracked 6 s stretch hidden and recovered; the phone
  walked ~1.4 m): drones 4.2 px from their detections (960-px frames; the OptiTrack poses
  themselves 12 px, hand-eye calibration bias), pose vs OptiTrack 0.22 m / 3 deg median.

* Never tracked, static camera (tripod): one pose, the focal length, k1 and the time
  offset from the drones (moving blobs against the median background). Initial guess:
  camera on rings around the drones looking at them, focal length and offset (+- 1 s
  around the rough one) on grids; then robust least squares. Test (MPC flight of 13 Nov,
  0.5x ultra wide, no pose ever tracked): 2.1 px median, camera within 2 cm and the same
  offset as a solve seeded by hand-clicked drones.

Both return the camera in flight-log time; flight_overlay.py uses it as a tracked camera.
A hand-held camera that was never tracked is not handled (structure from motion).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from overlay_lib import CameraTrack, Intrinsics, inv_T, make_T, video_frames, video_info

WORK_W = 960            # processing width (features, detections)


# ============================================================
# helpers
# ============================================================

def _pos(track, t):
    ts = track["t"]
    if len(ts) == 0 or t < ts[0] or t > ts[-1]:
        return None
    return np.array([np.interp(t, ts, track["p"][:, a]) for a in range(3)])


def _project(P, T_wc, K, dist):
    T_cw = inv_T(T_wc)
    pc = np.atleast_2d(P) @ T_cw[:3, :3].T + T_cw[:3, 3]
    uv = np.full((len(pc), 2), np.nan)
    ok = (pc[:, 2] > 0.1) & np.all(np.abs(pc[:, :2] / np.maximum(pc[:, 2:3], 1e-9)) < 2.5, axis=1)
    if ok.any():
        im, _ = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        uv[ok] = im.reshape(-1, 2)
    return uv


def _distort_project(pc, K, dist):
    """OpenCV projection (k1 k2 p1 p2 k3) of camera-frame points (..., 3); nan behind / far off axis."""
    k = np.r_[np.ravel(dist), np.zeros(5)][:5]
    z = pc[..., 2]
    ok = z > 0.1
    x = np.where(ok, pc[..., 0] / np.where(ok, z, 1.0), np.nan)
    y = np.where(ok, pc[..., 1] / np.where(ok, z, 1.0), np.nan)
    bad = (np.abs(x) > 2.5) | (np.abs(y) > 2.5)
    r2 = x * x + y * y
    rad = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[4] * r2 ** 3
    xd = x * rad + 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x)
    yd = y * rad + k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y
    uv = np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)
    uv[bad] = np.nan
    return uv


def _scaled(intr: Intrinsics, w):
    s = w / intr.width
    K = intr.K.copy()
    K[:2] *= s
    return K, s


def moving_blobs(prev_bgr, bgr, thr=60.0):
    """
    Drones in a hand-held video: frame difference after warping the previous frame onto
    this one with the background homography. The spinning propellers leave compact bright
    clusters, the parallax of the background only thin lines (removed by an opening).
    Centroids in the frame px of bgr.
    """
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, 1500, 0.01, 8)
    if p0 is None:
        return np.zeros((0, 2))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return np.zeros((0, 2))
    H, _ = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
    if H is None:
        return np.zeros((0, 2))
    h, w = g1.shape
    warped = cv2.warpPerspective(prev_bgr, H, (w, h))
    valid = cv2.warpPerspective(np.ones((h, w), np.uint8), H, (w, h)) > 0
    d = np.abs(bgr.astype(np.float32) - warped.astype(np.float32)).max(2)
    m = ((d > thr) & valid).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    keep = [k for k in range(1, n) if 120 <= st[k, 4] <= 4000 and max(st[k, 2], st[k, 3]) < 90]
    return cen[keep].reshape(-1, 2)


def _rot_fit(b1, b2):
    """R with b2 ~ R b1 (Kabsch on unit vectors)."""
    H = b1.T @ b2
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def relative_rotation(p1, p2, K, dist, rng, thr=2.0, iters=200):
    """
    Rotation between two frames from tracked features: x2 = R x1 in camera
    coordinates (rotation-only model, RANSAC on pairs, refit on the inliers).
    thr in px. Returns (R, inlier ratio).
    """
    if len(p1) < 8:
        return np.eye(3), 0.0
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    b1 = np.c_[n1, np.ones(len(n1))]
    b2 = np.c_[n2, np.ones(len(n2))]
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    ang = thr / K[0, 0]
    best, best_in = np.eye(3), None
    for _ in range(iters):
        i = rng.choice(len(b1), 2, replace=False)
        R = _rot_fit(b1[i], b2[i])
        inl = np.linalg.norm(b2 - b1 @ R.T, axis=1) < ang
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = R, inl
    if best_in.sum() >= 6:
        best = _rot_fit(b1[best_in], b2[best_in])
        best_in = np.linalg.norm(b2 - b1 @ best.T, axis=1) < ang
    return best, float(best_in.mean())


def translation_direction(p1, p2, R, K, dist, rng, thr=1.5, iters=200):
    """
    Direction of t in x2 = R x1 + t (camera 2 coordinates) for a known R, from the
    epipolar constraint t . ((R b1) x b2) = 0 (2-point RANSAC). Returns (t unit or
    None, parallax [px]): the parallax tells how well t is determined.
    """
    if len(p1) < 10:
        return None, 0.0
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    b1 = np.c_[n1, np.ones(len(n1))] @ R.T
    b2 = np.c_[n2, np.ones(len(n2))]
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    A = np.cross(b1, b2)
    par = np.linalg.norm(A, axis=1)
    parallax = float(np.median(par) * K[0, 0])
    if parallax < 0.3:
        return None, parallax
    tol = thr / K[0, 0]
    best, best_in = None, None
    for _ in range(iters):
        i, j = rng.choice(len(A), 2, replace=False)
        t = np.cross(A[i], A[j])
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            continue
        t /= nt
        inl = np.abs(A @ t) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = t, inl
    if best is None or best_in.sum() < 8:
        return None, parallax
    t = np.linalg.svd(A[best_in])[2][-1]
    # cheirality: depths l1 (R b1) + t = l2 b2 positive for most inliers
    pos = 0
    for a, c in zip(b1[best_in], b2[best_in]):
        M = np.column_stack([a, -c])
        l = np.linalg.lstsq(M, -t, rcond=None)[0]
        pos += int(np.all(l > 0)) - int(np.all(l < 0))
    return (t if pos >= 0 else -t), parallax


def solve_path(c0, c1, U, Wt, smooth=0.05):
    """
    Camera centres at the keyframes. Step j: v_j ~ sigma U[j] (direction U[j] from the
    epipolar geometry, weight Wt[j]; the step length is not observable from the video, so
    the prior is a common length sigma), the steps are smooth and sum to c1 - c0 (closure on
    the second anchor). Linear least squares in (v, sigma). Returns the J + 1 centres.
    (Tried: step length proportional to the de-rotated parallax; worse, the nearby nets
    dominate the parallax.)
    """
    J = len(U)
    n = 3 * J + 1
    rows, rhs = [], []
    for j in range(J):
        r = np.zeros((3, n))
        r[:, 3 * j:3 * j + 3] = np.sqrt(Wt[j]) * np.eye(3)
        r[:, -1] = -np.sqrt(Wt[j]) * U[j]
        rows.append(r)
        rhs.append(np.zeros(3))
    for j in range(J - 1):
        r = np.zeros((3, n))
        r[:, 3 * j:3 * j + 3] = -np.eye(3) * np.sqrt(smooth)
        r[:, 3 * j + 3:3 * j + 6] = np.eye(3) * np.sqrt(smooth)
        rows.append(r)
        rhs.append(np.zeros(3))
    r = np.zeros((3, n))
    r[:, :3 * J] = np.tile(np.eye(3), (1, J)) * 100.0
    rows.append(r)
    rhs.append((c1 - c0) * 100.0)
    x = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    v = x[:3 * J].reshape(J, 3)
    return np.vstack([c0, c0 + np.cumsum(v, 0)])


# ============================================================
# gaps of a tracked camera
# ============================================================

def find_gaps(cam: CameraTrack | None, t_a, t_b, t_lo=-np.inf, t_hi=np.inf, max_gap=0.1):
    """
    Intervals without camera pose that intersect the flight time [t_a, t_b]. A gap
    between two tracked stretches is returned whole (to its anchors, even outside
    [t_a, t_b]) but clipped to the time covered by the video [t_lo, t_hi]; a gap at
    the start / end of the camera log is cut at t_a / t_b.
    """
    if cam is None or len(cam.t) == 0:
        return [(t_a, t_b)]
    gaps, ts = [], cam.t
    if ts[0] > t_a:
        gaps.append((t_a, min(ts[0], t_b)))
    for i in np.flatnonzero(np.diff(ts) > max_gap):
        a, b = ts[i], ts[i + 1]
        if b > t_a and a < t_b:
            gaps.append((max(a, t_lo), min(b, t_hi)))
    if ts[-1] < t_b:
        gaps.append((max(ts[-1], t_a), t_b))
    return gaps


def recover_gaps(video, offset, intr: Intrinsics, T_bc, cam: CameraTrack, real, t_a, t_b,
                 knot_dt=0.5, refine=True, verbose=True):
    """
    Fill the gaps of cam in flight time [t_a, t_b] (see the module docstring).
    real: {id: {"t", "p"}} OptiTrack drone tracks. Returns (CameraTrack, report).
    """
    K, s = _scaled(intr, WORK_W)
    dist = intr.dist
    T_cb = inv_T(T_bc)
    rng = np.random.default_rng(0)
    info = video_info(video)
    dur = info["frames"] / (info["fps"] or 30.0)
    gaps = find_gaps(cam, t_a, t_b, -offset + 0.1, dur - offset - 0.1)
    times, Ts, report = [], [], []
    for g0, g1 in gaps:
        if g1 - g0 < 0.05:
            continue
        T0 = cam.T_wb(g0 - 1e-3) if g0 > cam.t[0] else None
        T1 = cam.T_wb(g1 + 1e-3) if g1 < cam.t[-1] else None
        T0 = None if T0 is None else T0 @ T_bc
        T1 = None if T1 is None else T1 @ T_bc
        if T0 is None and T1 is None:
            raise ValueError("gap without anchor (camera never tracked): use the static solver")
        # --- frames of the gap: relative rotations and LED detections
        ft, Rrel, inl, dets = [], [], [], []
        prev = None
        kf_stride = 6                                      # keyframes for the translation (0.1 s at 60 fps)
        kf_idx, kf_pairs, kf_gray = [], [], None
        hist = []                                          # last frames (WORK_W) for the drone detector
        for _, tv, fr in video_frames(video, start_s=max(0.0, g0 + offset - 0.05), end_s=g1 + offset + 0.05):
            small = cv2.resize(fr, (WORK_W, int(round(fr.shape[0] * WORK_W / fr.shape[1]))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                p0 = cv2.goodFeaturesToTrack(prev, 600, 0.01, 12)
                if p0 is not None:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
                    ok = st.ravel() == 1
                    R, q = relative_rotation(p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok], K, dist, rng)
                else:
                    R, q = np.eye(3), 0.0
                Rrel.append(R)
                inl.append(q)
            k = len(ft)
            if k % kf_stride == 0:
                if kf_gray is not None:
                    q0 = cv2.goodFeaturesToTrack(kf_gray, 800, 0.01, 10)
                    if q0 is not None:
                        q1, st, _ = cv2.calcOpticalFlowPyrLK(kf_gray, gray, q0, None, winSize=(31, 31), maxLevel=4)
                        q0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, kf_gray, q1, None, winSize=(31, 31), maxLevel=4)
                        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(q0b - q0, axis=2).ravel() < 1.0)
                        kf_pairs.append((q0.reshape(-1, 2)[ok], q1.reshape(-1, 2)[ok]))
                    else:
                        kf_pairs.append((np.zeros((0, 2)), np.zeros((0, 2))))
                kf_idx.append(k)
                kf_gray = gray
            ft.append(tv - offset)
            hist = (hist + [small])[-5:]
            dets.append(moving_blobs(hist[0], small) if (refine and len(hist) == 5) else np.zeros((0, 2)))
            prev = gray
        ft = np.array(ft)
        n = len(ft)
        if n < 2:
            continue
        # --- keyframe rotations from the essential matrix (not biased by the parallax of the
        #     translation, unlike the frame-to-frame rotation-only fit): the frame-to-frame
        #     rotations of every keyframe interval are corrected to match it
        n_ess = 0
        for j, (q0, q1) in enumerate(kf_pairs):
            a, b = kf_idx[j], kf_idx[j + 1] if j + 1 < len(kf_idx) else None
            if b is None or len(q0) < 15:
                continue
            R_chain = np.eye(3)
            for k in range(a, b):
                R_chain = Rrel[k] @ R_chain
            n0 = cv2.undistortPoints(q0.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            n1 = cv2.undistortPoints(q1.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            if np.median(np.linalg.norm(n1 - n0, axis=1)) * K[0, 0] < 2.0:
                continue
            E, msk = cv2.findEssentialMat(n0, n1, np.eye(3), cv2.RANSAC, 0.999, 1.0 / K[0, 0])
            if E is None or E.shape != (3, 3) or msk.sum() < 15:
                continue
            _, R_E, _, _ = cv2.recoverPose(E, n0, n1, np.eye(3), mask=msk)
            dR = Rotation.from_matrix(R_E @ R_chain.T).as_rotvec()
            if np.degrees(np.linalg.norm(dR)) > 3.0:     # degenerate / wrong decomposition
                continue
            step = Rotation.from_rotvec(dR / (b - a)).as_matrix()
            for k in range(a, b):
                Rrel[k] = step @ Rrel[k]
            n_ess += 1
        # --- rotation: integrate forward from T0 and backward from T1, blend
        Rf = [None] * n
        Rb = [None] * n
        if T0 is not None:
            Rf[0] = T0[:3, :3]
            for k in range(1, n):
                Rf[k] = Rf[k - 1] @ Rrel[k - 1].T          # R_wc_k = R_wc_{k-1} R_rel^T
        if T1 is not None:
            Rb[-1] = T1[:3, :3]
            for k in range(n - 2, -1, -1):
                Rb[k] = Rb[k + 1] @ Rrel[k]
        Rw = []
        for k in range(n):
            if Rf[k] is None:
                Rw.append(Rb[k])
            elif Rb[k] is None:
                Rw.append(Rf[k])
            else:
                w = (ft[k] - ft[0]) / max(ft[-1] - ft[0], 1e-9)
                Rw.append(Slerp([0, 1], Rotation.from_matrix([Rf[k], Rb[k]]))([w]).as_matrix()[0])
        # --- optical centre: with both anchors, along the translation directions measured in the
        #     video (epipolar, keyframes), scaled so that the path closes on the second anchor;
        #     with one anchor (scale unknown) held at it
        c0 = T0[:3, 3] if T0 is not None else T1[:3, 3]
        c1 = T1[:3, 3] if T1 is not None else T0[:3, 3]
        w = (ft - ft[0]) / max(ft[-1] - ft[0], 1e-9)
        C = (1 - w)[:, None] * c0 + w[:, None] * c1
        if T0 is not None and T1 is not None and len(kf_pairs) >= 2:
            kf = list(kf_idx)
            if kf[-1] != n - 1:                            # last frame: no measured step, smooth only
                kf.append(n - 1)
                kf_pairs.append((np.zeros((0, 2)), np.zeros((0, 2))))
            U, Wt = [], []
            for j, (q0, q1) in enumerate(kf_pairs):
                a, b = kf[j], kf[j + 1]
                R21 = Rw[b].T @ Rw[a]
                t, par = translation_direction(q0, q1, R21, K, dist, rng)
                if t is None:
                    U.append(np.zeros(3))
                    Wt.append(0.0)
                else:
                    U.append(-(Rw[b] @ t))                 # C_b - C_a = -R_wc_b t
                    Wt.append(float(np.clip(par / 3.0, 0.0, 1.0)))
            Ck = solve_path(c0, c1, np.array(U), np.array(Wt))
            C = np.column_stack([np.interp(np.arange(n), kf, Ck[:, a]) for a in range(3)])
        # --- refinement on the drones: smooth corrections (knots), zero at the anchors
        nk = max(2, int(np.ceil((ft[-1] - ft[0]) / knot_dt)) + 1)
        kt = np.linspace(ft[0], ft[-1], nk)
        free = np.ones(nk, bool)
        if T0 is not None:
            free[0] = False
        if T1 is not None:
            free[-1] = False
        nf = int(free.sum())
        P_dr = [np.array([_pos(real[i], t) if _pos(real[i], t) is not None else [np.nan] * 3 for i in real])
                for t in ft]
        # frames used by the fit: a drone in the air (spinning propellers: parked drones are not
        # detected) and something detected; at most 400, evenly spread
        flying = np.array([np.nanmax(np.r_[P_dr[k][:, 2], 0.0]) > 0.08 for k in range(n)])
        cand = np.flatnonzero(flying & np.array([len(d) > 0 for d in dets]))
        sel = cand[np.unique(np.linspace(0, len(cand) - 1, min(400, len(cand))).astype(int))] if len(cand) else cand

        def corr(x):
            dc = np.zeros((nk, 3))
            dr = np.zeros((nk, 3))
            dc[free] = x[:3 * nf].reshape(-1, 3)
            dr[free] = x[3 * nf:].reshape(-1, 3)
            DC = np.column_stack([np.interp(ft, kt, dc[:, a]) for a in range(3)])
            DR = np.column_stack([np.interp(ft, kt, dr[:, a]) for a in range(3)])
            return DC, DR

        def pose(k, DC, DR):
            R = Rotation.from_rotvec(DR[k]).as_matrix() @ Rw[k]
            return make_T(R, C[k] + DC[k])

        # vectorized: selected frames x drones, detections padded
        Rw_s = np.array([Rw[k] for k in sel])
        P_s = np.array([P_dr[k] for k in sel])                      # (S, D, 3)
        nb = max(1, max((len(dets[k]) for k in sel), default=1))
        B_s = np.full((len(sel), nb, 2), 1e6)
        for j, k in enumerate(sel):
            if len(dets[k]):
                B_s[j, :len(dets[k])] = dets[k]

        def resid(x, cap):
            DC, DR = corr(x)
            Rc = Rotation.from_rotvec(DR[sel]).as_matrix() @ Rw_s     # (S, 3, 3) R_wc
            Cc = C[sel] + DC[sel]
            pc = np.einsum("sji,sdj->sdi", Rc, P_s - Cc[:, None, :])  # camera coordinates
            uv = _distort_project(pc, K, dist)                       # (S, D, 2), nan if behind
            d = np.linalg.norm(uv[:, :, None, :] - B_s[:, None, :, :], axis=3).min(2)
            r = np.where(np.isfinite(d), np.minimum(d, cap), cap).ravel()
            # priors: corrections small (position 0.5 m, rotation 3 deg ~ 3 px each)
            reg_c = x[:3 * nf] / 0.5 * 3.0
            reg_r = x[3 * nf:] / np.radians(3.0) * 3.0
            return np.r_[r, reg_c, reg_r]

        x = np.zeros(6 * nf)
        if len(sel) < 10:
            refine_gap = False
        else:
            refine_gap = refine
        if not refine_gap:
            r_before = r_after = np.array([np.nan])
        else:
            r_before = resid(x, 12.0)[:-6 * nf]
        for cap in ((100.0, 50.0, 25.0, 12.0) if refine_gap else ()):
            x = least_squares(resid, x, args=(cap,), loss="soft_l1", f_scale=3.0, diff_step=1e-3, max_nfev=200).x
        if refine_gap:
            r_after = resid(x, 12.0)[:-6 * nf]
        DC, DR = corr(x)
        for k in range(n):
            times.append(ft[k])
            Ts.append(pose(k, DC, DR) @ T_cb)
        rep = {"gap": [float(g0), float(g1)], "frames": n, "anchors": [T0 is not None, T1 is not None],
               "keyframes_essential": n_ess,
               "flow_inliers_median": float(np.median(inl)) if inl else float("nan"),
               "drone_px_median_before": float(np.median(r_before) / s),
               "drone_px_median_after": float(np.median(r_after) / s),
               "drone_frames": int(len(sel)),
               "drone_inliers_after": float(np.mean(r_after < 5.0)) if refine_gap else float("nan"),
               "max_pos_correction_m": float(np.abs(DC).max()),
               "max_rot_correction_deg": float(np.degrees(np.linalg.norm(DR, axis=1).max()))}
        report.append(rep)
        if verbose:
            print(f"  gap {g0:6.2f}-{g1:6.2f} s ({n} frames, anchors {'before' if T0 is not None else ''}"
                  f"{' after' if T1 is not None else ''}): drones median {rep['drone_px_median_before']:.1f} -> "
                  f"{rep['drone_px_median_after']:.1f} px, correction {rep['max_pos_correction_m']:.2f} m / "
                  f"{rep['max_rot_correction_deg']:.1f} deg")
    # merge with the tracked poses
    t_all = np.r_[cam.t, times]
    p_all = np.r_[cam.pos, [T[:3, 3] for T in Ts]] if Ts else cam.pos
    q_all = np.r_[cam.quat, [Rotation.from_matrix(T[:3, :3]).as_quat() for T in Ts]] if Ts else cam.quat
    return CameraTrack(t_all, p_all, q_all), report


# ============================================================
# never tracked, static camera
# ============================================================

def _static_samples(video, v_a, v_b, step=0.25):
    """Moving blobs every step s of a static camera (background: median of the frames), WORK_W px."""
    frames = {}
    for _, tv, fr in video_frames(video, start_s=max(0.0, v_a)):
        if tv > v_b:
            break
        key = round(tv / step)
        if key not in frames:
            frames[key] = (tv, cv2.resize(fr, (WORK_W, int(round(fr.shape[0] * WORK_W / fr.shape[1]))),
                                          interpolation=cv2.INTER_AREA))
    arr = list(frames.values())
    bg = np.median(np.stack([f.astype(np.float32) for _, f in arr[::4]]), 0)
    tv_s, blobs = [], []
    for tv, f in arr:
        d = np.abs(f.astype(np.float32) - bg).max(2)
        m = cv2.morphologyEx((d > 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        tv_s.append(tv)
        blobs.append(cen[[k for k in range(1, n) if 4 <= st[k, 4] <= 600]].reshape(-1, 2))
    nb = max(1, max(len(b) for b in blobs))
    B = np.full((len(blobs), nb, 2), 1e6)
    for j, b in enumerate(blobs):
        B[j, :len(b)] = b
    return np.array(tv_s), B, bg.shape[1], bg.shape[0]


def _drone_positions(real, t):
    """(S, D, 3) OptiTrack positions at the times t (nan outside a track)."""
    out = np.full((len(t), len(real), 3), np.nan)
    for d, tr in enumerate(real.values()):
        if len(tr["t"]) == 0:
            continue
        inside = (t >= tr["t"][0]) & (t <= tr["t"][-1])
        for a in range(3):
            out[:, d, a] = np.where(inside, np.interp(t, tr["t"], tr["p"][:, a]), np.nan)
    return out


def _static_cost(R_wc, C, f, k1, P, B, W, H, cap):
    """Distance of every projected drone to the nearest blob, capped; (S*D,)."""
    pc = (P - C) @ R_wc                                    # camera coordinates (R_wc^T (P - C))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    uv = _distort_project(pc, K, np.array([k1, 0, 0, 0, 0]))
    d = np.linalg.norm(uv[:, :, None, :] - B[:, None, :, :], axis=3).min(2)
    d = np.where(np.isfinite(d), np.minimum(d, cap), cap)
    return d[np.all(np.isfinite(P), axis=2)]


def recover_static(video, real, t_a, t_b, offset0, intr: Intrinsics | None = None, verbose=True):
    """
    Static camera never tracked: T_wc, focal length, k1 and the offset (video t = flight t + offset)
    from the drones. offset0: rough offset (+- 1 s, e.g. take off in the video - take off in the log).
    intr: keep these intrinsics (only the pose and the offset are solved); None: solve f and k1.
    Returns (T_wc, Intrinsics at the video size, offset, report).
    """
    info = video_info(video)
    VW, VH = info["width"], info["height"]
    tv, B, W, H = _static_samples(video, t_a + offset0 - 1.0, t_b + offset0 + 1.0)
    s = W / VW
    if verbose:
        print(f"  static camera: {len(tv)} frames, {np.mean(np.sum(B[:, :, 0] < 1e5, 1)):.1f} moving blobs per frame")
    # --- initial guess: offsets around offset0, camera on rings around the drones looking at them
    sub = slice(None, None, 4)
    P_mid = _drone_positions(real, tv - offset0)
    centre = np.nanmedian(P_mid.reshape(-1, 3), 0)
    f_fixed = None if intr is None else intr.scaled_to(VW, VH).K[0, 0] * s
    k1_fixed = 0.0 if intr is None else float(intr.dist[0])
    f_grid = [f_fixed] if f_fixed else np.geomspace(0.35 * W, 1.6 * W, 10)
    best = None
    for off in offset0 + np.arange(-1.0, 1.01, 0.25):
        P = _drone_positions(real, tv[sub] - off)
        for rad in (1.5, 2.5, 3.5, 4.5):
            for az in np.radians(np.arange(0, 360, 10)):
                for hgt in (0.4, 0.9, 1.4, 1.9, 2.4):
                    eye = np.array([centre[0] + rad * np.cos(az), centre[1] + rad * np.sin(az), hgt])
                    for dz in (-0.6, 0.0, 0.6):
                        fwd = centre + [0, 0, dz] - eye
                        fwd /= np.linalg.norm(fwd)
                        right = np.cross(fwd, [0, 0, 1.0])
                        right /= np.linalg.norm(right)
                        R = np.column_stack([right, np.cross(fwd, right), fwd])
                        for f in f_grid:
                            c = _static_cost(R, eye, f, k1_fixed, P, B[sub], W, H, 60.0).mean()
                            if best is None or c < best[0]:
                                best = (c, R, eye, f, off)
    if verbose:
        print(f"  initial guess: {best[0]:.1f} px (capped 60), camera at {np.round(best[2], 2)}, "
              f"f {best[3] / s:.0f} px, offset {best[4]:.2f} s")

    def unpack(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        f, k1 = (x[6], x[7]) if f_fixed is None else (f_fixed, k1_fixed)
        return R, x[3:6], f, k1, x[-1]

    def resid(x, cap):
        R, C, f, k1, off = unpack(x)
        return _static_cost(R, C, f, k1, _drone_positions(real, tv - off), B, W, H, cap)

    x = np.r_[Rotation.from_matrix(best[1]).as_rotvec(), best[2]]
    x = np.r_[x, best[3], 0.0, best[4]] if f_fixed is None else np.r_[x, best[4]]
    n_res = len(resid(x, 40.0))
    for cap in (40.0, 20.0, 10.0):
        fun = lambda z, c=cap: np.resize(resid(z, c), n_res)   # noqa: E731 (fixed length)
        x = least_squares(fun, x, loss="soft_l1", f_scale=3.0, diff_step=1e-4).x
    r = resid(x, 10.0)
    R, C, f, k1, off = unpack(x)
    T_wc = make_T(R, C)
    K = np.array([[f / s, 0, VW / 2], [0, f / s, VH / 2], [0, 0, 1.0]])
    intr_v = Intrinsics(K, np.array([k1, 0, 0, 0, 0], float), VW, VH, float(np.median(r) / s))
    rep = {"mode": "static", "median_px": float(np.median(r) / s), "inliers_5px": float(np.mean(r < 5.0 * s)),
           "offset": float(off), "camera_position": C.tolist(), "f_px": float(f / s), "k1": float(k1),
           "hfov_deg": float(np.degrees(2 * np.arctan(VW / 2 / (f / s)))), "frames": int(len(tv))}
    if verbose:
        print(f"  static camera: median {rep['median_px']:.1f} px, {100 * rep['inliers_5px']:.0f} % within 5 px, "
              f"camera at {np.round(C, 2)}, f {rep['f_px']:.0f} px (hfov {rep['hfov_deg']:.0f} deg), "
              f"k1 {k1:.3f}, offset {off:.3f} s")
    return T_wc, intr_v, float(off), rep


def static_track(T_wc, T_bc, t_a, t_b):
    """CameraTrack of a static camera (rigid body pose T_wc T_bc^-1) over [t_a, t_b], 20 Hz."""
    T_wb = T_wc @ inv_T(T_bc)
    t = np.arange(t_a - 1.0, t_b + 1.0, 0.05)
    q = Rotation.from_matrix(T_wb[:3, :3]).as_quat()
    return CameraTrack(t, np.tile(T_wb[:3, 3], (len(t), 1)), np.tile(q, (len(t), 1)))


def save_track(cam: CameraTrack, path, recovered_mask=None):
    df = pd.DataFrame({"t": cam.t, "x": cam.pos[:, 0], "y": cam.pos[:, 1], "z": cam.pos[:, 2],
                       "qx": cam.quat[:, 0], "qy": cam.quat[:, 1], "qz": cam.quat[:, 2], "qw": cam.quat[:, 3]})
    if recovered_mask is not None:
        df["recovered"] = recovered_mask.astype(int)
    df.to_csv(path, index=False)


def save_report(report, path):
    Path(path).write_text(json.dumps(report, indent=2))
