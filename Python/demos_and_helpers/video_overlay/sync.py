"""
Time synchronization video <-> mocap:  t_mocap = scale * t_video + offset.

1. Flash detection (automatic): finds the frames where a region of the image
   suddenly brightens (the screen flashes of record_mocap.py --flash, an LED
   panel, a phone flash), with sub-frame onset estimation from the partial
   brightness of the first lit frame (exposure ~ frame period), and matches
   the flash pattern to the "flash" events of mocap_events.csv (irregular
   intervals -> unique match). With >= 2 flashes spread over > 5 s the clock
   rate (scale) is fitted too.
2. --refine (optional, static camera, needs intrinsics + extrinsics): motion
   based refinement / check, e.g. with the drone takeoffs: the offset that
   puts the projected mocap drones on the moving pixels of the video
   (background-difference mask), searched around the flash offset (or
   around --offset-guess, then this alone is the sync method).

    python sync.py SESSION                           # flash detection + matching -> SESSION/sync.json
    python sync.py SESSION --search-end 60           # only look for flashes in the first 60 s of video
    python sync.py SESSION --refine                  # then motion-based refinement (static camera)
    python sync.py SESSION --manual-video-time 3.40 --event-kind key   # flash seen at 3.40 s, matched to a key event
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import Intrinsics, Session, load_extrinsics, project, read_events, video_frames, video_info  # noqa: E402

WORK_WIDTH = 320          # px, frames downscaled for the flash detection


# ============================================================
# FLASH DETECTION
# ============================================================

def detect_flashes(video, search_start=0.0, search_end=None, min_gap=0.5, diff_thr=40, verbose=True):
    """
    Flash onset times [s, video time] with their peak lit fraction.

    s_k = fraction of pixels brighter by > diff_thr than the median of the 5
    previous frames. A flash = run of frames where s_k is far above its robust
    baseline. Sub-frame onset: in the region lit at the peak, brightness b_k
    is normalized to f_k in [0, 1] between the baseline before the flash and
    the peak; with an exposure of one frame period dt, the first lit frame k0
    was exposed for a fraction f_k0 after the onset:  t_on = t_k0 + (1 - f_k0) dt.
    """
    times, grays, s = [], [], []
    ref = deque(maxlen=5)
    for _, t, frame in video_frames(video, search_start, search_end):
        h, w = frame.shape[:2]
        g = cv2.cvtColor(cv2.resize(frame, (WORK_WIDTH, int(round(h * WORK_WIDTH / w)))), cv2.COLOR_BGR2GRAY)
        if len(ref) >= 3:
            base = np.median(np.stack(ref), axis=0)
            s.append(float(np.mean(g.astype(np.int16) - base > diff_thr)))
        else:
            s.append(0.0)
        ref.append(g)
        times.append(t)
        grays.append(g)
    times, s = np.array(times), np.array(s)
    if len(times) < 10:
        return []
    dt = float(np.median(np.diff(times)))
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-6
    thr = max(0.002, med + 10 * mad)
    on = s > thr

    flashes = []
    k = 0
    while k < len(s):
        if not on[k]:
            k += 1
            continue
        k1 = k
        while k1 + 1 < len(s) and on[k1 + 1]:
            k1 += 1
        kp = k + int(np.argmax(s[k:k1 + 1]))
        b_before = np.stack(grays[max(0, k - 6):max(1, k - 1)]).astype(float)
        region = (grays[kp].astype(float) - np.median(b_before, axis=0)) > diff_thr
        if region.sum() >= 4:
            b = np.array([grays[j][region].mean() for j in range(max(0, k - 6), min(len(s), k1 + 2))])
            j0 = max(0, k - 6)
            b0 = np.median(b[:max(1, k - 1 - j0)])
            b1 = b[kp - j0]
            f = (b - b0) / max(b1 - b0, 1e-6)
            # first lit frame: the frame before the detected run may already be partially lit
            first = k
            if k - 1 >= j0 and f[k - 1 - j0] > 0.08:
                first = k - 1
            f0 = float(np.clip(f[first - j0], 0.0, 1.0))
            t_on = times[first] + (1.0 - f0) * dt
            if not flashes or t_on - flashes[-1]["t"] > min_gap:
                flashes.append({"t": float(t_on), "frame_time": float(times[first]), "peak_fraction": float(s[kp]),
                                "lit_fraction_first_frame": f0})
        k = k1 + 1
    if verbose:
        print(f"Flash search {times[0]:.2f}-{times[-1]:.2f} s ({len(times)} frames, dt {dt * 1e3:.1f} ms, "
              f"threshold {thr:.4f}): {len(flashes)} flashes")
        for fl in flashes:
            print(f"  onset {fl['t']:.4f} s (lit area {100 * fl['peak_fraction']:.1f} %, "
                  f"first frame lit {100 * fl['lit_fraction_first_frame']:.0f} %)")
    return flashes


def match_events(video_t, mocap_t, tol=0.06):
    """
    Offset (and scale) mapping the video flash times onto the mocap event
    times. Every pair (video flash i, event j) proposes offset m_j - v_i; the
    one with most matches within tol (then smallest residual) wins, then
    least squares on the matched pairs.
    """
    video_t, mocap_t = np.asarray(video_t, float), np.asarray(mocap_t, float)
    best = None
    for v in video_t:
        for m in mocap_t:
            b = m - v
            d = np.abs(mocap_t[None, :] - (video_t[:, None] + b))
            j = np.argmin(d, axis=1)
            ok = d[np.arange(len(video_t)), j] < tol
            score = (ok.sum(), -d[np.arange(len(video_t)), j][ok].sum())
            if best is None or score > best[0]:
                best = (score, ok, j)
    _, ok, j = best
    v, m = video_t[ok], mocap_t[j[ok]]
    if len(v) >= 2 and v.max() - v.min() > 5.0:
        scale, offset = np.polyfit(v, m, 1)
    else:
        scale, offset = 1.0, float(np.mean(m - v))
    res = m - (scale * v + offset)
    return {"scale": float(scale), "offset": float(offset), "n_matched": int(len(v)),
            "n_video_flashes": int(len(video_t)), "n_mocap_events": int(len(mocap_t)),
            "residuals_ms": (1e3 * res).round(2).tolist(),
            "pairs": [[float(a), float(b)] for a, b in zip(v, m)]}


# ============================================================
# MOTION-BASED REFINEMENT
# ============================================================

def refine_offset(session: Session, scale, offset0, window=0.5, step=None, n_frames=150, scale_px=0.5,
                  t_range=None, verbose=True):
    """
    Static camera: background = median of frames over the clip; foreground =
    |frame - background| (blurred). Score(offset) = mean foreground at the
    projected mocap drone positions over n_frames frames with moving drones.
    Returns the best offset in [offset0 - window, offset0 + window] and the score curve.
    """
    video = session.path("video")
    info = video_info(video)
    intr = Intrinsics.load(session.path("intrinsics")).scaled_to(info["width"], info["height"])
    T_bc = load_extrinsics(session.path("extrinsics"))
    cam = session.camera_track()
    real = session.real_tracks()
    fps = info["fps"] or 30.0
    step = step or 1.0 / fps / 4

    # frames where the drones move (speed > 0.05 m/s), inside the camera log
    t0m, t1m = real.span()
    t0m, t1m = max(t0m, cam.t[0]), min(t1m, cam.t[-1])
    tv = (np.arange(t0m, t1m, 0.05) - offset0) / scale
    if t_range is not None:
        tv = tv[(tv >= t_range[0]) & (tv <= t_range[1])]
    moving = []
    for t in tv:
        tm = scale * t + offset0
        p0, p1 = real.at(tm - 0.05), real.at(tm + 0.05)
        sp = max((np.linalg.norm(p1[i] - p0[i]) / 0.1 for i in p0 if np.all(np.isfinite(p0[i]) & np.isfinite(p1[i]))),
                 default=0.0)
        if sp > 0.05:
            moving.append(t)
    if len(moving) < 10:
        raise RuntimeError("Not enough frames with moving drones for the motion-based sync")
    sel = np.array(moving)[np.linspace(0, len(moving) - 1, min(n_frames, len(moving))).astype(int)]

    dp, dr = cam.motion(scale * sel[0] + offset0, scale * sel[-1] + offset0)
    if dp > 0.03 or dr > 1.0:
        print(f"WARNING: camera moved {dp * 100:.1f} cm / {dr:.1f} deg: --refine assumes a static camera")

    # frames (decoded once, in order)
    want = set(np.round(sel * fps).astype(int))
    frames = {}
    bg_samples = []
    every = max(1, info["frames"] // 60)
    for k, t, fr in video_frames(video):
        small = cv2.resize(fr, None, fx=scale_px, fy=scale_px, interpolation=cv2.INTER_AREA)
        if k % every == 0:
            bg_samples.append(small)
        if k in want:
            frames[k] = (t, small)
    bg = np.median(np.stack(bg_samples), axis=0).astype(np.int16)
    fgs = []
    for k in sorted(frames):
        t, small = frames[k]
        d = np.abs(small.astype(np.int16) - bg).max(axis=2).astype(np.float32)
        fg = cv2.GaussianBlur((d > 25).astype(np.float32), (0, 0), 4)
        fgs.append((t, fg))

    offsets = np.arange(offset0 - window, offset0 + window + 1e-9, step)
    scores = np.zeros(len(offsets))
    for n, b in enumerate(offsets):
        acc, cnt = 0.0, 0
        for t, fg in fgs:
            tm = scale * t + b
            T_wb = cam.T_wb(tm)
            if T_wb is None:
                continue
            pos = real.at(tm)
            P = np.array([p for p in pos.values() if np.all(np.isfinite(p))])
            if len(P) == 0:
                continue
            uv, valid, _ = project(P, T_wb @ T_bc, intr, margin=0.0)
            for u, v in uv[valid] * scale_px:
                ui, vi = int(round(u)), int(round(v))
                if 0 <= vi < fg.shape[0] and 0 <= ui < fg.shape[1]:
                    acc += fg[vi, ui]
                cnt += 1
        scores[n] = acc / max(cnt, 1)
    kbest = int(np.argmax(scores))
    best = offsets[kbest]
    if 0 < kbest < len(offsets) - 1:   # parabolic peak
        a, b_, c = scores[kbest - 1:kbest + 2]
        den = a - 2 * b_ + c
        if den < 0:
            best += 0.5 * step * (a - c) / den
    if verbose:
        print(f"Motion-based refinement: offset {best:.4f} s (from {offset0:.4f} s), score {scores[kbest]:.3f}, "
              f"{len(fgs)} frames")
    return float(best), offsets, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--search-start", type=float, default=0.0, help="video time [s] where the flash search starts")
    ap.add_argument("--search-end", type=float, default=None, help="video time [s] where it ends (default: whole video)")
    ap.add_argument("--event-kind", default="flash", help="mocap event kind matched to the flashes (flash / key)")
    ap.add_argument("--manual-video-time", type=float, nargs="*",
                    help="video times of the sync events, if the automatic detection is not usable")
    ap.add_argument("--offset-guess", type=float, help="no flash: initial offset for --refine")
    ap.add_argument("--refine", action="store_true", help="motion-based refinement (static camera)")
    ap.add_argument("--refine-window", type=float, default=0.5)
    args = ap.parse_args()

    s = Session(args.session)
    out = {}
    if args.offset_guess is not None:
        out = {"scale": 1.0, "offset": args.offset_guess, "method": "guess"}
    else:
        if args.manual_video_time:
            vt = list(args.manual_video_time)
            out["method"] = "manual video times"
        else:
            fl = detect_flashes(s.path("video"), args.search_start, args.search_end)
            vt = [f["t"] for f in fl]
            out["flashes"] = fl
            out["method"] = "flash"
        ev = read_events(s.path("events"))
        mt = ev[ev["kind"] == args.event_kind]["time_s"].to_numpy(float)
        if len(vt) == 0 or len(mt) == 0:
            sys.exit(f"Sync failed: {len(vt)} video flashes, {len(mt)} mocap '{args.event_kind}' events")
        out.update(match_events(vt, mt))
        print(f"Matched {out['n_matched']}/{out['n_video_flashes']} flashes to {out['n_mocap_events']} events: "
              f"t_mocap = {out['scale']:.7f} t_video + {out['offset']:.4f} s, residuals {out['residuals_ms']} ms")

    if args.refine:
        s.cfg.setdefault("sync", "sync.json")
        best, offs, scores = refine_offset(s, out["scale"], out["offset"], window=args.refine_window)
        out["flash_offset"] = out["offset"]
        out["offset"] = best
        out["method"] += " + motion refinement"
        out["refine_curve"] = {"offset": offs.round(5).tolist(), "score": scores.round(5).tolist()}

    path = s.path("sync")
    path.write_text(json.dumps(out, indent=2))
    print(f"Sync saved to {path}")


if __name__ == "__main__":
    main()
