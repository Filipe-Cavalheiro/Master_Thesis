"""
End-to-end test of the pipeline on the synthetic sessions (make_synthetic.py):
runs every command-line tool exactly as on the arena day and compares what
they recover with the ground truth.

  1. calibrate_intrinsics.py on the checkerboard images     -> K, distortion vs truth
  2. sync.py (flash detection + matching)                    -> offset / scale vs truth
     sync.py --refine on the static session (motion based)   -> offset vs truth
  3. calibrate_extrinsics.py: simulated clicks (true projection + 0.7 px noise,
     standing in for the manual clicking) on 3 frames, solve, then auto
     detection of the markers over many frames, solve      -> T_bc vs truth
  4. render_overlay.py (both sessions)                       -> overlay.mp4
  5. stroboscopic.py (static session)                        -> strobe.png, error plot, metrics
  6. overlay alignment check: projected mocap drones vs the drones in the
     video (pixel error of the whole chain)

    python run_synthetic_test.py            # (re)uses test_output/synthetic, generates it if missing
    python run_synthetic_test.py --regen    # regenerate the synthetic sessions first

Writes test_output/report.json and copies the main outputs to test_output/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_synthetic as syn  # noqa: E402
from overlay_lib import Intrinsics, Session, inv_T, load_extrinsics, project  # noqa: E402

PY = sys.executable
OUT = HERE / "test_output"
SYN = OUT / "synthetic"


def run(*args):
    print("\n$ " + " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([PY, *map(str, args)], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"FAILED: {args}")


def simulated_clicks(session: Session, times, rng):
    """Click step stand-in: true projection of the surveyed points + 0.7 px noise, at the true video times."""
    rows = []
    for tv in times:
        t_m = syn.VIDEO_SCALE * tv + syn.VIDEO_OFFSET
        T_wc = syn.camera_pose(session.root.name, t_m)
        P = np.array([p[1:] for p in syn.POINTS])
        uv, ok, _ = project(P, T_wc, Intrinsics(syn.K_TRUE, syn.DIST_TRUE, syn.W, syn.H), margin=0.0)
        for (name, *xyz), q, v in zip(syn.POINTS, uv, ok):
            if v:
                q = q + rng.normal(0, 0.7, 2)
                rows.append([tv, name, *xyz, q[0], q[1], "click"])
    df = pd.DataFrame(rows, columns=["video_time_s", "name", "x", "y", "z", "u", "v", "source"])
    df.to_csv(session.root / "correspondences.csv", index=False)
    return len(df)


def pose_error(T_est, T_true):
    dR = Rotation.from_matrix(T_est[:3, :3].T @ T_true[:3, :3]).magnitude()
    return float(np.degrees(dR)), float(1e3 * np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))


def chain_pixel_error(session: Session, n=40):
    """Projected mocap drones (estimated chain) vs their true image position (true camera, true time)."""
    s = session
    intr = Intrinsics.load(s.path("intrinsics"))
    T_bc = load_extrinsics(s.path("extrinsics"))
    scale, offset = s.sync()
    cam, real = s.camera_track(), s.real_tracks()
    true_intr = Intrinsics(syn.K_TRUE, syn.DIST_TRUE, syn.W, syn.H)
    flight = syn.Tracks.from_traj(syn.read_traj(syn.FLIGHT_LOG))
    errs = []
    for tv in np.linspace(8.0, 45.0, n):
        t_m_true = syn.VIDEO_SCALE * tv + syn.VIDEO_OFFSET
        truth = syn.drones_at(flight, t_m_true - syn.LOG_OFFSET)
        t_m = scale * tv + offset
        est = real.at(t_m)
        T_wc = cam.T_wb(t_m) @ T_bc
        T_wc_true = syn.camera_pose(s.root.name, t_m_true)
        for i in truth:
            a, oka, _ = project(est[i][None], T_wc, intr, margin=0.0)
            b, okb, _ = project(truth[i][None], T_wc_true, true_intr, margin=0.0)
            if oka[0] and okb[0]:
                errs.append(np.linalg.norm(a[0] - b[0]))
    e = np.array(errs)
    return {"mean_px": float(e.mean()), "p95_px": float(np.percentile(e, 95)), "max_px": float(e.max()), "n": len(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    if args.regen or not (SYN / "static" / "video.mp4").exists():
        run("make_synthetic.py", "--out", SYN)
    truth = json.loads((SYN / "ground_truth.json").read_text())
    report = {}
    rng = np.random.default_rng(0)

    # 1. intrinsics
    run("calibrate_intrinsics.py", "--images", str(SYN / "calib" / "*.png"), "--board", 9, 6, "--square", 0.04,
        "--out", SYN / "intrinsics.json", "--preview", OUT / "intrinsics_undistort_preview.png")
    intr = Intrinsics.load(SYN / "intrinsics.json")
    K_err = np.abs(intr.K - syn.K_TRUE)[[0, 1, 0, 1], [0, 1, 2, 2]]
    report["intrinsics"] = {"rms_px": intr.rms_px, "fx_fy_cx_cy_abs_err_px": K_err.round(3).tolist(),
                            "dist_est": intr.dist.round(4).tolist(), "dist_true": syn.DIST_TRUE.tolist()}
    print(f"Intrinsics: |K - K_true| (fx fy cx cy) = {K_err.round(2)} px")

    for kind in ("moving", "static"):
        d = SYN / kind
        shutil.copy(SYN / "intrinsics.json", d / "intrinsics.json")
        for f in ("extrinsics.json", "correspondences.csv", "sync.json"):
            (d / f).unlink(missing_ok=True)
        s = Session(d)
        r = {}

        # 2. sync (flashes)
        run("sync.py", d)
        sy = json.loads((d / "sync.json").read_text())
        r["sync"] = {"offset_err_ms": 1e3 * (sy["offset"] - truth["video_offset"]),
                     "scale_err_ppm": 1e6 * (sy["scale"] - truth["video_scale"]),
                     "n_matched": sy["n_matched"], "residuals_ms": sy["residuals_ms"]}
        # offset error at mid video (offset and scale are correlated)
        tmid = 25.0
        r["sync"]["time_err_mid_video_ms"] = 1e3 * ((sy["scale"] * tmid + sy["offset"])
                                                    - (truth["video_scale"] * tmid + truth["video_offset"]))
        print(f"[{kind}] sync: time error at 25 s of video {r['sync']['time_err_mid_video_ms']:.2f} ms")

        # 3. extrinsics: clicks -> solve -> auto -> solve
        click_times = [2.0, 12.0, 22.0] if kind == "moving" else [2.0]
        r["n_clicks"] = simulated_clicks(s, click_times, rng)
        run("calibrate_extrinsics.py", "solve", d)
        e_click = pose_error(load_extrinsics(d / "extrinsics.json"), syn.T_BC_TRUE)
        run("calibrate_extrinsics.py", "auto", d, "--points", SYN / "points.csv", "--t0", 1.0, "--t1", 40.0,
            "--step", 1.0, "--radius", 15)
        run("calibrate_extrinsics.py", "solve", d)
        ext = json.loads((d / "extrinsics.json").read_text())
        e_final = pose_error(np.array(ext["T_bc"]), syn.T_BC_TRUE)
        r["extrinsics"] = {"after_clicks_err_deg_mm": e_click, "final_err_deg_mm": e_final,
                           "rms_px": ext["rms_px"], "n_correspondences": ext["n_correspondences"]}
        print(f"[{kind}] T_bc error: clicks {e_click[0]:.3f} deg / {e_click[1]:.1f} mm, "
              f"final {e_final[0]:.3f} deg / {e_final[1]:.1f} mm")

        # 2b. motion-based sync refinement (static camera)
        if kind == "static":
            flash_offset = sy["offset"]
            run("sync.py", d, "--refine")
            sy2 = json.loads((d / "sync.json").read_text())
            r["sync_refine"] = {"offset_err_ms": 1e3 * (sy2["offset"] - truth["video_offset"]),
                                "flash_offset_err_ms": 1e3 * (flash_offset - truth["video_offset"])}
            # motion-only sync from a 0.3 s wrong guess (takeoff/flight motion alone)
            run("sync.py", d, "--offset-guess", truth["video_offset"] + 0.3, "--refine", "--refine-window", 0.6)
            sy3 = json.loads((d / "sync.json").read_text())
            r["sync_motion_only"] = {"offset_err_ms": 1e3 * (sy3["offset"] - truth["video_offset"])}
            print(f"[static] motion-based sync: refined {r['sync_refine']['offset_err_ms']:.1f} ms, "
                  f"motion only {r['sync_motion_only']['offset_err_ms']:.1f} ms")
            # keep the flash sync for the products
            (d / "sync.json").write_text(json.dumps(sy, indent=2))

        # 6. chain accuracy
        r["overlay_pixel_error"] = chain_pixel_error(s)
        print(f"[{kind}] overlay alignment (projected mocap vs true image position): "
              f"mean {r['overlay_pixel_error']['mean_px']:.2f} px, max {r['overlay_pixel_error']['max_px']:.2f} px")

        # 4. overlay video
        run("render_overlay.py", d, "--grid" if kind == "moving" else "--trail", *([] if kind == "moving" else [4.0]))
        shutil.copy(d / "overlay.mp4", OUT / f"overlay_{kind}.mp4")
        report[kind] = r

    # 5. stroboscopic composite (static)
    run("stroboscopic.py", SYN / "static", "--n", 8)
    m = json.loads((SYN / "static" / "strobe" / "metrics.json").read_text())
    report["strobe"] = {"camera_motion_cm_deg": m["camera_motion_cm_deg"],
                        "real_vs_sim": m["error_formation_phase"]["all"],
                        "strobe_pixel_error": m["error_formation_phase"]["strobe_pixel_error"]}
    for f in ("strobe.png", "strobe_clean.png", "strobe_overlay.png", "error.pdf", "error.png", "metrics.json"):
        shutil.copy(SYN / "static" / "strobe" / f, OUT / f"strobe_{f}" if not f.startswith("strobe") else OUT / f)

    # pass / fail
    checks = {
        "intrinsics fx/fy/cx/cy within 2 px": bool(np.all(K_err < 2.0)),
        "intrinsics RMS < 0.5 px": intr.rms_px < 0.5,
    }
    for kind in ("moving", "static"):
        r = report[kind]
        checks[f"{kind}: sync error < 1/4 frame (8.3 ms)"] = abs(r["sync"]["time_err_mid_video_ms"]) < 8.3
        # a single camera pose (static) leaves the translation along the optical axis weakly observable
        # (~1 cm at 3 m); it then barely moves the projections (checked by the overlay alignment below)
        tol_mm = 10 if kind == "moving" else 20
        checks[f"{kind}: T_bc error < 0.5 deg and {tol_mm} mm"] = (r["extrinsics"]["final_err_deg_mm"][0] < 0.5
                                                                  and r["extrinsics"]["final_err_deg_mm"][1] < tol_mm)
        checks[f"{kind}: overlay alignment mean < 2 px"] = r["overlay_pixel_error"]["mean_px"] < 2.0
    checks["static: motion-only sync within 1 frame (33 ms)"] = abs(report["static"]["sync_motion_only"]["offset_err_ms"]) < 33
    report["checks"] = checks
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\n================ SYNTHETIC END-TO-END TEST ================")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Report: {OUT / 'report.json'}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
