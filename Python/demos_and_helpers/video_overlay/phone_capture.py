"""
Step 1 of the phone-camera correction: record the OptiTrack poses of the
phone rigid body (default ID 0) and of a static object (default ID 1, e.g. a
Crazyflie sitting on the floor) while the phone films the object.

The phone is held STILL (tripod / clamp / resting on a box) at a few
"stations". At each station press Enter: the poses of the phone and of the
object are averaged over --avg seconds and saved, with a stillness check.
Use 3-5 stations at clearly different positions / heights / view
directions around the object (one station cannot determine the full
correction: see phone_align.py). Keep the object in view at every station.

    python phone_capture.py --out sessions/phone_calib                 # phone 0, object 1
    python phone_capture.py --out sessions/phone_calib --phone-id 0 --object-ids 1 2

Turn the phone TORCH on: the object's retro-reflective markers then glow in
the video and can be clicked (or snapped automatically) in phone_align.py.
Start the phone recording BEFORE the first station and stop it after the
last one (one video for all stations). At each station keep the phone still
for the whole averaging window (the terminal prints "hold" -> "saved").

Outputs (in --out):
    capture.json     stations: mean pose of every body (world Z-up frame of the
                     flight code, natnet_funcs.py), mean position of every marker
                     of the object(s) (labeled markers, NatNet 3+), stillness
                     stats, mocap times
    capture_raw.csv  every NatNet frame of the requested bodies (raw Motive Y-up
                     AND converted Z-up), for checks / a moving-phone variant
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import motive_to_zup  # noqa: E402

# Same network settings as the flight code (Crazy_Flies/*/Structs.py)
MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

RAW_COLUMNS = ["time_s", "frame", "id", "valid",
               "mx", "my", "mz", "mqx", "mqy", "mqz", "mqw",      # Motive, Y-up
               "x", "y", "z", "qx", "qy", "qz", "qw",              # world Z-up
               "marker_error"]


class Capture:
    def __init__(self, out: Path, ids: list[int]):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.ids = ids
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.latest: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
        self.window: dict[int, list] | None = None      # collecting a station
        self.markers: dict[str, list] = {}               # object markers during the station window
        self.n_frames = 0
        self.seen: set[int] = set()
        self.f_raw = (out / "capture_raw.csv").open("w", newline="")
        self.w_raw = csv.writer(self.f_raw)
        self.w_raw.writerow(RAW_COLUMNS)
        self.last_flush = self.t0

    def now(self):
        return time.monotonic() - self.t0

    def on_frame(self, frame):
        t = self.now()
        frame_no = getattr(getattr(frame, "prefix", None), "frame_number", -1)
        with self.lock:
            self.n_frames += 1
            for rb in frame.rigid_bodies:
                self.seen.add(rb.id_num)
                if rb.id_num not in self.ids:
                    continue
                valid = bool(rb.tracking_valid) if rb.tracking_valid is not None else True
                pos, quat = motive_to_zup(rb.pos, rb.rot)
                self.w_raw.writerow([f"{t:.6f}", frame_no, rb.id_num, int(valid),
                                     *(f"{v:.6f}" for v in rb.pos), *(f"{v:.7f}" for v in rb.rot),
                                     *(f"{v:.6f}" for v in pos), *(f"{v:.7f}" for v in quat),
                                     rb.marker_error if rb.marker_error is not None else ""])
                if not valid:
                    continue
                p, q = np.array(pos, float), np.array(quat, float)
                self.latest[rb.id_num] = (t, p, q)
                if self.window is not None:
                    self.window[rb.id_num].append((t, p, q))
            # markers of the object rigid bodies (NatNet 3+: model id in the upper 16 bits)
            if self.window is not None:
                for m in getattr(frame, "labeled_markers", ()) or ():
                    model, mid = m.id_num >> 16, m.id_num & 0xFFFF
                    if model in self.ids and model != self.ids[0]:
                        pos, _ = motive_to_zup(m.pos, (0.0, 0.0, 0.0, 1.0))
                        self.markers.setdefault(f"{model}:{mid}", []).append(pos)
            if t - (self.last_flush - self.t0) > 1.0:
                self.f_raw.flush()
                self.last_flush = time.monotonic()

    def status(self):
        with self.lock:
            now = self.now()
            parts = []
            for i in self.ids:
                if i in self.latest and now - self.latest[i][0] < 0.2:
                    parts.append(f"id {i}: {np.round(self.latest[i][1], 3)}")
                else:
                    parts.append(f"id {i}: NOT TRACKED")
            return f"{now:7.1f} s  frames {self.n_frames}  " + "  ".join(parts) + f"  (seen {sorted(self.seen)})"

    def collect(self, duration):
        with self.lock:
            self.window = {i: [] for i in self.ids}
            self.markers = {}
        t_start = self.now()
        time.sleep(duration)
        with self.lock:
            w, self.window = self.window, None
            mk, self.markers = self.markers, {}
        return t_start, self.now(), w, mk

    def close(self):
        with self.lock:
            self.f_raw.close()


def summarize(samples):
    """Mean pose of one body over a station window + stillness statistics."""
    t = np.array([s[0] for s in samples])
    P = np.array([s[1] for s in samples])
    Q = np.array([s[2] for s in samples])
    rots = Rotation.from_quat(Q)
    mean_rot = rots.mean()
    p = P.mean(0)
    dev_mm = 1e3 * np.linalg.norm(P - p, axis=1)
    dev_deg = np.degrees((mean_rot.inv() * rots).magnitude())
    return {"n": int(len(t)), "t_mean": float(t.mean()),
            "pos": p.tolist(), "quat": mean_rot.as_quat().tolist(),
            "pos_std_mm": float(np.sqrt(np.mean(dev_mm ** 2))), "pos_max_dev_mm": float(dev_mm.max()),
            "rot_std_deg": float(np.sqrt(np.mean(dev_deg ** 2))), "rot_max_dev_deg": float(dev_deg.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--phone-id", type=int, default=0, help="Motive rigid-body ID of the phone")
    ap.add_argument("--object-ids", type=int, nargs="+", default=[1], help="Motive IDs of the static object(s)")
    ap.add_argument("--avg", type=float, default=2.0, help="averaging window per station [s]")
    ap.add_argument("--max-move-mm", type=float, default=3.0, help="stillness threshold (max deviation) [mm]")
    ap.add_argument("--max-move-deg", type=float, default=0.5, help="stillness threshold (max deviation) [deg]")
    ap.add_argument("--motive-ip", default=MOTIVE_IP)
    ap.add_argument("--local-ip", default=LOCAL_IP)
    ap.add_argument("--multicast", action="store_true", help="NatNet multicast instead of unicast")
    args = ap.parse_args()

    from natnet import NatNetClient

    out = Path(args.out)
    ids = [args.phone_id, *[i for i in args.object_ids if i != args.phone_id]]
    cap = Capture(out, ids)
    result = {"started": datetime.now().isoformat(timespec="seconds"),
              "phone_id": args.phone_id, "object_ids": args.object_ids,
              "frame": "world Z-up of the flight code: (x, y, z) = (Zm, Xm, Ym), quat (x, y, z, w)",
              "stations": []}
    path = out / "capture.json"

    client = NatNetClient(server_ip_address=args.motive_ip, local_ip_address=args.local_ip,
                          use_multicast=args.multicast)
    client.on_data_frame_received_event.handlers.append(cap.on_frame)

    stop = threading.Event()

    def ticker():
        while not stop.wait(2.0):
            print(cap.status())

    try:
        with client:
            client.request_modeldef()
            client.run_async()
            threading.Thread(target=ticker, daemon=True).start()
            print(f"Phone ID {args.phone_id}, object ID(s) {args.object_ids}. "
                  "Start the phone video now.\n"
                  "Enter: record a station (hold the phone still)   r+Enter: redo the last one   q+Enter: finish")
            while True:
                cmd = input().strip().lower()
                if cmd == "q":
                    break
                if cmd == "r":
                    if result["stations"]:
                        dropped = result["stations"].pop()
                        print(f"Station {dropped['index']} removed.")
                        path.write_text(json.dumps(result, indent=2))
                    continue
                k = len(result["stations"])
                print(f"Station {k}: HOLD STILL ({args.avg:.1f} s)...")
                t0, t1, w, mk = cap.collect(args.avg)
                st = {"index": k, "t_start": t0, "t_end": t1, "bodies": {}, "markers": {}}
                for name, P in sorted(mk.items()):
                    P = np.asarray(P, float)
                    if len(P) >= 5:
                        st["markers"][name] = {"pos": P.mean(0).tolist(), "n": int(len(P)),
                                               "std_mm": float(1e3 * np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, 1))))}
                ok = True
                for i in ids:
                    if len(w[i]) < 5:
                        print(f"  id {i}: not tracked during the window -> station rejected")
                        ok = False
                        continue
                    s = summarize(w[i])
                    st["bodies"][str(i)] = s
                    still = s["pos_max_dev_mm"] <= args.max_move_mm and s["rot_max_dev_deg"] <= args.max_move_deg
                    print(f"  id {i}: {s['n']} frames, pos {np.round(s['pos'], 4)} m, "
                          f"moved max {s['pos_max_dev_mm']:.1f} mm / {s['rot_max_dev_deg']:.2f} deg"
                          + ("" if still else "  <-- NOT STILL"))
                    ok &= still
                if not ok:
                    print("  Station rejected: press Enter to try again.")
                    continue
                if result["stations"]:
                    prev = result["stations"][-1]["bodies"][str(args.phone_id)]
                    d = np.linalg.norm(np.subtract(st["bodies"][str(args.phone_id)]["pos"], prev["pos"]))
                    if d < 0.15:
                        print(f"  note: only {100 * d:.0f} cm from the previous station; "
                              "spread the stations more for a well-determined correction")
                print(f"  object markers: {len(st['markers'])} ({', '.join(st['markers']) or 'none streamed'})")
                result["stations"].append(st)
                path.write_text(json.dumps(result, indent=2))
                print(f"Station {k} saved (mocap t = {t0:.1f}-{t1:.1f} s). Move to the next one, then Enter.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        cap.close()
        path.write_text(json.dumps(result, indent=2))
        n = len(result["stations"])
        print(f"{n} station(s) saved to {path}")
        if n:
            ts = [s["t_start"] for s in result["stations"]]
            print("Station start times relative to station 0 [s]: " + ", ".join(f"{t - ts[0]:.1f}" for t in ts))


if __name__ == "__main__":
    main()
