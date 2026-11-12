"""
Records the OptiTrack poses of the drones AND of the camera rigid body
during a filmed flight, plus the time-sync events, into a session directory.

Runs next to the flight script (its own NatNet client; same machine or
another one on the Motive network) and writes, in one time base
(time.monotonic() since the start of the recording, like TrajectoryLogger):

    <out>/mocap_trajectory.csv   drones, TrajectoryLogger columns (only time_s, id, opti_* filled)
    <out>/mocap_camera.csv       camera rigid body: time_s, frame, motive_time_s, id, valid, x y z, qx qy qz qw
    <out>/mocap_events.csv       sync events: screen flashes (--flash) and key presses (Enter)
    <out>/session.json           file names of the session (edit it to add video / flight log / sim)

Every NatNet frame is logged (no resampling): the camera pose is needed at
the video frame times. Positions/quaternions are converted to the Z-up frame
of the flight code (natnet_funcs.py).

    python record_mocap.py --out sessions/2026_11_20_flight1 --camera-id 10 --drone-ids 1 2 3 4
    python record_mocap.py --out ... --camera-id 10 --flash 4      # screen flashes for the sync (needs a display)

Sync events:
  --flash N  : a full-screen window on this computer turns white N times at
               irregular intervals (after pressing Space in it). Film the
               screen. Each flash time is logged at the moment the white frame
               is drawn; sync.py finds them in the video automatically.
  Enter      : (terminal, without --flash) logs a "key" event, e.g. when you
               clap / fire a phone flash (~0.1 s accurate; use as a fallback).
Stop with Ctrl+C (or Esc in the flash window).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_lib import CAMERA_COLUMNS, EVENT_COLUMNS, TRAJ_COLUMNS, motive_to_zup  # noqa: E402

# Same network settings as the flight code (Crazy_Flies/*/Structs.py)
MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"


class MocapRecorder:
    """NatNet frame callback -> CSV rows (thread-safe, flushed about once per second)."""

    def __init__(self, out_dir: Path, camera_id: int, drone_ids: list[int] | None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.camera_id = camera_id
        self.drone_ids = set(drone_ids) if drone_ids else None   # None: every other rigid body
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.n_frames = 0
        self.n_cam_valid = 0
        self.last_flush = self.t0
        self.seen_ids: set[int] = set()

        self.f_traj = (self.out / "mocap_trajectory.csv").open("w", newline="")
        self.f_cam = (self.out / "mocap_camera.csv").open("w", newline="")
        self.f_evt = (self.out / "mocap_events.csv").open("w", newline="")
        self.w_traj = csv.DictWriter(self.f_traj, fieldnames=TRAJ_COLUMNS)
        self.w_cam = csv.writer(self.f_cam)
        self.w_evt = csv.writer(self.f_evt)
        self.w_traj.writeheader()
        self.w_cam.writerow(CAMERA_COLUMNS)
        self.w_evt.writerow(EVENT_COLUMNS)

        meta = {"started": datetime.now().isoformat(timespec="seconds"),
                "monotonic_t0": self.t0, "camera_id": camera_id,
                "drone_ids": sorted(self.drone_ids) if self.drone_ids else "all"}
        (self.out / "recorder_meta.json").write_text(json.dumps(meta, indent=2))
        session = self.out / "session.json"
        if not session.exists():
            session.write_text(json.dumps({
                "video": "video.mp4", "drones": "mocap_trajectory.csv", "camera": "mocap_camera.csv",
                "events": "mocap_events.csv", "camera_id": camera_id,
                "flight_log": None, "sim": None}, indent=2))

    def now(self):
        return time.monotonic() - self.t0

    def on_frame(self, frame):
        """NatNet data frame (natnet.DataFrame)."""
        t = self.now()
        try:
            motive_t = float(frame.suffix.timestamp)
        except AttributeError:
            motive_t = float("nan")
        frame_no = getattr(getattr(frame, "prefix", None), "frame_number", -1)
        with self.lock:
            self.n_frames += 1
            for rb in frame.rigid_bodies:
                self.seen_ids.add(rb.id_num)
                valid = bool(rb.tracking_valid) if rb.tracking_valid is not None else True
                pos, quat = motive_to_zup(rb.pos, rb.rot)
                if rb.id_num == self.camera_id:
                    self.n_cam_valid += valid
                    self.w_cam.writerow([f"{t:.6f}", frame_no, f"{motive_t:.6f}", rb.id_num, int(valid),
                                         *(f"{v:.6f}" for v in pos), *(f"{v:.7f}" for v in quat),
                                         rb.marker_error if rb.marker_error is not None else ""])
                elif valid and (self.drone_ids is None or rb.id_num in self.drone_ids):
                    self.w_traj.writerow({"time_s": f"{t:.6f}", "flight_state": "", "id": rb.id_num,
                                          "opti_x": pos[0], "opti_y": pos[1], "opti_z": pos[2]})
            if t - (self.last_flush - self.t0) > 1.0:
                for f in (self.f_traj, self.f_cam, self.f_evt):
                    f.flush()
                self.last_flush = time.monotonic()

    def event(self, kind, label="", t=None):
        t = self.now() if t is None else t
        with self.lock:
            self.w_evt.writerow([f"{t:.6f}", kind, label])
            self.f_evt.flush()
        print(f"[{t:8.3f} s] event {kind} {label}")

    def status(self):
        with self.lock:
            return (f"{self.now():7.1f} s  frames {self.n_frames}  camera valid {self.n_cam_valid}  "
                    f"rigid bodies seen {sorted(self.seen_ids)}")

    def close(self):
        with self.lock:
            for f in (self.f_traj, self.f_cam, self.f_evt):
                f.close()
        print(f"Session saved to {self.out}")


def run_flash_window(rec: MocapRecorder, n_flashes: int, duration=0.12, seed=1):
    """
    Full-screen black window; Space starts N white flashes at irregular
    intervals (1.5-3.5 s, so the pattern is unambiguous), Esc stops the recording.
    The event time is taken right after the white frame is pushed to the screen
    (display latency ~1 refresh, 16 ms at 60 Hz: refine with sync.py --refine if needed).
    """
    import random
    import tkinter as tk

    rng = random.Random(seed)
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    label = tk.Label(root, text="Space: flash sequence   Esc: stop", fg="gray30", bg="black", font=("Arial", 24))
    label.pack(expand=True)
    state = {"running": False, "done": 0}

    def flash():
        root.configure(bg="white")
        label.configure(bg="white", fg="white")
        root.update_idletasks()
        root.update()
        rec.event("flash", f"{state['done'] + 1}")
        root.after(int(duration * 1e3), unflash)

    def unflash():
        root.configure(bg="black")
        label.configure(bg="black", fg="gray30")
        state["done"] += 1
        if state["done"] < n_flashes:
            root.after(int(rng.uniform(1.5, 3.5) * 1e3), flash)
        else:
            state["running"] = False
            label.configure(text=f"{n_flashes} flashes done. Space: again   Esc: stop")

    def on_space(_):
        if not state["running"]:
            state["running"], state["done"] = True, 0
            root.after(1000, flash)

    def status():
        label.configure(text=label.cget("text").split("\n")[0] + "\n" + rec.status())
        root.after(1000, status)

    root.bind("<space>", on_space)
    root.bind("<Escape>", lambda _: root.destroy())
    status()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="session directory")
    ap.add_argument("--camera-id", type=int, required=True, help="Motive rigid-body ID of the camera")
    ap.add_argument("--drone-ids", type=int, nargs="*", help="Motive IDs of the drones (default: all other rigid bodies)")
    ap.add_argument("--motive-ip", default=MOTIVE_IP)
    ap.add_argument("--local-ip", default=LOCAL_IP)
    ap.add_argument("--multicast", action="store_true",
                    help="NatNet multicast (the flight code uses unicast; use multicast if Motive serves both)")
    ap.add_argument("--flash", type=int, default=0, help="number of screen flashes per sequence (0: no window)")
    args = ap.parse_args()

    from natnet import NatNetClient

    rec = MocapRecorder(Path(args.out), args.camera_id, args.drone_ids)
    client = NatNetClient(server_ip_address=args.motive_ip, local_ip_address=args.local_ip,
                          use_multicast=args.multicast)
    client.on_data_description_received_event.handlers.append(lambda d: print("NatNet data descriptions received."))
    client.on_data_frame_received_event.handlers.append(rec.on_frame)

    try:
        with client:
            client.request_modeldef()
            client.run_async()
            print(f"Recording to {args.out} (camera rigid body {args.camera_id}). Ctrl+C to stop.")
            if args.flash > 0:
                run_flash_window(rec, args.flash)
            else:
                stop = threading.Event()

                def keys():
                    while not stop.is_set():
                        try:
                            line = input()
                        except EOFError:
                            return
                        rec.event("key", line.strip())
                threading.Thread(target=keys, daemon=True).start()
                print("Press Enter to log a sync event (clap / phone flash).")
                while True:
                    time.sleep(2.0)
                    print(rec.status())
    except KeyboardInterrupt:
        pass
    finally:
        rec.close()


if __name__ == "__main__":
    main()
