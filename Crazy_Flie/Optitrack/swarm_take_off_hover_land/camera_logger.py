"""
Logs the pose of the filming camera (phone rigid body, CAMERA_RIGID_BODY_ID
in Structs.py) from the NatNet frames the flight script already receives:
no second NatNet client (two unicast clients on one computer would share
port 1511 and the flight script could lose its tracking).

Every frame of that rigid body is written to <log>_camera.csv next to the
trajectory log:

    mono_s, valid, x, y, z, qx, qy, qz, qw

mono_s is time.monotonic() (same clock as TrajectoryLogger); the
trajectory log's time origin is saved in <log>_camera.json
("flight_log_t0_monotonic"), so time_s = mono_s - flight_log_t0_monotonic.
Pose in the Z-up frame of the flight code: (x, y, z) = (Zm, Xm, Ym),
quaternion (x, y, z, w) = (qz, qx, qy, qw) of Motive.

Never interferes with the flight: every error is caught and counted.
"""

import csv
import json
import threading
import time
from pathlib import Path


class CameraLogger:
    def __init__(self, trajectory_path, rigid_body_id):
        p = Path(trajectory_path)
        self.path = p.with_name(p.stem + "_camera.csv")
        self.meta_path = p.with_name(p.stem + "_camera.json")
        self.rigid_body_id = rigid_body_id
        self.lock = threading.Lock()
        self.errors = 0
        self.n_valid = 0
        self.n = 0
        self.last_flush = time.monotonic()
        self.meta = {"rigid_body_id": rigid_body_id, "created_monotonic": time.monotonic(),
                     "flight_log_t0_monotonic": None,
                     "frame": "Z-up: (x, y, z) = (Zm, Xm, Ym), quat (x, y, z, w) = (qz, qx, qy, qw) of Motive"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["mono_s", "valid", "x", "y", "z", "qx", "qy", "qz", "qw"])
        self._write_meta()

    def _write_meta(self):
        self.meta_path.write_text(json.dumps(self.meta, indent=2))

    def on_frame(self, frame):
        """NatNet data frame handler (append to client.on_data_frame_received_event.handlers)."""
        try:
            t = time.monotonic()
            for rb in frame.rigid_bodies:
                if rb.id_num != self.rigid_body_id:
                    continue
                valid = 1 if (rb.tracking_valid is None or rb.tracking_valid) else 0
                mx, my, mz = rb.pos
                qx, qy, qz, qw = rb.rot
                with self.lock:
                    if self.file is None:
                        return
                    self.writer.writerow([f"{t:.6f}", valid, f"{mz:.6f}", f"{mx:.6f}", f"{my:.6f}",
                                          f"{qz:.7f}", f"{qx:.7f}", f"{qy:.7f}", f"{qw:.7f}"])
                    self.n += 1
                    self.n_valid += valid
                    if t - self.last_flush > 1.0:
                        self.file.flush()
                        self.last_flush = t
        except Exception:
            self.errors += 1

    def mark_flight_start(self, t0_monotonic):
        """Time origin of the trajectory log (TrajectoryLogger.start_time)."""
        try:
            self.meta["flight_log_t0_monotonic"] = t0_monotonic
            self._write_meta()
        except Exception:
            self.errors += 1

    def close(self):
        with self.lock:
            if self.file is None:
                return
            self.file.close()
            self.file = None
        self.meta.update({"frames": self.n, "valid_frames": self.n_valid, "errors": self.errors})
        self._write_meta()
        print(f"Camera (rigid body {self.rigid_body_id}): {self.n_valid}/{self.n} valid frames -> {self.path}")
