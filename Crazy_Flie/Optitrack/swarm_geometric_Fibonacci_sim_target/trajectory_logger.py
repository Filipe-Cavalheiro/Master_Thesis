import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from cflib.crazyflie.log import LogConfig

from Structs import RIGID_BODY_IDS

class TrajectoryLogger:
    """
    Logs, at a fixed rate, into a single long-format CSV (one row per drone per sample):
      - OptiTrack position (ground truth, same frame sent to the EKF)
      - Kalman estimate: position, velocity, attitude (deg), gyro (deg/s)
    `id` is the OptiTrack rigid-body ID of the drone.

    Next to it (same time base, time_s):
      - <log>_setpoints.csv: every position setpoint sent (record_setpoint),
        with the virtual target, radius and q_ref of the controller
      - <log>_meta.json: the flight setup (write_meta) and, at close(), the
        measured setpoint periods per phase and drone
    """

    SETPOINT_COLUMNS = [
        "time_s", "flight_state", "id", "sp_x", "sp_y", "sp_z",
        "target_x", "target_y", "target_z", "radius", "q_x", "q_y", "q_z",
    ]

    COLUMNS = [
        "time_s", "flight_state", "id",
        "opti_x", "opti_y", "opti_z",
        "est_x", "est_y", "est_z",
        "est_vx", "est_vy", "est_vz",
        "roll_deg", "pitch_deg", "yaw_deg",
        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    ]

    VAR_TO_COL = {
        "stateEstimate.x": "est_x",
        "stateEstimate.y": "est_y",
        "stateEstimate.z": "est_z",
        "stateEstimate.vx": "est_vx",
        "stateEstimate.vy": "est_vy",
        "stateEstimate.vz": "est_vz",
        "stateEstimate.roll": "roll_deg",
        "stateEstimate.pitch": "pitch_deg",
        "stateEstimate.yaw": "yaw_deg",
        "gyro.x": "gyro_x_dps",
        "gyro.y": "gyro_y_dps",
        "gyro.z": "gyro_z_dps",
    }

    def __init__(self, drone_states, position_lock,
                 output_dir=None, run_name=None, sample_period=0.02):
        self.drone_states = drone_states          # {uri: DroneState}
        self.position_lock = position_lock
        self.sample_period = sample_period

        # One folder per run, in the repository Results folder:
        #   Results/<run_name>_trajectory_<timestamp>/trajectory_<timestamp>.csv
        # (the files derived from the log, _sim.csv, _camera.csv, ..., go next to it)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[3] / "Results"
        run_name = run_name or Path(__file__).resolve().parent.name
        self.run_dir = output_dir / f"{run_name}_trajectory_{timestamp}"
        self.output_path = self.run_dir / f"trajectory_{timestamp}.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.setpoints_path = self.run_dir / f"trajectory_{timestamp}_setpoints.csv"
        self.meta_path = self.run_dir / f"trajectory_{timestamp}_meta.json"
        self.meta = {}
        self.sp_file = None
        self.sp_writer = None
        self.sp_lock = threading.Lock()
        self.sp_times = {}          # {(phase, id): [time_s, ...]}

        self.stop_evt = threading.Event()
        self.thread = None
        self.start_time = None
        self.current_state = ""

        # Latest Crazyflie values, per drone: {uri: {col: value}}
        self.latest = {uri: {} for uri in drone_states}
        self.latest_lock = threading.Lock()

        self.log_configs = {uri: [] for uri in drone_states}
        self.file = None
        self.writer = None

    # ---------------- Crazyflie log blocks (per drone) ----------------
    @staticmethod
    def _make_log_configs():
        # A log packet holds max 26 bytes -> 6 floats (24 B) per block
        pos_vel = LogConfig(name="KalmanPosVel", period_in_ms=20)
        for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                  "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
            pos_vel.add_variable(v, "float")

        att_gyro = LogConfig(name="AttGyro", period_in_ms=20)
        for v in ("stateEstimate.roll", "stateEstimate.pitch",
                  "stateEstimate.yaw",
                  "gyro.x", "gyro.y", "gyro.z"):
            att_gyro.add_variable(v, "float")

        return [pos_vel, att_gyro]

    def _on_log_data(self, timestamp, data, logconf, uri):
        with self.latest_lock:
            store = self.latest[uri]
            for var, value in data.items():
                col = self.VAR_TO_COL.get(var)
                if col:
                    store[col] = value

    def start_logging(self, scf):
        """Run via swarm.parallel_safe(logger.start_logging): one call per drone."""
        uri = scf.cf.link_uri
        for lc in self._make_log_configs():
            scf.cf.log.add_config(lc)
            # bind uri so the shared callback knows which drone the data is from
            lc.data_received_cb.add_callback(
                lambda ts, data, conf, uri=uri: self._on_log_data(ts, data, conf, uri)
            )
            lc.start()
            self.log_configs[uri].append(lc)
        print(f"{uri}: logging blocks started.")

    def stop_logging(self, scf):
        """Run via swarm.parallel_safe(logger.stop_logging): one call per drone."""
        uri = scf.cf.link_uri
        for lc in self.log_configs[uri]:
            try:
                lc.stop()
                lc.delete()
            except Exception:
                pass
        self.log_configs[uri].clear()

    # ---------------- Sampling thread (shared) ----------------
    def begin(self):
        """Open the CSV and start the sampling thread. Call after start_logging."""
        self.start_time = time.monotonic()
        self.file = self.output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.COLUMNS)
        self.writer.writeheader()
        with self.sp_lock:
            self.sp_file = self.setpoints_path.open("w", newline="", encoding="utf-8")
            self.sp_writer = csv.DictWriter(self.sp_file, fieldnames=self.SETPOINT_COLUMNS)
            self.sp_writer.writeheader()

        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"Started logging {len(self.drone_states)} drones at {1/self.sample_period:.0f} Hz.")

    def now(self):
        """Time on the log's time base (time_s), None before begin()."""
        return None if self.start_time is None else time.monotonic() - self.start_time

    def record_setpoint(self, uri, sp, target=None, radius=None, q_ref=None):
        """One position setpoint sent to drone `uri` (thread safe; ignored outside begin()..close())."""
        t = self.now()
        if t is None:
            return
        sp = np.asarray(sp, dtype=float).reshape(-1)
        row = {"time_s": t, "flight_state": self.current_state, "id": RIGID_BODY_IDS[uri],
               "sp_x": sp[0], "sp_y": sp[1], "sp_z": sp[2]}
        if target is not None:
            tg = np.asarray(target, dtype=float).reshape(-1)
            row.update(target_x=tg[0], target_y=tg[1], target_z=tg[2])
        if radius is not None:
            row["radius"] = float(radius)
        if q_ref is not None:
            q = np.asarray(q_ref, dtype=float).reshape(-1)
            row.update(q_x=q[0], q_y=q[1], q_z=q[2])
        with self.sp_lock:
            if self.sp_writer is None:
                return
            self.sp_writer.writerow(row)
            self.sp_times.setdefault((self.current_state, str(RIGID_BODY_IDS[uri])), []).append(t)

    def write_meta(self, meta):
        """Save the flight setup to <log>_meta.json (merged with what is already there)."""
        self.meta.update(meta)
        self.meta_path.write_text(json.dumps(self.meta, indent=2, default=_json_default))

    def _setpoint_periods(self):
        """Measured period between setpoints, per phase and drone."""
        out = {}
        for (phase, rb_id), ts in self.sp_times.items():
            d = np.diff(ts)
            if len(d):
                out.setdefault(phase, {})[rb_id] = {
                    "n_setpoints": len(ts), "first_s": ts[0], "last_s": ts[-1],
                    "period_mean_s": float(d.mean()), "period_std_s": float(d.std()),
                    "period_max_s": float(d.max())}
        return out

    def mark_state(self, state_name):
        self.current_state = state_name
        print(f"[Log] State: {state_name}")

    def close(self):
        """Stop the sampling thread and close the file."""
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.file is not None:
            self.file.close()
            self.file = None
        with self.sp_lock:
            if self.sp_file is not None:
                self.sp_file.close()
                self.sp_file = self.sp_writer = None
        self.write_meta({"measured_setpoint_periods": self._setpoint_periods()})
        print(f"Log saved to: {self.output_path}")

    def _run(self):
        next_sample = time.monotonic()
        rows_since_flush = 0

        while not self.stop_evt.is_set():
            t = time.monotonic() - self.start_time   # same timestamp for all drones
            state = self.current_state

            with self.position_lock:
                opti = {
                    uri: np.asarray(s.position, dtype=float).reshape(-1)[:3].copy()
                    for uri, s in self.drone_states.items()
                }
            with self.latest_lock:
                est = {uri: dict(v) for uri, v in self.latest.items()}

            for uri in self.drone_states:
                ox, oy, oz = opti[uri]
                row = {
                    "time_s": t,
                    "flight_state": state,
                    "id": RIGID_BODY_IDS[uri],
                    "opti_x": ox, "opti_y": oy, "opti_z": oz,
                }
                row.update(est[uri])   # missing keys are written as empty
                self.writer.writerow(row)
                rows_since_flush += 1

            if rows_since_flush >= 50 * len(self.drone_states):   # ~once per second
                self.file.flush()
                rows_since_flush = 0

            next_sample += self.sample_period
            delay = next_sample - time.monotonic()
            if delay > 0:
                self.stop_evt.wait(delay)
            else:
                next_sample = time.monotonic()

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
