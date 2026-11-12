import time
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment

from Structs import *
from voronoi_mesh_funcs import *

class Formation:
    """
    Target and Voronoi formation shared by the controllers (the target
    thread, voronoi_worker and desired_positions of the lighthouse version).
    Desired position i belongs to drone i of the order given to assign_slots.

    target_state None: virtual target, the formation follows the target
    path. Else the DroneState of the target drone (it flies the path, see
    target_reference): the formation follows its OptiTrack position and
    velocity.
    """

    def __init__(self, param: Param, target_state: DroneState | None = None,
                 position_lock=None):
        self.param = param
        self.lock = threading.Lock()
        self.target_state = target_state
        self.position_lock = position_lock
        self.start_time = None      # time.monotonic() at which the target path starts

        self.voronoi = Voronoi(param)
        p_d, _, self.voronoi = pre_compute_desired_pos(self.voronoi, param)
        self.target = np.array(param.initial_target_pos, dtype=float)
        self.target_velocity = np.zeros(3)
        self.desired_positions = clip_goals(p_d)

    def converge(self, max_iters=CVT_PRECONVERGE_ITERS):
        """CVT on the initial mesh (target still), so the drones fly to spread slots."""
        for k in range(max_iters):
            new_sites = runCVT(self.voronoi, self.voronoi.sites)
            if np.array_equal(new_sites, self.voronoi.sites):
                break
            self.voronoi.sites = new_sites
        with self.lock:
            self.desired_positions = clip_goals(self.voronoi.vertices[self.voronoi.sites, :])
        print(f"Formation converged after {k + 1} CVT iterations.")

    def assign_slots(self, positions):
        """
        Reorders the sites so drone i (row i of `positions`) gets the slot
        minimizing the sum of squared distances. Returns the slots.
        """
        with self.lock:
            slots = self.desired_positions
            cost = np.sum((np.asarray(positions)[:, None, :] - slots[None, :, :]) ** 2, axis=2)
            _, slot_of = linear_sum_assignment(cost)
            self.voronoi.sites = self.voronoi.sites[slot_of]
            self.desired_positions = slots[slot_of]
            return self.desired_positions.copy()

    def min_slot_distance(self):
        with self.lock:
            p = self.desired_positions
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        return float(np.min(d[np.triu_indices(len(p), k=1)])) if len(p) > 1 else np.inf

    def goal(self, idx):
        with self.lock:
            return self.desired_positions[idx].copy()

    def get_target(self):
        with self.lock:
            return self.target.copy()

    def get_target_state(self):
        """Target position and velocity (measured for a target drone)."""
        with self.lock:
            return self.target.copy(), self.target_velocity.copy()

    def get_goals(self):
        """
        Desired positions (nD, 3) and the goal velocity (3,) shared by all
        drones: the target velocity, without its x component while the
        formation is held on the x = 0 plane (target before the wall, as
        the wall-normal component dropped in Python/compute_desired_goal.py).
        """
        with self.lock:
            goal_vel = self.target_velocity.copy()
            n = self.param.wall_normal
            if np.dot(self.target - self.param.wall_point, n) < self.param.sphere_radius / 2:
                goal_vel -= np.dot(goal_vel, n) * n
            return self.desired_positions.copy(), goal_vel

    # ============================================================
    # Threads
    # ============================================================
    def update_virtual_target(self, stop_event: threading.Event):
        start_time = self.start_time
        while not stop_event.is_set():
            if self.target_state is not None:
                with self.position_lock:
                    pos = np.ravel(self.target_state.position)[:3].copy()
                    vel = np.ravel(self.target_state.velocity)[:3].copy()
            else:
                t = time.monotonic() - start_time
                pos = self.param.ref_target_traj["pos"](t)
                vel = self.param.ref_target_traj["vel"](t)
            with self.lock:
                self.target = np.array(pos, dtype=float)
                self.target_velocity = np.array(vel, dtype=float)
            stop_event.wait(self.param.dTi)

    def voronoi_worker(self, stop_event: threading.Event):
        voronoi = self.voronoi
        k = 0
        while not stop_event.is_set():
            target_pos = self.get_target()

            voronoi.vertices, voronoi.faces, voronoi.mesh_pos = (
                pos_based_mesh_transform(target_pos, voronoi, self.param)
            )

            if k == 1 or k % 5 == 0:
                voronoi.sites = runCVT(voronoi, voronoi.sites)

            p_d = clip_goals(voronoi.vertices[voronoi.sites, :])
            with self.lock:
                self.desired_positions = p_d

            k += 1
            stop_event.wait(self.param.dTi)

    def start_threads(self, stop_event: threading.Event):
        self.start_time = time.monotonic()
        threads = [
            threading.Thread(target=self.update_virtual_target, args=(stop_event,), daemon=True),
            threading.Thread(target=self.voronoi_worker, args=(stop_event,), daemon=True),
        ]
        for t in threads:
            t.start()
        return threads


def target_reference(param: Param, t):
    """
    Path of the target drone at time t of the Formation phase: position,
    velocity (3,), kept in the geofence (velocity zero along a clipped axis).
    """
    p = np.asarray(param.ref_target_traj["pos"](t), dtype=float).ravel()
    v = np.asarray(param.ref_target_traj["vel"](t), dtype=float).ravel()
    p_clip = np.clip(p, GOAL_MIN, GOAL_MAX)
    v = np.where(p_clip != p, 0.0, v)
    return p_clip, v


def clip_goals(p_d):
    """Keeps the formation goals inside the arena (GEOFENCE_MARGIN from the walls), in front of the wall (trajectory 5)."""
    return clip_to_arena(p_d)
