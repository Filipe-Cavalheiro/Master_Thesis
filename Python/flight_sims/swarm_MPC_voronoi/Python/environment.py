import numpy as np

# Obstacles seen by a drone are vertical columns (2.5D world: buildings and
# terrain have no overhangs), stored as rows [x0, x1, y0, y1, top]: the solid
# {x0 <= x <= x1, y0 <= y <= y1, z <= top}.
NO_OBSTACLES = np.zeros((0, 5))


def column_distance(P, boxes):
    """
    Distance from points P (n, 3) to obstacle columns (m, 5), the closest
    point of each column and whether the point is inside it.

    Returns (d (n, m), closest (n, m, 3), inside (n, m)).
    """
    x0, x1, y0, y1, top = (boxes[:, i][None] for i in range(5))
    px, py, pz = (P[:, i][:, None] for i in range(3))
    cx = np.clip(px, x0, x1)
    cy = np.clip(py, y0, y1)
    cz = np.minimum(pz, top)
    d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
    inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1) & (pz <= top)
    return d, np.stack([cx, cy, cz], axis=-1), inside


def project_out(P, boxes, clearance, prefer_up=False, iterations=4):
    """
    Push points (n, 3) at least `clearance` away from the obstacle columns.

    A point inside an obstacle moves to the nearest free position: sideways to
    the closest sensed cell that is free at its height, or up over the
    obstacle (prefer_up=True: always up, i.e. fly over). A point closer than
    `clearance` is pushed straight away from the closest obstacle point.
    """
    P = np.array(P, dtype=float, copy=True)
    if len(boxes) == 0:
        return P
    centres = np.column_stack([(boxes[:, 0] + boxes[:, 1]) / 2, (boxes[:, 2] + boxes[:, 3]) / 2])
    for _ in range(iterations):
        d, closest, inside = column_distance(P, boxes)
        moved = False
        for k in np.flatnonzero(inside.any(axis=1)):
            up = boxes[inside[k], 4].max() + clearance
            target = np.array([P[k, 0], P[k, 1], up])
            if not prefer_up:
                free = boxes[:, 4] <= P[k, 2] - clearance          # cells free at this height
                if np.any(free):
                    dist = np.linalg.norm(centres[free] - P[k, :2], axis=1)
                    j = np.argmin(dist)
                    if dist[j] < up - P[k, 2]:
                        target = np.array([*centres[free][j], P[k, 2]])
            P[k] = target
            moved = True
        d, closest, inside = column_distance(P, boxes)
        dmin = d.min(axis=1)
        k = np.flatnonzero(~inside.any(axis=1) & (dmin < clearance - 1e-9))
        if len(k):
            j = np.argmin(d[k], axis=1)
            direction = P[k] - closest[k, j]
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            direction = np.where(norm > 1e-9, direction / np.maximum(norm, 1e-9), np.array([0.0, 0.0, 1.0]))
            P[k] = closest[k, j] + clearance * direction
            moved = True
        if not moved:
            break
    return P


def safe_setpoint(position, velocity, acceleration, boxes, clearance, drone_position=None, brake_gain=1.0):
    """
    Safety filter of a position setpoint (3, 1): pushed out of the obstacles,
    with the velocity / acceleration components towards them removed. With
    drone_position, the velocity towards the closest obstacle is also limited
    to brake_gain * (distance - clearance), so the drone slows down before
    reaching the clearance instead of overshooting it.
    """
    if len(boxes) == 0:
        return position, velocity, acceleration
    if drone_position is not None:
        here = drone_position.reshape(1, 3)
        d, closest, inside = column_distance(here, boxes)
        j = np.argmin(d[0])
        if d[0, j] > 1e-9 and not inside[0, j]:
            n = (closest[0, j] - here[0]) / d[0, j]               # towards the obstacle
            allowed = brake_gain * max(d[0, j] - clearance, 0.0)
            v = velocity.ravel()
            if v @ n > allowed:
                velocity = (v - (v @ n - allowed) * n).reshape(-1, 1)
    p = position.reshape(1, 3)
    q = project_out(p, boxes, clearance)
    push = (q - p).ravel()
    if np.linalg.norm(push) < 1e-9:
        return position, velocity, acceleration
    n = push / np.linalg.norm(push)
    v, a = velocity.ravel().copy(), acceleration.ravel().copy()
    if v @ n < 0:
        v -= (v @ n) * n
    if a @ n < 0:
        a -= (a @ n) * n
    return q.reshape(-1, 1), v.reshape(-1, 1), a.reshape(-1, 1)


class FlatGround:
    """Environment of the Python simulation: flat ground at z = 0."""

    def detect(self, position, sensing_range):
        """Obstacle columns within sensing_range of the drone."""
        if np.ravel(position)[2] < sensing_range:
            return np.array([[-1e9, 1e9, -1e9, 1e9, 0.0]])
        return NO_OBSTACLES

    def height(self, xy):
        return np.zeros(len(np.atleast_2d(xy)))


class HeightField:
    """
    2.5D environment from a regular grid of column tops (e.g. the 3D city
    map): cell (i, j) is the column [x0 + j*res, x0 + (j+1)*res] x
    [y0 + i*res, y0 + (i+1)*res] up to H[i, j]. Outside the grid the ground
    is flat at `outside`.
    """

    def __init__(self, x0, y0, res, H, outside=0.0):
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.H = np.asarray(H, dtype=float)
        self.outside = float(outside)

    def detect(self, position, sensing_range):
        """Obstacle columns within sensing_range of the drone (its sensor)."""
        p = np.ravel(position)
        r = sensing_range
        ny, nx = self.H.shape
        j0 = int(np.floor((p[0] - r - self.x0) / self.res))
        j1 = int(np.floor((p[0] + r - self.x0) / self.res))
        i0 = int(np.floor((p[1] - r - self.y0) / self.res))
        i1 = int(np.floor((p[1] + r - self.y0) / self.res))
        if j1 < 0 or i1 < 0 or j0 >= nx or i0 >= ny:
            # outside the grid: flat ground
            return FlatGround().detect(p - [0, 0, self.outside], r) + [0, 0, 0, 0, self.outside]
        j0, i0 = max(j0, 0), max(i0, 0)
        j1, i1 = min(j1, nx - 1), min(i1, ny - 1)
        jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
        x0 = self.x0 + jj.ravel() * self.res
        y0 = self.y0 + ii.ravel() * self.res
        boxes = np.column_stack([x0, x0 + self.res, y0, y0 + self.res, self.H[ii.ravel(), jj.ravel()]])
        # every cell within the horizontal range: obstacles and free space
        dx = np.maximum(np.maximum(boxes[:, 0] - p[0], p[0] - boxes[:, 1]), 0)
        dy = np.maximum(np.maximum(boxes[:, 2] - p[1], p[1] - boxes[:, 3]), 0)
        boxes = boxes[np.hypot(dx, dy) < r]
        # nothing within range vertically: no obstacle sensed
        d, _, _ = column_distance(p.reshape(1, 3), boxes)
        return boxes if np.any(d[0] < r) else NO_OBSTACLES

    def height(self, xy):
        xy = np.atleast_2d(xy)
        j = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        i = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.H.shape[0]) & (j >= 0) & (j < self.H.shape[1])
        h = np.full(len(xy), self.outside)
        h[ok] = self.H[i[ok], j[ok]]
        return h


def avoid_obstacles_reference(agent, environment, param):
    """
    Reactive collision avoidance of a drone that follows a reference without
    MPC, with the obstacles it senses (sensing_range):
      - reference near an obstacle: pushed out (it stops in front of walls);
      - reference inside an obstacle (e.g. its path crosses a building): the
        drone holds at the wall at its current height and climbs to the top
        of the obstacle + clearance, then flies over it.
    """
    obstacles = environment.detect(agent.position, param.sim.sensing_range)
    agent.perceived_obstacles = obstacles
    if len(obstacles) == 0:
        return agent
    clr = param.sim.env_clearance
    ref = agent.desired.position.reshape(1, 3)
    # only plan within what is sensed: a reference farther than the sensing
    # range is brought back towards the drone
    here = agent.position.reshape(1, 3)
    reach = max(param.sim.sensing_range - clr, 0.0)
    gap = np.linalg.norm(ref - here)
    if gap > reach:
        ref = here + (ref - here) * (reach / gap)
    over = project_out(ref, obstacles, clr, prefer_up=True)
    if over[0, 2] > agent.position[2, 0] + 0.5:
        # climb first: hold the current horizontal position (kept at
        # clearance from the wall) and go up to the top of the obstacle
        hold = project_out(agent.position.reshape(1, 3), obstacles, clr)
        z_now = agent.position[2, 0]
        climb = param.sim.env_climb_speed
        # moving setpoint at climb speed (no 10+ m setpoint jump)
        z_set = min(over[0, 2], z_now + climb * 0.5)
        agent.desired.position = np.array([[hold[0, 0]], [hold[0, 1]], [z_set]])
        agent.desired.velocity = np.array([[0.0], [0.0], [min(climb, max(over[0, 2] - z_now, 0.0))]])
        return agent
    p, v, _ = safe_setpoint(over.reshape(-1, 1), agent.desired.velocity, np.zeros((3, 1)), obstacles, clr,
                            agent.position, param.sim.env_brake_gain)
    # resuming after a hold/climb the reference can be metres ahead: move the
    # setpoint progressively (at most climb_speed * 0.5 s from the drone)
    step = p - agent.position
    max_step = param.sim.env_climb_speed * 0.5
    if np.linalg.norm(step) > max_step:
        p = agent.position + step * (max_step / np.linalg.norm(step))
    agent.desired.position, agent.desired.velocity = p, v
    return agent
