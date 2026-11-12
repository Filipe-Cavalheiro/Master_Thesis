"""
Crazyflie arena experiment: an intrusion across a straight red border edge
(as trajectory 5) in the OptiTrack arena, with the Crazyflie model instead of
the X500.

Used by the Python simulation (SimulationConfig.arena_experiment = True) and
by the real flight (Crazy_Flies/), so both fly the same scenario.

Top view (OptiTrack frame, origin at the centre of the floor). The drones fly
in x in [-2.5, 2.5]; the 0.5 m strips at both ends of the 6 m room are only
for the (virtual) target: outside the wall, and inside the red zone with no
drone around it. The target flies a loop: in through the wall, across the
orange zone (encircled), into the red zone (the drones hold on the red
border), along it, back out across the orange zone and along the wall.

    y      wall                                     red border
    ^       |            orange zone                   |   red zone
    |  D <--|--------------------------------------------|--- C
    |  |    |                                          |    ^
    |  v    |                                          |    |
    |  A ---|--------------------------------------------|--> B
    +-------+------------------------------------------+-------> x
     -2.6  -2.0                                       2.2  2.6
"""
import numpy as np

from restricted_airspace import RestrictedAirspace, Polygon2D

# The room is 6 x 6 x 3 m; the drones only use the central 5 x 5 x 2.5 m.
ARENA_MIN = np.array([-2.5, -2.5, 0.0])
ARENA_MAX = np.array([2.5, 2.5, 2.5])
GEOFENCE_MARGIN = 0.3     # [m] goals, horizons and setpoints kept this far from the walls and the
                          # 2.5 m ceiling; no minimum height (floor at z = 0)

DRONE_MODEL = "cf2x_L250"     # entry of drone_models.toml

RED_BORDER_X = 2.2        # [m] straight red border, red zone at x > RED_BORDER_X
WALL_X = -2.0             # [m] wall: the drones wait on it until the target crosses it (at -2.2 they
                          # held ~8 cm past their slots, 0.17 m from the -2.5 m arena edge)
FAR = 20.0                # [m] the zones extend far beyond the arena (straight border, no joints nearby)

# Target path: loop A -> B -> C -> D -> A (see the top view), starting
# outside the wall. Its y extent leaves room for the formation (radius 0.6 m)
# inside the arena.
TARGET_HEIGHT = 1.3       # [m] formation (radius 0.6 m) between ~0.7 and ~1.9 m
TARGET_X = (-2.6, 2.6)    # [m] outside the wall / inside the red zone
TARGET_Y = (-1.2, 1.2)    # [m]
TARGET_SPEED = 0.3        # [m/s]

# Trajectory 5: a window of the Lisbon trajectory 5 (Python/Gazebo, X500,
# formation radius 5 m) in the arena, with the formation radius 0.8 m.
# Lengths are scaled by TRAJ5_SCALE and time is kept, so the formation goes
# through the same wall -> cone -> sphere passage at the same pace: target
# speed, MPC velocity / acceleration limits, wall to red border distance
# and red border margin are the Lisbon values times TRAJ5_SCALE (the target
# measurement noise is not scaled: it models the sensor, not the formation); the MPC horizon (1 s) and the CVT update period (1 s) are the
# same. The wall has the orientation of the Lisbon wall at the crossing
# (KMZ red edge 1 -> 2), in the arena frame (x axis towards ARENA_X_BEARING,
# z up), and the target crosses it with the same velocity 1.5 n + 0.5 t
# (scaled). Differences: the target
# starts 0.5 m (Lisbon: 20 m = 4 radii) before the wall, flies at 1.3 m
# (Lisbon 15 m = 3 radii, the arena is 2.5 m high) and stops when the sphere
# reaches the arena geofence, 2.9 m past the wall (Lisbon: 40 m inside the
# red zone); the red border (30 m -> 4.8 m past the wall) is outside the
# arena; the inter-drone envelope is a 0.3 m sphere (Crazyflies).
LISBON_RADIUS = 5.0       # [m] formation radius of the Lisbon simulation
TRAJ5_RADIUS = 0.8        # [m] formation radius in the arena
TRAJ5_SCALE = TRAJ5_RADIUS / LISBON_RADIUS
ARENA_X_BEARING = 7.0     # [deg] the arena x axis points 007 (from North, clockwise); y points 277
TRAJ5_N_IN_ENU = np.array([0.92230468, -0.38646355])   # Lisbon wall normal towards the red zone (East,
                                                       # North): bearing 112.7, from the KMZ (check_traj5_geometry)


def enu_to_arena(v_enu):
    """Horizontal ENU vector (East, North) in the arena frame (x at ARENA_X_BEARING, y 90 deg to its left)."""
    b = np.radians(ARENA_X_BEARING)
    x_axis = np.array([np.sin(b), np.cos(b)])
    y_axis = np.array([-np.cos(b), np.sin(b)])
    return np.array([v_enu @ x_axis, v_enu @ y_axis])


TRAJ5_N_IN = enu_to_arena(TRAJ5_N_IN_ENU)   # [-0.271, -0.963]: the target crosses the arena along -y
TRAJ5_START = np.array([-0.09, 2.20])    # [m] target start, 0.5 m before the wall; x centres the path, y
                                         # on the geofence (the target can be a Crazyflie, swarm_MPC_voronoi)
TRAJ5_HOLD = 5.0          # [s] simulated after the target stops


class ArenaAirspace(RestrictedAirspace):
    """
    Straight red border with normal n_in (towards the red zone), the drones'
    wall wall_offset before it through wall_point; red zone beyond the
    border, orange zone before it (max altitude orange_max_agl), same
    interface as the KMZ airspace. project() also keeps the points inside the
    arena (GEOFENCE_MARGIN from the walls and the ceiling, above the floor),
    so formation goals and MPC horizons stay in the arena. Default: the loop,
    wall at x = WALL_X, red border at x = RED_BORDER_X.
    """
    kind = "arena"

    def __init__(self, sim, n_in=(1.0, 0.0), wall_point=(WALL_X, 0.0)):
        n = np.asarray(n_in, dtype=float) / np.linalg.norm(n_in)
        t = np.array([-n[1], n[0]])
        self.n_in = n
        self.wall_point = np.asarray(wall_point, dtype=float)
        b = self.wall_point + sim.wall_offset * n          # point of the red border
        red = np.array([b - FAR * t, b - FAR * t + FAR * n, b + FAR * t + FAR * n, b + FAR * t])
        orange = np.array([b - FAR * n - FAR * t, b - FAR * t, b + FAR * t, b - FAR * n + FAR * t])
        self.red = Polygon2D(red)
        self.orange = [Polygon2D(orange)]
        self.red_zone_name = sim.red_zone_name
        self.orange_zone_name = sim.orange_zone_name
        self.zones = [(sim.red_zone_name, [red]), (sim.orange_zone_name, [orange])]
        self.wall_offset = sim.wall_offset
        self.wall_joint_blend = sim.wall_joint_blend
        m = len(self.red.vertices)
        self.adjacent_edges = {frozenset((i, (i + 1) % m)) for i in range(m)}
        self.red_margin = sim.red_margin
        self.orange_max_agl = sim.orange_max_agl

        self.box_min = ARENA_MIN + GEOFENCE_MARGIN
        self.box_min[2] = ARENA_MIN[2]          # no minimum height
        self.box_max = ARENA_MAX - GEOFENCE_MARGIN

    def project(self, P):
        # Closed form of RestrictedAirspace.project for the straight border
        # (same result near the arena): out of the red zone by red_margin,
        # then everything is in the orange zone, under its ceiling (flat
        # ground). The general polygon code is the most expensive part of the
        # real-time MPC.
        P = np.array(P, dtype=float, copy=True)
        # nothing behind the drones' wall (the formation mesh near the wall
        # reaches up to half a radius behind it)
        u = (P[:, :2] - self.wall_point) @ self.n_in
        u_clip = np.clip(u, 0.0, self.wall_offset - self.red_margin)
        P[:, :2] += (u_clip - u)[:, None] * self.n_in
        P[:, 2] = np.minimum(P[:, 2], self.orange_max_agl)
        return np.clip(P, self.box_min, self.box_max)


class ArenaTrajectory:
    """Target of the arena experiment (TrajectoryConfig interface): the loop A B C D A at TARGET_SPEED."""

    def __init__(self):
        (x0, x1), (y0, y1) = TARGET_X, TARGET_Y
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
        t_corner = np.concatenate([[0.0], np.cumsum(lengths)]) / TARGET_SPEED
        t_stop = t_corner[-1]

        def leg(t):
            return min(int(np.searchsorted(t_corner, t, side="right")) - 1, len(lengths) - 1)

        def pos(t):
            t = min(max(t, 0.0), t_stop)
            k = leg(t)
            f = (t - t_corner[k]) / (t_corner[k + 1] - t_corner[k])
            xy = corners[k] + f * (corners[k + 1] - corners[k])
            return np.array([[xy[0]], [xy[1]], [TARGET_HEIGHT]])

        def vel_fn(t):
            if not 0.0 <= t < t_stop:
                return np.zeros((3, 1))
            k = leg(t)
            v = (corners[k + 1] - corners[k]) / lengths[k] * TARGET_SPEED
            return np.array([[v[0]], [v[1]], [0.0]])

        self.ref_target_traj = {"pos": pos, "vel": vel_fn}
        self.initial_target_pos = pos(0)
        self.t_stop = t_stop


class ArenaTrajectory5:
    """
    Arena trajectory 5 (TrajectoryConfig interface): the Lisbon trajectory 5
    scaled (see TRAJ5_*), from TRAJ5_START in a straight line at the scaled
    Lisbon velocity until the formation sphere around it reaches the arena
    geofence, then still.
    """

    def __init__(self):
        n = TRAJ5_N_IN
        t = np.array([-n[1], n[0]])
        vel = (1.5 * n + 0.5 * t) * TRAJ5_SCALE          # [m/s] Lisbon 1.5 m/s across the wall, 0.5 along it
        start = TRAJ5_START
        # stop: the sphere (radius TRAJ5_RADIUS) around the target still in the geofence
        lim = ARENA_MAX[:2] - GEOFENCE_MARGIN - TRAJ5_RADIUS
        t_stop = min((np.sign(v) * lim[i] - start[i]) / v for i, v in enumerate(vel) if v != 0)

        def pos(t_):
            xy = start + vel * min(max(t_, 0.0), t_stop)
            return np.array([[xy[0]], [xy[1]], [TARGET_HEIGHT]])

        def vel_fn(t_):
            v = vel if 0.0 <= t_ < t_stop else np.zeros(2)
            return np.array([[v[0]], [v[1]], [0.0]])

        self.ref_target_traj = {"pos": pos, "vel": vel_fn}
        self.initial_target_pos = pos(0)
        self.t_stop = t_stop
        self.wall_point = start + 0.5 * n                # the wall 0.5 m ahead of the start


def check_traj5_geometry():
    """TRAJ5_N_IN against the Lisbon KMZ (needs Assets/): the wall normal of trajectory 5."""
    from Structs_sim import SimulationConfig
    from restricted_airspace import RestrictedAirspace
    red = RestrictedAirspace(SimulationConfig()).red
    crossing = 0.6 * red.vertices[1] + 0.4 * red.vertices[2]   # as in TrajectoryConfig, trajectory 5
    n_out, _, _ = red.outward_normal(crossing.reshape(1, 2))
    return -n_out[0]


# Onboard controller of the real Crazyflies (Mellinger, Kalman with the
# OptiTrack position), as seen by the setpoint -> OptiTrack position response
# of the 26 Oct flights (Crazy_Flies/flight_sim.py). Position gains [N/m,
# N s/m] as in the firmware (controller_mellinger.c: kp_xy 0.4, kd_xy 0.2,
# kp_z 1.25, kd_z 0.4); attitude loop with natural frequency wn_att.
CRAZYFLIE_ONBOARD = {
    "kp": [0.4, 0.4, 1.25],
    "kv": [0.2, 0.2, 0.4],
    "wn_att": 20.0,       # [rad/s]
    "max_tilt_deg": 30.0,
    "delay": 0.02,        # [s] setpoint -> onboard controller (radio, Kalman)
    "z_offset": 0.03,     # [m] the drones hover this far below their setpoint
    "rate": 500,          # [Hz] onboard control loop
}


def configure_onboard_gains(param, onboard=CRAZYFLIE_ONBOARD):
    """Mellinger gains of param.controller (Python simulation) set to the Crazyflie onboard controller."""
    J = param.sim.J
    wn = onboard["wn_att"]
    param.controller.Mellinger["kp"] = np.diag(onboard["kp"])
    param.controller.Mellinger["kv"] = np.diag(onboard["kv"])
    param.controller.Mellinger["kR"] = J * wn ** 2
    param.controller.Mellinger["kom"] = J * 2 * wn
    param.controller.max_tilt_deg = onboard["max_tilt_deg"]


def configure_arena_experiment(param, dTi=None):
    """
    Turn param (Structs_sim.Param) into the arena experiment: Crazyflie model
    and controller gains, parameters scaled to the arena, arena airspace and
    target path (param.sim.arena_traj: 5 = Lisbon trajectory 5 window,
    "loop" = the A B C D loop). dTi: control period (None keeps
    param.sim.dTi; the real flight uses its own control period).
    """
    sim = param.sim
    sim.load_drone_model(DRONE_MODEL)
    arena_traj = getattr(sim, "arena_traj", 5)
    if arena_traj == 5:
        traj = ArenaTrajectory5()
    elif arena_traj == "loop":
        traj = ArenaTrajectory()
    else:
        raise ValueError(f"Unknown arena trajectory {arena_traj!r} (5 or 'loop')")

    if dTi is not None:
        sim.dTi = dTi
    sim.Tend = traj.t_stop + (TRAJ5_HOLD if arena_traj == 5 else 2.0)
    sim.Nsim = round(sim.Tend / sim.dTi) + 1
    sim.t = np.arange(0, sim.Tend + sim.dTi, sim.dTi)

    sim.nD_drop_time = -1
    sim.update_CVT_interval = max(1, round(1.0 / sim.dTi))   # CVT update every 1 s
    sim.converge_initial_formation = True  # the flight converges the formation before arming
    sim.mpc_tight_tracking = True         # MPC as in the flight code (the Crazyflies track their setpoints tightly)
    sim.target_is_drone = getattr(sim, "arena_target_drone", True)   # target Crazyflie, avoided by the MPC
    sim.replan_reset_dist = 0.2           # [m]
    sim.obstacle_avoidance_dim = 0.3      # [m] radius of the spherical inter-drone (and target) envelope
    sim.rho_obs = 5000.0                  # as the flight code (MPC_RHO_OBS): with 100 the goal cost keeps them stacked

    if arena_traj == 5:
        # Lisbon values (SimulationConfig defaults) scaled, same time
        s = TRAJ5_SCALE
        sim.sphere_radius = TRAJ5_RADIUS
        sim.v_max *= s                    # [m/s] 3 -> 0.48
        sim.a_max *= s                    # [m/s^2] 5 -> 0.8
        sim.wall_offset *= s              # [m] 30 -> 4.8: the red border is outside the arena
        sim.red_margin *= s               # [m]
        sim.orange_max_agl *= s           # [m] 4.8: above the arena, its ceiling (geofence) limits instead
        sim.sphere_from_circle = False    # Fibonacci sphere, as in Lisbon and in the flight code
        airspace = ArenaAirspace(sim, TRAJ5_N_IN, traj.wall_point)
    else:
        sim.sphere_radius = 0.6           # [m] formation radius
        sim.noise_std = 0.01              # [m] target measurement noise
        sim.mesh_points = 301             # coarser formation mesh (real time at 20 Hz)
        sim.sphere_from_circle = True     # the sites keep their angle around the target through the
                                          # circle -> cone -> sphere morph (else 3 drones get stacked)
        sim.v_max = 0.5                   # [m/s]
        sim.a_max = 1.0                   # [m/s^2]
        sim.wall_offset = RED_BORDER_X - WALL_X   # [m] wall before the red border
        sim.red_margin = 0.15             # [m]
        sim.orange_max_agl = 2.0          # [m] also the ceiling of the arena
        airspace = ArenaAirspace(sim)

    # obstacles: only the floor
    sim.sensing_range = 0.5
    sim.env_clearance = 0.3
    sim.env_brake_gain = 1.0
    sim.env_climb_speed = 0.3

    # Mellinger gains of the Crazyflie firmware (see CRAZYFLIE_ONBOARD); the
    # simulation runs with the CrazyfliePlant (plant.py) for the setpoint
    # delay, z offset and the onboard control rate
    configure_onboard_gains(param)

    param.airspace = airspace
    param.traj = traj
    param.display.trajectory_overview_plot = False   # KMZ map, not meaningful here
    return param
