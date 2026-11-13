"""
Arena scene (SimulationConfig.scene = "arena"): the Lisbon scenario scaled to
the Crazyflie OptiTrack arena (6 x 6 x 3 m room), with the Crazyflie model
instead of the X500. Every trajectory (SimulationConfig.traj) keeps the whole
flight inside the room (GEOFENCE_MARGIN from its walls and ceiling):

  1-4  hover, circle, Lissajous, 3D Lissajous around the arena centre, across
       the drones' wall x = SimulationConfig.arena_wall_x (facing +x, the grey
       band beyond it): the formation morphs wall <-> sphere as the target
       crosses it. The red border (wall_offset past the wall) is outside the room.
  5    a slice of the Lisbon trajectory 5: across the straight wall at the
       Lisbon angle (ArenaTrajectory5).
  6    a slice of the Lisbon trajectory 6: into the corner where the two walls
       of the red border notch meet, at the Lisbon angle (ArenaTrajectory6).

Lengths are the Lisbon ones times ARENA_SCALE (formation radius 5 m -> 0.8 m)
and time is kept where the path allows it.

Used by the Python simulation and by the real flight (Crazy_Flies/), so both
fly the same scenario.
"""
import numpy as np

from restricted_airspace import RestrictedAirspace, Polygon2D

# The room is 6 x 6 x 3 m; the drones only use the central 5 x 5 x 2.5 m.
ARENA_MIN = np.array([-2.5, -2.5, 0.0])
ARENA_MAX = np.array([2.5, 2.5, 2.5])
GEOFENCE_MARGIN = 0.3     # [m] goals, horizons and setpoints kept this far from the walls and the
                          # 2.5 m ceiling; no minimum height (floor at z = 0)

DRONE_MODEL = "cf2x_L250"     # entry of drone_models.toml

FAR = 20.0                # [m] the zones extend far beyond the arena (straight border, no joints nearby)
TARGET_HEIGHT = 1.3       # [m] formation (radius 0.8 m) between ~0.5 and ~2.1 m

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
ARENA_SCALE = TRAJ5_SCALE     # every trajectory of the arena scene
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
TRAJ5_HOLD = 5.0          # [s] simulated after the target stops (trajectories 5 and 6)


class ArenaAirspace(RestrictedAirspace):
    """
    Straight red border with normal n_in (towards the red zone), the drones'
    wall wall_offset before it through wall_point; red zone beyond the
    border, orange zone before it (max altitude orange_max_agl), same
    interface as the KMZ airspace. project() also keeps the points inside the
    arena (GEOFENCE_MARGIN from the walls and the ceiling, above the floor),
    so formation goals and MPC horizons stay in the arena. Default: the wall
    x = 0 facing +x (trajectories 1-4).
    """
    kind = "arena"

    def __init__(self, sim, n_in=(1.0, 0.0), wall_point=(0.0, 0.0)):
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
    """
    Arena trajectories 1-4 (TrajectoryConfig interface), around the arena
    centre: the target stays within ARENA_MAX - GEOFENCE_MARGIN - TRAJ5_RADIUS
    (1.4 m) so the formation sphere around it stays in the geofence, and
    below the scaled MPC speed limit (0.48 m/s). They cross the wall x =
    arena_wall_x when it is inside that range (|x| < 1.2 m).
    """

    def __init__(self, traj):
        h = TARGET_HEIGHT
        if traj == 1:
            # hover in the grey band, 1 m past the centre (Lisbon: 10 m past the wall)
            p = np.array([1.0, 0.0, h])
            pos, vel = (lambda t: p), (lambda t: np.zeros(3))
            duration = 20.0
        elif traj == 2:
            # circle of 1.2 m at 0.36 m/s (Lisbon: 15 m at 3.75 m/s)
            w, r = 0.3, 1.2
            pos = lambda t: np.array([r * np.sin(w * t), r * np.cos(w * t), h])
            vel = lambda t: np.array([r * w * np.cos(w * t), -r * w * np.sin(w * t), 0.0])
            duration = 2 * np.pi / w
        elif traj == 3:
            # Lissajous 1.2 m, max speed 0.36 m/s
            w, A = 0.15, 1.2
            pos = lambda t: np.array([A * np.sin(w * t), A * np.sin(2 * w * t), h])
            vel = lambda t: np.array([A * w * np.cos(w * t), 2 * A * w * np.cos(2 * w * t), 0.0])
            duration = 2 * np.pi / w
        elif traj == 4:
            # 3D Lissajous: as 3, height 1.25 +- 0.15 m (the sphere under the 2.2 m geofence)
            w, A, hz, Az = 0.15, 1.2, 1.25, 0.15
            pos = lambda t: np.array([A * np.sin(w * t), A * np.sin(2 * w * t), hz + Az * np.cos(2 * w * t)])
            vel = lambda t: np.array([A * w * np.cos(w * t), 2 * A * w * np.cos(2 * w * t),
                                      -2 * Az * w * np.sin(2 * w * t)])
            duration = 2 * np.pi / w
        else:
            raise ValueError(f"ArenaTrajectory: trajectories 1-4, not {traj!r}")

        self.ref_target_traj = {"pos": lambda t: pos(t).reshape(3, 1),
                                "vel": lambda t: vel(t).reshape(3, 1)}
        self.initial_target_pos = self.ref_target_traj["pos"](0)
        self.t_stop = duration


def _straight_until_geofence(start, vel):
    """Time at which the formation sphere (TRAJ5_RADIUS) around start + vel t reaches the geofence."""
    lim = ARENA_MAX[:2] - GEOFENCE_MARGIN - TRAJ5_RADIUS
    return min((np.sign(v) * lim[i] - start[i]) / v for i, v in enumerate(vel) if v != 0)


def _straight_path(start, vel, t_stop):
    """pos / vel functions of a straight path at TARGET_HEIGHT that stops at t_stop."""
    def pos(t_):
        xy = start + vel * min(max(t_, 0.0), t_stop)
        return np.array([[xy[0]], [xy[1]], [TARGET_HEIGHT]])

    def vel_fn(t_):
        v = vel if 0.0 <= t_ < t_stop else np.zeros(2)
        return np.array([[v[0]], [v[1]], [0.0]])

    return {"pos": pos, "vel": vel_fn}


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
        t_stop = _straight_until_geofence(start, vel)
        self.ref_target_traj = _straight_path(start, vel, t_stop)
        self.initial_target_pos = self.ref_target_traj["pos"](0)
        self.t_stop = t_stop
        self.wall_point = start + 0.5 * n                # the wall 0.5 m ahead of the start


def _enu_to_arena_matrix():
    b = np.radians(ARENA_X_BEARING)
    return np.array([[np.sin(b), np.cos(b)], [-np.cos(b), np.sin(b)]])   # rows: arena x, y axes in ENU


class ArenaTrajectory6:
    """
    Arena trajectory 6 (TrajectoryConfig interface): a slice of the Lisbon
    trajectory 6 (Structs_sim.lisbon_traj6_geometry), scaled by ARENA_SCALE
    and rotated into the arena frame (ARENA_X_BEARING). The target flies the
    notch bisector through the arena centre, starting on the geofence 0.5 m
    before the corner where the two walls meet, at the scaled Lisbon speed,
    until the formation sphere reaches the geofence. lisbon_airspace: the
    Lisbon RestrictedAirspace (red polygon in ENU, Lisbon wall_offset).
    """

    def __init__(self, lisbon_airspace):
        from Structs_sim import lisbon_traj6_geometry
        g = lisbon_traj6_geometry(lisbon_airspace)
        R = _enu_to_arena_matrix()
        d = R @ g["dir_in"]
        d /= np.linalg.norm(d)
        lim = ARENA_MAX[:2] - GEOFENCE_MARGIN
        s = min(lim[i] / abs(v) for i, v in enumerate(d) if abs(v) > 1e-9)
        start = -s * d                                   # on the geofence, line through the centre
        corner = start + 0.5 * d                         # the wall corner 0.5 m ahead of the start
        vel = g["speed"] * ARENA_SCALE * d
        t_stop = _straight_until_geofence(start, vel)

        self.ref_target_traj = _straight_path(start, vel, t_stop)
        self.initial_target_pos = self.ref_target_traj["pos"](0)
        self.t_stop = t_stop
        self.wall_point = corner
        # Lisbon ENU -> arena: about the wall corner, rotated and scaled
        wc = g["wall_corner"]
        self.red_vertices = corner + ARENA_SCALE * (lisbon_airspace.red.vertices - wc) @ R.T


class ArenaCornerAirspace(RestrictedAirspace):
    """
    Red border polygon of a Lisbon slice in the arena (trajectory 6: the notch
    at red vertex 8), with the general wall frame and projection of
    RestrictedAirspace (joints included, lengths scaled by ARENA_SCALE), no
    orange zone, and the points kept inside the arena geofence.
    """
    kind = "arena"

    def __init__(self, sim, red_vertices):
        self.red = Polygon2D(red_vertices)
        self.orange = []
        self.red_zone_name = sim.red_zone_name
        self.orange_zone_name = sim.orange_zone_name
        self.zones = [(sim.red_zone_name, [self.red.vertices])]
        self.wall_offset = sim.wall_offset
        self.wall_joint_blend = sim.wall_joint_blend
        self.joint_min_scale = RestrictedAirspace.joint_min_scale * ARENA_SCALE
        m = len(self.red.vertices)
        self.adjacent_edges = {frozenset((i, (i + 1) % m)) for i in range(m)}
        self.red_margin = sim.red_margin
        self.orange_max_agl = sim.orange_max_agl

        self.box_min = ARENA_MIN + GEOFENCE_MARGIN
        self.box_min[2] = ARENA_MIN[2]          # no minimum height
        self.box_max = ARENA_MAX - GEOFENCE_MARGIN

    def project(self, P):
        return np.clip(super().project(P), self.box_min, self.box_max)


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
    "mass": 0.032,        # [kg] ctrlMel.mass: feedforward mass of the controller (set by the flight code,
                          # MELLINGER_PARAMS); the model mass (drone_models.toml) stays in the dynamics
    "ki": [0.05, 0.05, 0.05],     # firmware ki_xy, ki_xy, ki_z (integral of the position error; firmware
                                  # defaults, the --IRL_log replay uses the values read back from the drones)
    "i_range": [2.0, 2.0, 0.4],   # firmware i_range_xy, i_range_xy, i_range_z (clamp of that integral)
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
    param.controller.Mellinger["mass"] = onboard["mass"]
    param.controller.Mellinger["ki"] = np.diag(onboard["ki"])
    param.controller.Mellinger["i_range"] = np.array(onboard["i_range"], dtype=float)
    param.controller.Mellinger["kR"] = J * wn ** 2
    param.controller.Mellinger["kom"] = J * 2 * wn
    param.controller.max_tilt_deg = onboard["max_tilt_deg"]


def configure_arena_experiment(param, dTi=None):
    """
    Turn param (Structs_sim.Param) into the arena scene: Crazyflie model and
    controller gains, the Lisbon parameters scaled by ARENA_SCALE, arena
    airspace and target path of param.sim.traj (1-6, see the module
    docstring). dTi: control period (None keeps param.sim.dTi; the real
    flight uses its own control period).
    """
    sim = param.sim
    traj_id = sim.traj
    lisbon = None
    if traj_id == 6:
        # the Lisbon red polygon and wall, before the parameters are scaled
        lisbon = RestrictedAirspace(sim)
    sim.load_drone_model(DRONE_MODEL)

    if traj_id in (1, 2, 3, 4):
        traj, hold = ArenaTrajectory(traj_id), 0.0
    elif traj_id == 5:
        traj, hold = ArenaTrajectory5(), TRAJ5_HOLD
    elif traj_id == 6:
        traj, hold = ArenaTrajectory6(lisbon), TRAJ5_HOLD
    else:
        raise ValueError(f"Unknown arena trajectory {traj_id!r} (1-6)")

    if dTi is not None:
        sim.dTi = dTi
    sim.Tend = traj.t_stop + hold
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

    # Lisbon values (SimulationConfig defaults) scaled, same time
    s = ARENA_SCALE
    sim.sphere_radius = TRAJ5_RADIUS
    sim.v_max *= s                        # [m/s] 3 -> 0.48
    sim.a_max *= s                        # [m/s^2] 5 -> 0.8
    sim.wall_offset *= s                  # [m] 30 -> 4.8: the red border is outside the arena
    sim.red_margin *= s                   # [m]
    sim.orange_max_agl *= s               # [m] 4.8: above the arena, its ceiling (geofence) limits instead
    sim.sphere_from_circle = False        # Fibonacci sphere, as in Lisbon and in the flight code

    if traj_id == 5:
        airspace = ArenaAirspace(sim, TRAJ5_N_IN, traj.wall_point)
    elif traj_id == 6:
        airspace = ArenaCornerAirspace(sim, traj.red_vertices)
    else:
        airspace = ArenaAirspace(sim, (1.0, 0.0), (sim.arena_wall_x, 0.0))

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
