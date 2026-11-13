import numpy as np
from numpy.typing import NDArray
import math
import os
import tomllib
from dataclasses import dataclass

from restricted_airspace import RestrictedAirspace
from environment import FlatGround

# CONSTS
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

class SimulationConfig:
    def __init__(self, config_path=None, drone_name="hb_x500"):
        # Simulation
        self.Tend = 30 # [seconds]
        self.dTi = 0.01 # [seconds]
        self.Nsim = round(self.Tend / self.dTi) + 1

        self.t = np.arange(
            0,
            self.Tend + self.dTi,
            self.dTi
        )

        self.gravity = 9.81

        # Drones
        self.nD = 5
        self.sphere_radius = 0.8

        # Drop parameters
        self.nD_drop = 2
        self.nD_drop_time = -1 # -1 to deactivate

        # Quadrotor: fitted so_rpy_rotor_drag parameters from drone_models.toml
        self.load_drone_model(drone_name, config_path)

        # Disturbance
        self.x_delta = np.array([
            0.5,
            0.8,
            -1.0
        ])

        self.R_delta = np.array([
            0.2,
            1.0,
            -0.1
        ])

        # Target estimation
        self.ema_alpha = 0.2
        self.noise_std = 0.05

        # Target path: 1 hover, 2 circle, 3 Lissajous, 4 3D Lissajous, 5 intrusion across a
        # straight red border, 6 intrusion through a red border joint (in both scenes)
        self.traj = 2

        # MPC: plan a 1 s horizon to the goal (ADMM, with vel/acc limits,
        # inter-agent avoidance and the airspace). False: the goal position
        # (and goal velocity) is passed directly to the controller.
        self.use_mpc = True
        # Drones that track their setpoints tightly (the Crazyflies, set by
        # the arena experiment): MPC started from the previous plan, ADMM
        # multipliers reset every step, setpoint one control period ahead
        # (as the flight code; see AdmmMPC.solve and mpc_setpoint)
        self.mpc_tight_tracking = False
        self.replan_reset_dist = 0.2    # [m] farther from its plan, a drone is replanned from its measured state
        # ADMM multipliers carried over between control steps (warm start).
        # None: True, except with mpc_tight_tracking (reset every step, as the flight code)
        self.admm_warm_multipliers = None

        # constraints
        self.obstacle_avoidance = True 
        self.restricted_vel_acc = True
        self.obstacle_avoidance_dim = 0.20
        # Safety envelope around the other drones (and the target drone): a
        # sphere of radius obstacle_avoidance_dim (thesis, inter-agent separation)
        self.n_degree = 5
        self.horizon = 5           # horizon
        self.rho_obs = 100.0        # Obstacle avoidance penalty
        self.rho_ineq = 20.0        # max/min vel and acc penalty
        self.max_iter = 100         # max iterations per update
        self.w_goal = 1000.0        # gain to reduce dist between agent and goal
        self.w_smooth = 0.1         # gain to reduce high accelerations
        # Exponential goal reference over the horizon (1 s): the horizon is
        # pulled towards  goal + (p_now - goal) * exp(-t / goal_ref_tau)
        # instead of straight to the goal (0 = previous behaviour).
        self.goal_ref_tau = 0.5     # [s]
        # Desired velocity of the horizon: goal velocity (from the target
        # velocity) + the exponential approach (0 = no velocity objective)
        self.w_vel = 300.0          # gain to track the desired velocity
        self.v_max = 3             # max velocity constraint 
        self.a_max = 5             # max acceleration constraint 

        # --- CVT parameters ---
        # Number of simulation iterations between CVT site updates.
        # The underlying mesh points are still updated every iteration.
        # Only the CVT sites (goal positions) are recomputed at this interval
        # using the current mesh.
        self.update_CVT_interval = 100
        self.mesh_points = 1001     # vertices of the formation mesh (CVT cost grows with it)
        self.converge_initial_formation = False   # CVT on the initial mesh before the start (drones on spread slots)
        self.weights_sigma = 1.2
        
        # --- Measurement model ---
        # options: "estimated" or "ground_truth"
        self.measurement_model = "estimated"
        # 1: position
        # 2: position + velocity
        self.measurement_dimension = 2

        # Formation/distribution
        # options: "fibonacci" or "dijkstra" 
        self.distribution = "dijkstra"

        # --- Flight scene ---
        #   "lisbon": the Lisbon airport (KMZ zones, ENU frame at the LPPT ARP, same as Gazebo),
        #             X500. Trajectories 1-4 fly in a frame on the drones' wall at the
        #             trajectory 5 crossing (x into the grey band, red border at x = wall_offset);
        #             5 and 6 are the intrusions into the red zone.
        #   "arena":  the Crazyflie OptiTrack arena (experiment_arena.py): Crazyflie model, the
        #             Lisbon scenario scaled to the 6 x 6 x 3 m room (geofence). Trajectories 1-4
        #             cross the wall x = arena_wall_x (wall <-> sphere); 5 and 6 are a slice of
        #             the Lisbon trajectories 5 and 6 (the wall and its corner at the Lisbon angle).
        self.scene = "lisbon"
        self.arena_wall_x = 0.0         # [m] arena, trajectories 1-4: the wall (start of the grey band)
        self.arena_target_drone = True  # arena: the target is a Crazyflie (as Crazy_Flies/swarm_MPC_voronoi),
                                        # avoided by the chasers' MPC; False: virtual target (swarm_MPC_voronoi_sim_target)
        self.airspace_kmz = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Assets", "Aeroporto de Lisboa.kmz"
        )
        self.red_zone_name = "Área Proibida"     # formation drones may not enter
        self.orange_zone_name = "Área 1"         # altitude limited
        self.wall_offset = 30.0     # [m] wall distance outside the red border
        self.wall_joint_blend = 0.5  # relative width over which the wall frame turns at red border joints
        self.red_margin = 1.0       # [m] min horizontal clearance to the red border
        self.orange_max_agl = 30.0  # [m] max altitude above ground in the orange zone

        # --- Environment collision avoidance (chasers and target) ---
        self.sensing_range = 5.0    # [m] drones detect obstacles closer than this
        self.env_clearance = 1.5    # [m] min distance kept from detected obstacles
        self.env_brake_gain = 1.0   # [1/s] max approach speed = gain * (distance - clearance)
        self.env_climb_speed = 2.0  # [m/s] climb speed of the target over obstacles

    def load_drone_model(self, drone_name, config_path=None):
        """Mass, inertia, thrust limits and drag of `drone_name` (drone_models.toml)."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "drone_models.toml"
            )
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        drone = config[drone_name]
        self.drone_name = drone_name

        self.drone_mass = drone["mass"]
        self.J = np.array(drone["J"], dtype=float)

        self.thrust_min = drone["thrust_min"]   # [N] per motor
        self.thrust_max = drone["thrust_max"]   # [N] per motor

        self.acc_coef = drone["acc_coef"]
        self.thrust_dyn_coef = drone["thrust_dyn_coef"]

        self.drag_matrix = np.array(
            drone["drag_matrix"], dtype=float
        )

        self.rpy_coef = np.array(
            drone["rpy_coef"], dtype=float
        )

        self.rpy_rates_coef = np.array(
            drone["rpy_rates_coef"], dtype=float
        )

class ControllerConfig:
    def __init__(self):
        # Options: Mellinger
        self.selected_controller = "Mellinger"
        self.max_tilt_deg = None    # Mellinger tilt limit [deg] (None = no limit)

        self.Mellinger = {
            "kp": np.diag([1, 1, 1]),
            "kv": np.diag([15, 15, 15]),
            "kR": np.diag([8, 8, 8]),
            "kom": np.diag([0.5, 0.5, 0.5]),
        }

        # Consensus parameters
        self.c1 = 1.5
        self.sigma = 10
        self.c2 = 2
        self.c3 = 2

def lisbon_local_traj(traj):
    """Trajectories 1-4 of the Lisbon scene in the wall frame (x past the wall, y along it, z up), (3,) arrays."""
    if traj == 1:
        # Hover in the grey band, 10 m past the wall
        return {"pos": lambda t: np.array([10.0, 0.0, 5.0]),
                "vel": lambda t: np.zeros(3)}
    # Speeds: the target stays well under the MPC limit v_max (3 m/s), with room
    # for the chasers on the outer side of the formation (radius 5 m), which fly
    # faster than the target in the turns. At V16's 0.25 rad/s (3.75 m/s) the
    # chasers were held at v_max 25% of the time, 2 m behind their goals on average.
    if traj == 2:
        # Circle of 15 m around (-5, 0, 5): crosses the wall back and forth;
        # 0.12 rad/s: target 1.8 m/s, outer chasers ~2.4 m/s
        w, r = 0.12, 15.0
        return {"pos": lambda t: np.array([-5 + r * np.sin(w * t), r * np.cos(w * t), 5.0]),
                "vel": lambda t: np.array([r * w * np.cos(w * t), -r * w * np.sin(w * t), 0.0])}
    if traj == 3:
        # Lissajous centred on the wall; 0.2 rad/s: target up to 2.2 m/s (was 1 rad/s, 11 m/s)
        w = 0.2
        return {"pos": lambda t: np.array([5 * np.sin(w * t), 5 * np.sin(2 * w * t), 5.0]),
                "vel": lambda t: np.array([5 * w * np.cos(w * t), 10 * w * np.cos(2 * w * t), 0.0])}
    if traj == 4:
        # 3D Lissajous centred on the wall, as 3 with 0.2 m in height
        w = 0.2
        return {"pos": lambda t: np.array([5 * np.sin(w * t), 5 * np.sin(2 * w * t), 5 + 0.2 * np.cos(2 * w * t)]),
                "vel": lambda t: np.array([5 * w * np.cos(w * t), 10 * w * np.cos(2 * w * t),
                                           -0.4 * w * np.sin(2 * w * t)])}
    raise ValueError(f"Unknown trajectory {traj}")


def lisbon_traj6_geometry(airspace):
    """
    Trajectory 6 of the Lisbon scene: along the bisector of the notch at red
    vertex 8, from 30 m before the wall corner to 30 m inside the red zone.
    Returns start (2,), vel (2,), t_stop, the wall corner (2,) and the inward direction (2,).
    """
    V = airspace.red.vertices
    joint = V[8]
    u1 = (V[7] - joint) / np.linalg.norm(V[7] - joint)
    u2 = (V[9] - joint) / np.linalg.norm(V[9] - joint)
    bisector_out = (u1 + u2) / np.linalg.norm(u1 + u2)
    half_angle = 0.5 * np.arccos(np.clip(u1 @ u2, -1.0, 1.0))
    wall_corner = airspace.wall_offset / np.sin(half_angle)     # distance joint -> wall corner
    speed = 2.2                                                  # [m/s]
    return {
        "start": joint + (wall_corner + 30.0) * bisector_out,
        "vel": -speed * bisector_out,
        "speed": speed,
        "t_stop": (wall_corner + 60.0) / speed,
        "wall_corner": joint + wall_corner * bisector_out,
        "dir_in": -bisector_out,
    }


class TrajectoryConfig:
    def __init__(
        self,
        airspace: RestrictedAirspace,
        traj=5,
        radius=2.2,
        speed=0.5,
        height=1.3
    ):

        self.target_traj_radius = radius
        self.default_speed = speed
        self.default_height = height
        self.airspace = airspace

        self.ref_target_traj = self.make_traj(traj)

        self.initial_target_pos = (
            self.ref_target_traj["pos"](0)
        )


    def wall_frame_origin(self):
        """
        Frame of trajectories 1-4: origin on the drones' wall at the trajectory 5
        crossing (40% along red edge 1, a straight 6 km edge over open ground),
        x into the red zone (normal of the wall), y along the wall, z up. The
        grey band is 0 < x < wall_offset; the red border is at x = wall_offset.
        """
        crossing = 0.6 * self.airspace.red.vertices[1] + 0.4 * self.airspace.red.vertices[2]
        n_out, c, _ = self.airspace.red.outward_normal(crossing.reshape(1, 2))
        n_in = -n_out[0]
        tangent = np.array([-n_in[1], n_in[0]])
        origin = c[0] - self.airspace.wall_offset * n_in
        return origin, n_in, tangent

    def make_traj(self, traj):
        if traj in (1, 2, 3, 4):
            # Paths of V16 in the wall frame (wall at x = 0, see wall_frame_origin):
            # the target crosses the wall into the grey band and back, the
            # formation morphs wall <-> sphere; the red zone is never reached.
            local = lisbon_local_traj(traj)
            origin, n_in, tangent = self.wall_frame_origin()
            R = np.array([[n_in[0], tangent[0], 0.0],
                          [n_in[1], tangent[1], 0.0],
                          [0.0, 0.0, 1.0]])
            o = np.array([origin[0], origin[1], 0.0])
            return {
                "pos": lambda t: (o + R @ local["pos"](t)).reshape(3, 1),
                "vel": lambda t: (R @ local["vel"](t)).reshape(3, 1),
            }

        elif traj == 5:
            # Intrusion into the red zone: starts in the orange zone 50 m
            # outside the red border (30 m before the wall), crosses the wall
            # and the red border, and hovers 40 m inside the red zone.
            # Crossing point: 40% along red edge 1 (vertices 1 -> 2), over open
            # flat ground, so the path and the chasers' wall stay clear of
            # buildings in the Gazebo city (no path planning around them yet).
            crossing = 0.6 * self.airspace.red.vertices[1] + 0.4 * self.airspace.red.vertices[2]
            n_out, c, _ = self.airspace.red.outward_normal(crossing.reshape(1, 2))
            n_in = -n_out[0]
            tangent = np.array([-n_in[1], n_in[0]])
            start = c[0] - 50.0 * n_in
            height = 15.0
            vel = 1.5 * n_in + 0.5 * tangent        # [m/s]
            t_stop = 90.0 / 1.5                     # stops 40 m inside the red zone

            def pos(t):
                xy = start + vel * min(t, t_stop)
                return np.array([[xy[0]], [xy[1]], [height]])

            def vel_fn(t):
                v = vel if t < t_stop else np.zeros(2)
                return np.array([[v[0]], [v[1]], [0.0]])

            return {"pos": pos, "vel": vel_fn}

        elif traj == 6:
            # Intrusion through a joint between two red border edges: the
            # target flies along the bisector of the sharp (34.6 deg) notch
            # at red vertex 8, where the two walls meet. Starts 30 m before
            # the wall corner and stops 30 m inside the red zone.
            g = lisbon_traj6_geometry(self.airspace)
            start, vel, t_stop, height = g["start"], g["vel"], g["t_stop"], 15.0

            def pos(t):
                xy = start + vel * min(t, t_stop)
                return np.array([[xy[0]], [xy[1]], [height]])

            def vel_fn(t):
                v = vel if t < t_stop else np.zeros(2)
                return np.array([[v[0]], [v[1]], [0.0]])

            return {"pos": pos, "vel": vel_fn}

        else:
            raise ValueError(
                f"Unknown trajectory {traj}"
            )

PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_FILE = "Results/python/simulation_results.pkl"   # relative to the Python folder

def results_file_path(display, latest=False):
    """
    Path of the results .pkl of display.results_file (relative to the Python
    folder, or absolute). None: the newest .pkl of this simulation if latest
    (loading: the Results/ folder of the Python folder, and the runs of the
    repository Results/ folder, swarm_MPC_voronoi_*, e.g. the replays of real
    flights), else DEFAULT_RESULTS_FILE (saving).
    """
    if display.results_file is not None:
        return os.path.join(PYTHON_DIR, display.results_file)
    if latest:
        from pathlib import Path
        local = Path(PYTHON_DIR, "Results")
        repo = next(p for p in Path(PYTHON_DIR).parents if (p / ".git").exists()) / "Results"
        pkls = [*local.rglob("*.pkl")]
        for run in repo.glob("swarm_MPC_voronoi*"):
            pkls += run.rglob("*.pkl")
        if not pkls:
            raise FileNotFoundError(f"No .pkl in {local} or {repo}/swarm_MPC_voronoi*")
        return str(max(pkls, key=lambda p: p.stat().st_mtime))
    return os.path.join(PYTHON_DIR, DEFAULT_RESULTS_FILE)


# SimulationConfig values stored with the results (.pkl, "sim"): what the
# figures need to be redrawn (animation_display.py) for the run's own scene
SAVED_SIM_KEYS = ("scene", "traj", "nD", "Tend", "dTi", "Nsim", "nD_drop_time", "sphere_radius", "v_max", "a_max",
                  "obstacle_avoidance_dim", "arena_wall_x")


def sim_config(param):
    """The SimulationConfig values of SAVED_SIM_KEYS (stored in the .pkl)."""
    return {k: getattr(param.sim, k) for k in SAVED_SIM_KEYS if hasattr(param.sim, k)}


def param_from_config(cfg):
    """Param of a stored run (sim_config): its scene and trajectory, then its stored values."""
    sim = SimulationConfig()
    sim.scene, sim.traj = cfg.get("scene", sim.scene), cfg.get("traj", sim.traj)
    if "arena_wall_x" in cfg:
        sim.arena_wall_x = cfg["arena_wall_x"]
    param = Param(sim)
    for k, v in cfg.items():
        setattr(param.sim, k, v)
    param.sim.t = np.arange(param.sim.Nsim) * param.sim.dTi
    return param


def simulation_data(d):
    """The SimulationData of a results dict (.pkl), without its extra entries (sim, phases, ...)."""
    from dataclasses import fields
    return SimulationData(**{f.name: d[f.name] for f in fields(SimulationData)})


class GraphicalDisplay:
    def __init__(self):
        self.verbose = True

        self.composed = False

        self.detailed_close_up_bool = False

        self.radial_hist_plot =  False
        self.radial_hist_animation = False

        self.inter_agent_dist  =  False

        self.low_level_tracking_error =  False

        self.consensus_plot =  False 
        self.consensus_animation = False 

        self.velocity_plot =  False
        self.velocity_animate = False 

        self.acceleration_plot =  False
        self.acceleration_animate = False

        self.trajectory_plot = False
        self.trajectory_overview_plot = True
        self.trajectory_animate = True

        self.three_view_plot = False

        self.store_data = True
        self.save_to_mp4 = False
        self.mp4_fps = 30

        self.save_location = "/Results/python"          # figures, relative to the Python folder
        # .pkl of the run, relative to the Python folder (or absolute). None: saved to
        # DEFAULT_RESULTS_FILE; animation_display.py (redraw) loads the newest .pkl in Results/
        self.results_file = DEFAULT_RESULTS_FILE
        self.trail_length = 100
        self.step = 5

@dataclass
class SimulationData:
    target_true_hist: list
    removed_initial_pos: bool
    colors: np.ndarray
    active_ids: list
    numAgents_history: list

    position_history: NDArray[np.float64]
    velocity_history: NDArray[np.float64]
    acceleration_history: NDArray[np.float64]
    perceived_target_hist: NDArray[np.float64]
    p_d_hist: NDArray[np.float64]
    radial_history: NDArray[np.float64]

class GazeboConfig:
    """Gazebo plant options (Gazebo/sim/run_gazebo_sim.py), unused by the Python simulation."""
    def __init__(self):
        # "live":   the Gazebo GUI opens and follows the simulation while it
        #           runs (physics at live_rtf, slow with many drones)
        # "replay": the simulation runs headless as fast as possible and is
        #           recorded; the GUI then plays it back in real time (with
        #           play/pause/seek), next to the result figures
        self.mode = "replay"
        self.live_rtf = 1.0             # real-time factor in live mode (0 = as fast as possible)
        self.record_period = 0.04       # [s] state recording period of the replay log
        self.log_dir = "Results/Gazebo/log"     # replay log, relative to the Python folder
        self.show_area1 = False         # show the Área 1 (orange) zone, hidden by default

class Param:
    def __init__(self, sim: SimulationConfig | None = None):
        self.sim = SimulationConfig() if sim is None else sim

        self.controller = ControllerConfig()

        # obstacles the drones can sense (flat ground here; the 3D city in Gazebo)
        self.environment = FlatGround()

        self.display = GraphicalDisplay()

        self.gazebo = GazeboConfig()

        if self.sim.traj not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"Unknown trajectory {self.sim.traj!r} (1-6)")
        if self.sim.scene == "arena":
            # Crazyflie arena: own airspace, path, model and scale
            from experiment_arena import configure_arena_experiment
            configure_arena_experiment(self)
        elif self.sim.scene == "lisbon":
            self.airspace = RestrictedAirspace(self.sim)
            self.traj = TrajectoryConfig(self.airspace, traj=self.sim.traj)
        else:
            raise ValueError(f"Unknown scene {self.sim.scene!r} (lisbon, arena)")

class Voronoi:
    def __init__(self, param: Param):
        self.mesh_pos = np.array(None)
        self.airspace: RestrictedAirspace = param.airspace
        self.v_circle = np.array(None)
        self.v_cone = np.array(None)
        self.v_sphere = np.array(None)
        self.vertices = np.empty((0, 3))
        self.faces = np.empty((0, 3), dtype=int)
        self.sites: NDArray[np.int_] | None = None
        self.newSites: NDArray[np.int_] | None = None
        self.ran_u2_eq_1 = False
        self.weights = np.array(None)

class MeshVertex:
    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = np.array(pos, dtype=float)

        self.dist = np.inf
        self.origin_site = -1
        self.origin_site_weight = -1
        
        self.neigh = []
        self.neighW = []