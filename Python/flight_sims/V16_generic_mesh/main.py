import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.linalg import lu_factor, lu_solve

from Structs_sim import Param, Voronoi, SimulationData
from Structs_agent import DroneState
from voronoi_mesh_funcs import pre_compute_desired_pos
from delaunay_triangulation_network import delaunay_triangulation_network
from update_target_measurement import update_target_measurement
from compute_desired_goal import compute_desired_goal
from actuate_control import actuate_control
from drone_3dfull_dyn import drone_3dfull_dyn
from animation_display import create_figures
from helper_funcs import *
from obstacle_avoidance_funcs import *

# =============================================================================
# Simulation initialization
# =============================================================================
param = Param()
voronoi = Voronoi(param)

origin_p_d, swarm_center, voronoi = pre_compute_desired_pos(voronoi, param)

# =============================================================================
# Chaser initialization
# =============================================================================
chaser_list: list[DroneState] = []
for i in range(param.sim.nD):
    drone = DroneState(param)
    drone.position = origin_p_d[:, i:i+1] + swarm_center
    drone.desired.goal = drone.position
    chaser_list.append(drone)

obs_vars = init_obstacle_avoidance(chaser_list, param)
Q = obs_vars["Q"]
A_eq = obs_vars["A_eq"]
A_obs_blk = obs_vars["A_obs_blk"]
G = obs_vars["G"]
h = obs_vars["h"]
W0 = obs_vars["W0"]
W1 = obs_vars["W1"]
W2 = obs_vars["W2"]
n_ctrl = obs_vars["n_ctrl"]
dim = obs_vars["dim"]
x = np.zeros(dim)
goal_vec = W0.T @ np.ones((param.sim.horizon,1))
H = param.sim.horizon

# A_obs initialized for a unconstrained motion
for iD, agent in enumerate(chaser_list):
    # update goal
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        agent.constraints.q[idx] = -param.sim.w_goal * agent.desired.goal[axis,0] * goal_vec
    agent.constraints.s = np.zeros(G.shape[0]).reshape(-1,1)
    agent.constraints.lambda_obs = np.zeros([3*param.sim.horizon,1])
    agent.constraints.lambda_ineq = np.zeros([G.shape[0],1])
    agent.constraints.A_obs = A_obs_blk
    # get 1st trajectory
    solve_unconstrained_motion(obs_vars, agent, param)

# =============================================================================
# MPC / ADMM precomputation
#
# A_obs of every agent is just A_obs_blk tiled once per obstacle, so the
# matrix  Q + rho_obs*A_obs^T A_obs + rho_ineq*G^T G  (and hence the KKT
# matrix) is CONSTANT as long as the number of obstacles does not change.
# We factorize it once (and again only when drones are dropped) and only
# the right-hand side is rebuilt in the loop.
# =============================================================================
GT = G.T.copy()
GTG = GT @ G
obstacle_radius_vec = np.diag(chaser_list[0].constraints.obstacle_radius).copy()  # (3,)
mpc_iters = getattr(param.sim, "mpc_iters", 15)   # ADMM iterations per step

# horizon (position) of every agent, shape (nD, H, 3)
horizons = np.zeros((param.sim.nD, H, 3))
for iD, agent in enumerate(chaser_list):
    horizons[iD] = W0 @ agent.constraints.C

mpc_cache = {}   # filled by build_mpc_cache


def build_b_obs(traj, others, r_vec):
    """
    Vectorized obstacle projection points.

    traj   : (H, 3)    own horizon
    others : (n, H, 3) horizons of the other agents
    returns: (n*3*H, 1), per obstacle laid out as [x(H), y(H), z(H)]
    """
    delta = (traj[None, :, :] - others) / r_vec          # (n, H, 3)
    d = np.linalg.norm(delta, axis=2, keepdims=True)     # (n, H, 1)
    safe = d > 1e-9
    direction = np.where(safe, delta / np.where(safe, d, 1.0), 0.0)
    # if the two agents coincide, push along +x to avoid a zero direction
    direction[~safe[..., 0], :] = np.array([1.0, 0.0, 0.0])
    b = others + r_vec * np.maximum(d, 1.0) * direction  # (n, H, 3)
    return b.transpose(0, 2, 1).reshape(-1, 1)


def build_mpc_cache(active_ids):
    """
    (Re)build everything that depends on the number of obstacles:
    tiled A_obs, LU factorization of the KKT matrix, neighbour index lists,
    and reset of b_obs / lambda_obs of every active agent.
    """
    n_obs = len(active_ids) - 1
    if n_obs > 0:
        A_full = np.tile(A_obs_blk, (n_obs, 1))
    else:
        A_full = np.zeros((0, dim))
    A_full_T = A_full.T.copy()

    A_bar = (
        Q
        + param.sim.rho_obs * (A_full_T @ A_full)
        + param.sim.rho_ineq * GTG
    )
    KKT = np.block([
        [A_bar, A_eq.T],
        [A_eq, np.zeros((A_eq.shape[0], A_eq.shape[0]))]
    ])

    mpc_cache["A_full"] = A_full
    mpc_cache["A_full_T"] = A_full_T
    mpc_cache["lu"] = lu_factor(KKT)
    mpc_cache["others"] = {
        int(i): np.array([int(j) for j in active_ids if j != i], dtype=int)
        for i in active_ids
    }

    for i in active_ids:
        c = chaser_list[i].constraints
        c.A_obs = A_full
        others = horizons[mpc_cache["others"][int(i)]]
        if n_obs > 0:
            c.b_obs = build_b_obs(horizons[i], others, obstacle_radius_vec)
        else:
            c.b_obs = np.zeros((0, 1))
        c.lambda_obs = np.zeros((A_full.shape[0], 1))


target_true_hist = np.zeros([24, param.sim.Nsim])
estimated_target_variables = np.zeros((3, param.sim.nD))

numAgents_history = np.ones(param.sim.Nsim) * param.sim.nD
# =============================================================================
# Target initialization
# =============================================================================
target = DroneState(param)
target.position = param.traj.initial_target_pos.copy()

# =============================================================================
# Simulation variables initialization
# =============================================================================
ema_estimated = np.tile(
    np.concatenate([target.position.reshape(3, 1), target.velocity.reshape(3, 1)]),
    param.sim.nD)
active_mask = np.ones(param.sim.nD, dtype=bool)
active_ids = np.where(active_mask)[0]
inactive_ids = np.where(~active_mask)[0]

if param.sim.use_mpc:
    build_mpc_cache(active_ids)

# =============================================================================
# Logs initialization
# =============================================================================
position_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
velocity_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
acceleration_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
perceived_target_hist = np.zeros([param.sim.nD, 6, param.sim.Nsim])
p_d_hist = np.zeros((param.sim.nD, 3, param.sim.Nsim))
radial_hist = np.zeros((param.sim.nD, param.sim.Nsim))

# =============================================================================
# Main simulation loop
# =============================================================================
for k in range(param.sim.Nsim):
    if k % 100 == 0:
        print(f"Computing simulation iteration: {k:04d}")
    current_time = param.sim.dTi*k
    # ---------------------------------------------------------
    # Drop drones (power loss simulation)
    # ---------------------------------------------------------
    if k == param.sim.nD_drop_time:
        remove_local = np.random.choice(
            len(active_ids),
            round(param.sim.nD_drop),
            replace=False
        )

        remove_idx = active_ids[remove_local]

        active_mask[remove_idx] = False
        active_ids = np.where(active_mask)[0]
        if len(active_ids) == 0:
            raise(ValueError("No drones left all drones have been destroyed")) 
        inactive_ids = np.where(~active_mask)[0]

        ema_estimated = np.delete(
            ema_estimated,
            remove_idx,
            axis=1
        )

        for idx in remove_idx:
            agent:DroneState = chaser_list[idx]
            agent.desired.goal = np.nan

        # number of obstacles changed -> new tiled A_obs and factorization.
        # Dropped drones are no longer treated as obstacles.
        if param.sim.use_mpc:
            build_mpc_cache(active_ids)


    # ---------------------------------------------------------
    # Move target
    # ---------------------------------------------------------
    target.desired.position = param.traj.ref_target_traj["pos"](current_time+param.sim.dTi)
    target.desired.velocity = param.traj.ref_target_traj["vel"](current_time+param.sim.dTi)
    target = actuate_control(target, param)

    # ---------------------------------------------------------
    # Network
    # ---------------------------------------------------------
    current_positions = np.array(
        [chaser_list[idx].position for idx in active_ids]
    ).squeeze(-1)
    connections, laplacian, _ = delaunay_triangulation_network(
        current_positions
    )

    # ---------------------------------------------------------
    # Inactive drones
    # ---------------------------------------------------------
    for iD in inactive_ids:
        pos = chaser_list[iD].position

        if pos[2] <= 0:
            chaser_list[iD].position = np.array(
                [pos[0,0], pos[1,0], 0]
            ).reshape(-1,1)

            chaser_list[iD].velocity = np.array([0,0,0]).reshape(-1,1)

        else:
            vel = chaser_list[iD].velocity
            attitude = chaser_list[iD].attitude.reshape(3, 3)
            om = chaser_list[iD].ang_vel

            dot_p, dot_v, dot_om = drone_3dfull_dyn(
                vel,
                attitude,
                om,
                0,
                np.zeros([3,1]),
                param
            )

            chaser_list[iD].position = pos + param.sim.dTi * dot_p
            chaser_list[iD].velocity = vel + param.sim.dTi * dot_v

            Rp = rot_integrate(
                attitude,
                om,
                param.sim.dTi
            )

            chaser_list[iD].attitude = Rp.reshape(-1)
            chaser_list[iD].ang_vel = om + param.sim.dTi * dot_om

    # ---------------------------------------------------------
    # Active drones
    # ---------------------------------------------------------
    for idx, iD in enumerate(active_ids):
        agent = chaser_list[iD]

        ema_estimated = (
            update_target_measurement(
                target,
                agent,
                ema_estimated,
                idx,
                laplacian,
                param
            )
        )

        neighbours = active_ids[connections[idx]]
        agent, voronoi = compute_desired_goal(
            k,
            iD,
            origin_p_d,
            voronoi,
            agent,
            neighbours,
            param,
        )


    # =====================================================
    # MPC (only active agents)
    # =====================================================
    if param.sim.use_mpc:
        A_full = mpc_cache["A_full"]
        A_full_T = mpc_cache["A_full_T"]
        KKT_lu = mpc_cache["lu"]
        others_idx = mpc_cache["others"]
        has_obstacles = A_full.shape[0] > 0

        active_agents = list(active_ids)
        for _ in range(mpc_iters):
            for iD in active_agents[:]:
                agent = chaser_list[iD]
                c = agent.constraints
                # =====================================================
                # Equality constraint becomes current position
                # =====================================================
                b_eq = np.vstack([agent.position, agent.velocity, agent.acceleration]).reshape(-1,1)

                # Step 1 - solve trajectory (only the RHS changes, KKT is factorized)
                b_bar = (
                    - c.q
                    - A_full_T @ c.lambda_obs
                    + param.sim.rho_obs * A_full_T @ c.b_obs
                    - GT @ c.lambda_ineq
                    + param.sim.rho_ineq * GT @ (h - c.s)
                )

                solution = lu_solve(KKT_lu, np.concatenate([b_bar, b_eq]))
                x = solution[:dim]                       # (dim, 1)
                c.C = x.reshape(n_ctrl, 3)

                # Step 2 - Update b_obs (vectorized over obstacles and horizon)
                horizons[iD] = W0 @ c.C
                if has_obstacles:
                    c.b_obs = build_b_obs(
                        horizons[iD],
                        horizons[others_idx[int(iD)]],
                        obstacle_radius_vec,
                    )

                # Step 3 - Update slack term
                c.s = np.maximum(0, h - G @ x - c.lambda_ineq/param.sim.rho_ineq)

                # calculate residuals
                residual_eq   = A_eq @ x - b_eq
                residual_ineq = G @ x - h + c.s
                residual_obs  = A_full @ x - c.b_obs

                # Step 4 - Update lambdas (kept across iterations AND time steps)
                c.lambda_ineq += param.sim.rho_ineq * residual_ineq
                c.lambda_obs  += param.sim.rho_obs * residual_obs

                eq_norm = np.linalg.norm(residual_eq)
                ineq_norm = np.linalg.norm(residual_ineq)
                obs_norm = np.linalg.norm(residual_obs)

                if (
                    eq_norm + ineq_norm + obs_norm < 1e-2
                ):
                    active_agents.remove(iD)

    for iD, agent in enumerate([chaser_list[i] for i in active_ids]):
        if param.sim.use_mpc:
            # Pass desired values to agent
            C = agent.constraints.C
            traj = W0 @ C
            vel = W1 @ C
            acc = W2 @ C

            # first value of the horizon
            agent.desired.acceleration = acc[1].reshape(-1,1)
            agent.desired.velocity = vel[1].reshape(-1,1)
            agent.desired.position = traj[1].reshape(-1,1)
        else:
            # track the formation goal directly (desired.velocity: the
            # target velocity, set by compute_desired_goal)
            agent.desired.position = agent.desired.goal
            agent.desired.acceleration = np.zeros((3, 1))

        # Distance to target for logging purposes
        q_s = agent.position - target.position
        agent.radii_to_target = np.linalg.norm(q_s)

        # update desired position goal
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            agent.constraints.q[idx] = -param.sim.w_goal * agent.desired.goal[axis] * goal_vec

        # actuate control
        agent = actuate_control(agent, param)

    if np.isnan(origin_p_d[:, active_ids]).any():
        raise RuntimeError("A position value is NaN!")

    voronoi.sites = voronoi.newSites

    #logs values
    for iD, agent in enumerate(chaser_list):
        position_history[iD,:, k:k+1] = agent.position
        velocity_history[iD,:, k:k+1] = agent.velocity
        acceleration_history[iD,:, k:k+1] = agent.acceleration
        perceived_target_hist[iD,:,k:k+1] = np.concatenate(
            (agent.estimated.target_position, 
             agent.estimated.target_velocity
            ), axis=0
        )
        p_d_hist[iD,:,k:k+1] = agent.desired.goal
        radial_hist[iD, k] = agent.radii_to_target
    target_true_hist[:, k:k+1] = np.concatenate([
        target.position,
        target.velocity,
        target.attitude,
        target.ang_vel,
        target.ei,
        target.eI
    ])
            
# =============================================================================
# Results
# =============================================================================
data = {
    "position_history": position_history,
    "velocity_history": velocity_history,
    "acceleration_history": acceleration_history,
    "target_true_hist": target_true_hist,
    "radial_history": radial_hist,
    "removed_initial_pos": False,
    "p_d_hist": p_d_hist,
    "perceived_target_hist": perceived_target_hist,
    "colors": plt.cm.tab10(np.linspace(0, 1, param.sim.nD)),
    "active_ids": active_ids,
    "numAgents_history": numAgents_history
}

if param.display.store_data == True:
    print("Saving simulation data (.pkl)")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "simulation_results.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

if param.display.verbose:
    sim_data = SimulationData(**data)
    create_figures(sim_data, param)