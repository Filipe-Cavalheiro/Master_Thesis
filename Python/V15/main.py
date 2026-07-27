import numpy as np
import matplotlib.pyplot as plt
import pickle

# Project Capture
# Bruno Guerreiro
# Filipe Cavalheiro

from Structs import Param, Voronoi, DroneState, SimulationData
from voronoi_mesh_funcs import pre_compute_desired_pos
from delaunay_triangulation_network import delaunay_triangulation_network
from update_target_measurement import update_target_measurement
from compute_desired_goal import compute_desired_goal
from actuate_control import actuate_control
from drone_3dfull_dyn import drone_3dfull_dyn
from Animation_display import create_figures
from helper_funcs import *
from obstacle_avoidance_funcs import *

# =============================================================================
# Simulation initialization
# =============================================================================
param = Param()
voronoi = Voronoi(param)

origin_p_d, swarm_center, voronoi = pre_compute_desired_pos(voronoi, param)
gfx = None

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
wN = W0[-1].reshape(-1,1)
x = np.zeros(dim)

# A_obs initialized for a unconstrained motion
for iD, agent in enumerate(chaser_list):
    # update goal
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        agent.constraints.q[idx] = -param.w_goal * agent.desired.goal[axis,0] * wN
    agent.constraints.s = np.zeros(G.shape[0]).reshape(-1,1)
    agent.constraints.lambda_obs = np.zeros([3*param.horizon,1])
    agent.constraints.lambda_ineq = np.zeros([G.shape[0],1])
    agent.constraints.A_obs = A_obs_blk
    # get 1st trajectory
    solve_unconstrained_motion(obs_vars, agent, param)

target_true_hist = np.zeros([24, param.sim.Nsim])
estimated_target_variables = np.zeros((3, param.sim.nD))

numAgents_history = np.ones(param.sim.Nsim) * param.sim.nD
# =============================================================================
# Target initialization
# =============================================================================
target = DroneState(param)
target.position = param.traj.initial_target_pos.copy()
ema_estimated = np.tile(
    np.concatenate([target.position.reshape(3, 1), target.velocity.reshape(3, 1)]),
    param.sim.nD)
active_mask = np.ones(param.sim.nD, dtype=bool)
active_ids = np.where(active_mask)[0]
inactive_ids = np.where(~active_mask)[0]

# =============================================================================
# Logs initialization
# =============================================================================
position_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
perceived_target_hist = np.zeros([param.sim.nD, 6, param.sim.Nsim])
p_d_hist = np.zeros((param.sim.nD, 3, param.sim.Nsim))
radial_hist = np.zeros((param.sim.nD, param.sim.Nsim))

# =============================================================================
# Main simulation loop
# =============================================================================
for k in range(param.sim.Nsim):
    print(f"Computing simulation iteration: {k:04d}")
    current_time = param.sim.dTi*k
    # ---------------------------------------------------------
    # Drop drones
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
        inactive_ids = np.where(~active_mask)[0]

        ema_estimated = np.delete(
            ema_estimated,
            remove_idx,
            axis=1
        )

    # ---------------------------------------------------------
    # Move target
    # ---------------------------------------------------------
    target.desired.position = param.traj.ref_target_traj["pos"](current_time+param.sim.dTi)
    target.desired.velocity = param.traj.ref_target_traj["vel"](current_time+param.sim.dTi)
    target = actuate_control(target, param)

    # ---------------------------------------------------------
    # Network
    # ---------------------------------------------------------
    desired_positions = np.array(
        [chaser_list[idx].position for idx in active_ids]
    ).squeeze(-1)
    connections, laplacian, _ = delaunay_triangulation_network(
        desired_positions
    )

    # ---------------------------------------------------------
    # Inactive drones
    # ---------------------------------------------------------
    for iD in inactive_ids:
        p_d_hist[iD,:, k:k+1] = np.nan
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

    active_agents = list(range(len(active_ids)))
    for _ in range(100):
        for iD in active_agents[:]:
            agent = chaser_list[iD]
            # =====================================================
            # Equality constraint becomes current position
            # =====================================================
            b_eq = np.vstack([agent.position, agent.velocity, agent.acceleration]).reshape(-1,1)

            # Step 1 - solve trajectory
            A_bar = (
                Q
                + param.rho_obs * agent.constraints.A_obs.T @ agent.constraints.A_obs
                + param.rho_ineq * G.T @ G
            )

            b_bar = (
                - agent.constraints.q
                - agent.constraints.A_obs.T @ agent.constraints.lambda_obs
                + param.rho_obs * agent.constraints.A_obs.T @ agent.constraints.b_obs
                - G.T @ agent.constraints.lambda_ineq
                + param.rho_ineq * G.T @ (h - agent.constraints.s)
            )

            KKT = np.block([
                [A_bar, A_eq.T],
                [A_eq, np.zeros((A_eq.shape[0], A_eq.shape[0]))]
            ])

            rhs = np.concatenate([
                b_bar,
                b_eq
            ])

            solution = np.linalg.solve(KKT, rhs)
            x = solution[:dim]
            agent.constraints.C = x.flatten().reshape(n_ctrl,3)
            # Step 2 - Update b_obs and A_obs,lambda_obs dimensions 
            other_chasers = chaser_list[:iD] + chaser_list[iD+1:]
            nD_obstacle_chaser = len(other_chasers)
            if nD_obstacle_chaser != 0:
                agent.constraints.A_obs = np.tile(A_obs_blk, (nD_obstacle_chaser,1))
                agent.constraints.b_obs = np.zeros((nD_obstacle_chaser*3 * param.horizon, 1))
                agent.constraints.lambda_obs = np.zeros([agent.constraints.A_obs.shape[0],1])
                traj_horizon = W0 @ agent.constraints.C
                for iD_obs, obstacle_agent in enumerate(other_chasers):
                    obstacle_horizon =  W0 @ obstacle_agent.constraints.C
                    off_set = (iD_obs*3*param.horizon)
                    for horizon_iter in range(param.horizon):
                        delta = np.linalg.solve(
                            agent.constraints.obstacle_radius,
                            (traj_horizon[horizon_iter] - obstacle_horizon[horizon_iter]).reshape(-1,1))
                        d, alpha, beta = cart2pol_3D(delta)
                        d = max(d, 1.0)
                        b = obstacle_horizon[horizon_iter,:].reshape(3,1) + agent.constraints.obstacle_radius @ (d * omega(alpha, beta))
                        rows = np.arange(horizon_iter+off_set, 3 * param.horizon+off_set, param.horizon)
                        agent.constraints.b_obs[rows,:] = b
                
            # Step 3 - Update slack term
            agent.constraints.s = np.maximum(0,h - G @ x - agent.constraints.lambda_ineq/param.rho_ineq)

            # calculate residuals
            residual_eq  = A_eq @ x - b_eq
            residual_ineq = G @ x - h + agent.constraints.s
            residual_obs = agent.constraints.A_obs @ x - agent.constraints.b_obs

            # Step 4 - Update lambdas
            agent.constraints.lambda_ineq += param.rho_ineq * residual_ineq
            agent.constraints.lambda_obs += param.rho_obs * residual_obs

            eq_norm = np.linalg.norm(residual_eq)
            ineq_norm = np.linalg.norm(residual_ineq)
            obs_norm = np.linalg.norm(residual_obs)

            if (
                eq_norm + ineq_norm + obs_norm < 1e-2
            ):
                active_agents.remove(iD)
                
    for iD, agent in enumerate(chaser_list):
        # reset residues 
        agent.constraints.lambda_obs = np.zeros([agent.constraints.A_obs.shape[0],1])
        agent.constraints.lambda_ineq = np.zeros([G.shape[0],1])

        # Pass desired values to agent
        C = agent.constraints.C
        traj = W0 @ C
        vel = W1 @ C
        acc = W2 @ C
        # first value of the horizon
        agent.desired.acceleration = acc[1].reshape(-1,1)
        agent.desired.velocity = vel[1].reshape(-1,1)
        agent.desired.position = traj[1].reshape(-1,1)

        # Distance to target for logging purposes
        q_s = agent.position - target.position
        agent.radii_to_target = np.linalg.norm(q_s)

        # update desired position goal
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            agent.constraints.q[idx] = -param.w_goal * agent.desired.goal[axis] * wN

        # actuate control
        agent = actuate_control(agent, param)

    if np.isnan(origin_p_d[:, active_ids]).any():
        raise RuntimeError("A position value is NaN!")

    voronoi.sites = voronoi.newSites

    #logs values
    for iD, agent in enumerate(chaser_list):
        position_history[iD,:, k:k+1] = agent.position
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
    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(data, f)

if param.display.verbose:
    sim_data = SimulationData(**data)
    create_figures(sim_data, param)
