import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from Structs_sim import Param, Voronoi, SimulationData
from Structs_agent import DroneState
from voronoi_mesh_funcs import pre_compute_desired_pos
from delaunay_triangulation_network import delaunay_triangulation_network
from update_target_measurement import update_target_measurement
from compute_desired_goal import compute_desired_goal
from plant import PythonPlant, CrazyfliePlant
from animation_display import create_figures
from environment import avoid_obstacles_reference, project_out, safe_setpoint
from obstacle_avoidance_funcs import init_obstacle_avoidance, solve_unconstrained_motion, set_goal_cost, AdmmMPC, mpc_setpoint

def main(param: Param | None = None, plant=None):
    """
    Run the simulation.

    param : simulation parameters (default: Param())
    plant : drone dynamics (default: PythonPlant, the Python quadrotor model,
            or CrazyfliePlant in the arena experiment; see plant.py for
            the interface)
    """
    # =========================================================================
    # Simulation initialization
    # =========================================================================
    if param is None:
        param = Param()
    if plant is None:
        plant = CrazyfliePlant(param) if param.sim.arena_experiment else PythonPlant(param)
    voronoi = Voronoi(param)
    airspace = param.airspace
    environment = param.environment

    origin_p_d, _, voronoi = pre_compute_desired_pos(voronoi, param)

    # =========================================================================
    # Chaser initialization
    # =========================================================================
    chaser_list: list[DroneState] = []
    for i in range(param.sim.nD):
        drone = DroneState(param)
        # start on the (airspace-feasible) formation goal
        start = airspace.project((origin_p_d[:, i:i+1] + param.traj.initial_target_pos).reshape(1, 3))
        # not inside / too close to an obstacle (placement, wide search)
        start = project_out(start, environment.detect(start, 10 * param.sim.sensing_range),
                            param.sim.env_clearance)
        drone.position = start.reshape(-1, 1)
        drone.desired.goal = drone.position
        chaser_list.append(drone)

    obs_vars = init_obstacle_avoidance(chaser_list, param)
    A_obs_blk = obs_vars["A_obs_blk"]
    G = obs_vars["G"]
    W0 = obs_vars["W0"]
    W1 = obs_vars["W1"]
    W2 = obs_vars["W2"]

    # A_obs initialized for a unconstrained motion
    for iD, agent in enumerate(chaser_list):
        # update goal
        set_goal_cost(agent, obs_vars, param)
        agent.constraints.s = np.zeros(G.shape[0]).reshape(-1,1)
        agent.constraints.lambda_obs = np.zeros([3*param.sim.horizon,1])
        agent.constraints.lambda_ineq = np.zeros([G.shape[0],1])
        agent.constraints.A_obs = A_obs_blk
        # get 1st trajectory
        solve_unconstrained_motion(obs_vars, agent, param)

    # MPC / ADMM (KKT factorized once, rebuilt only when drones are dropped)
    mpc = AdmmMPC(obs_vars, chaser_list, param)

    target_true_hist = np.zeros([24, param.sim.Nsim])
    numAgents_history = np.ones(param.sim.Nsim) * param.sim.nD

    # =========================================================================
    # Target initialization
    # =========================================================================
    target = DroneState(param)
    # start position not inside / too close to an obstacle (placement, wide search)
    target.position = project_out(param.traj.initial_target_pos.reshape(1, 3),
                                  environment.detect(param.traj.initial_target_pos, 10 * param.sim.sensing_range),
                                  param.sim.env_clearance).reshape(-1, 1)

    # =========================================================================
    # Simulation variables initialization
    # =========================================================================
    ema_estimated = np.tile(
        np.concatenate([target.position.reshape(3, 1), target.velocity.reshape(3, 1)]),
        param.sim.nD)
    active_mask = np.ones(param.sim.nD, dtype=bool)
    active_ids = np.where(active_mask)[0]
    inactive_ids = np.where(~active_mask)[0]

    mpc.target_state = (target.position, target.velocity)
    mpc.rebuild(active_ids)

    # might do nothing (python version) but important in the Gazebo version of plant
    plant.start(chaser_list, target)

    # =========================================================================
    # Logs initialization
    # =========================================================================
    position_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
    velocity_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
    acceleration_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
    perceived_target_hist = np.zeros([param.sim.nD, 6, param.sim.Nsim])
    p_d_hist = np.zeros((param.sim.nD, 3, param.sim.Nsim))
    radial_hist = np.zeros((param.sim.nD, param.sim.Nsim))

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    for k in range(param.sim.Nsim):
        if k % 100 == 0:
            print(f"Computing simulation iteration: {k:04d} of {param.sim.Nsim}")
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
                agent: DroneState = chaser_list[idx]
                agent.desired.goal = np.nan

            # number of obstacles changed -> new tiled A_obs and factorization.
            # Dropped drones are no longer treated as obstacles.
            mpc.rebuild(active_ids)

        # ---------------------------------------------------------
        # Move target
        # ---------------------------------------------------------
        target.desired.position = param.traj.ref_target_traj["pos"](current_time+param.sim.dTi)
        target.desired.velocity = param.traj.ref_target_traj["vel"](current_time+param.sim.dTi)
        # the target avoids the obstacles it senses
        target = avoid_obstacles_reference(target, environment, param)
        target = plant.actuate(target)

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
            chaser_list[iD] = plant.actuate_passive(chaser_list[iD])

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

            # obstacles sensed by the drone (used by its MPC)
            agent.perceived_obstacles = environment.detect(agent.position, param.sim.sensing_range)

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
            for iD in active_ids:
                set_goal_cost(chaser_list[iD], obs_vars, param)
            mpc.target_state = (target.position, target.velocity)
            mpc.solve(active_ids)

        for iD, agent in enumerate([chaser_list[i] for i in active_ids]):
            if param.sim.use_mpc and param.sim.mpc_tight_tracking:
                p_ref, v_ref, a_ref = mpc_setpoint(agent, param)
            elif param.sim.use_mpc:
                # Pass desired values to agent: first value of the horizon
                C = agent.constraints.C
                p_ref = (W0 @ C)[1].reshape(-1,1)
                v_ref = (W1 @ C)[1].reshape(-1,1)
                a_ref = (W2 @ C)[1].reshape(-1,1)
            else:
                # no MPC: straight to the goal
                p_ref = agent.desired.goal
                v_ref = agent.desired.goal_velocity
                a_ref = np.zeros((3,1))

            # through the obstacle safety filter
            (agent.desired.position, agent.desired.velocity, agent.desired.acceleration) = safe_setpoint(
                p_ref, v_ref, a_ref,
                agent.perceived_obstacles, param.sim.env_clearance,
                agent.position, param.sim.env_brake_gain)

            # Distance to target for logging purposes
            q_s = agent.position - target.position
            agent.radii_to_target = np.linalg.norm(q_s)

            # actuate control
            agent = plant.actuate(agent)

        if np.isnan(origin_p_d[:, active_ids]).any():
            raise RuntimeError("A position value is NaN!")

        voronoi.sites = voronoi.newSites

        # propagate the dynamics (no-op for the Python plant, which integrates
        # in actuate; physics step for Gazebo)
        plant.advance(chaser_list, target)

        #logs values
        for iD, agent in enumerate(chaser_list):
            position_history[iD,:, k:k+1] = agent.position
            # actual acceleration over the step (agent.acceleration is never
            # updated; with the Euler integration this is exactly dot_v)
            prev_vel = velocity_history[iD,:, k-1:k] if k > 0 else np.zeros((3, 1))
            acceleration_history[iD,:, k:k+1] = (agent.velocity - prev_vel) / param.sim.dTi
            velocity_history[iD,:, k:k+1] = agent.velocity
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

    plant.stop()

    # =========================================================================
    # Results
    # =========================================================================
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
        file_path = os.path.join(current_folder, param.display.results_file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    if param.display.verbose:
        sim_data = SimulationData(**data)
        create_figures(sim_data, param)

if __name__ == "__main__":
    main()
