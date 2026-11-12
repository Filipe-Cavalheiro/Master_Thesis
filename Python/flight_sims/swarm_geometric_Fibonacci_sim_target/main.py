import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # flight_sims/ (root_results)
import root_results

# Project Capture
# Bruno Guerreiro
# Filipe Cavalheiro

from Structs import Param, DroneState, SimulationData
from drone_motion import drone_motion
from Animation_display import create_figures
from helper_funcs import *
from fiboancci_funcs import *

# =============================================================================
# Simulation initialization
# =============================================================================
param = Param()
# also copy the results of this run into <repo>/Results/<sim>_<date>/
root_results.mirror_on_exit("swarm_geometric_Fibonacci_sim_target", param.display.save_location)
fib_ref_points = get_fibonacci_points(param.sim.nD, 
                                          param.sim.sphere_radius, 
                                          np.zeros([3,1]))
min_safe_dist = get_min_safe_distance(fib_ref_points, param) * 0.5

# =============================================================================
# Target initialization
# =============================================================================
target = DroneState(param)
target.position = param.traj.initial_target_pos.copy()
active_mask = np.ones(param.sim.nD, dtype=bool)
active_ids = np.where(active_mask)[0]

# =============================================================================
# Chaser initialization
# =============================================================================
chaser_list: list[DroneState] = []


# =============================================================================
# Chaser initialization
# =============================================================================

chaser_list: list[DroneState] = []
if param.sim.compare_with_experimental or param.sim.load_experimental_state:
    (
        position_list,
        velocity_list,
        attitude_list,
        ang_vel_list,
        real_world_time_offset,
    ) = get_initial_real_offset(param)
    for iD in range(param.sim.nD):
        drone = DroneState(param)

        # Set drone states
        drone.position = np.asarray(position_list[iD], dtype=float)
        drone.velocity = np.asarray(velocity_list[iD], dtype=float)
        drone.attitude = np.asarray(attitude_list[iD], dtype=float)
        drone.ang_vel = np.asarray(ang_vel_list[iD], dtype=float)

        # Set actual distance from target
        drone.radii_to_target = np.float32(
            np.linalg.norm(drone.position - target.position)
        )

        # Set desired state
        drone.desired.position = (
            fib_ref_points[iD, :] + target.position
        )
        drone.desired.radii_to_target = np.linalg.norm(
            drone.desired.position - target.position
        )

        chaser_list.append(drone)

else:
    for iD in range(param.sim.nD):
        drone = DroneState(param)

        # Random spherical coordinates
        theta0 = 2 * np.pi * np.random.rand()
        phi0 = np.pi * np.random.rand()
        r0 = 15 * np.random.rand()

        # Convert spherical coordinates to Cartesian coordinates
        initial_pos = r0 * np.array([
            [np.sin(phi0) * np.cos(theta0)],
            [np.sin(phi0) * np.sin(theta0)],
            [np.cos(phi0)],
        ])

        # Set drone position
        drone.position = initial_pos + np.asarray(target.position).reshape(3, 1)

        # Set actual distance from target
        drone.radii_to_target = np.float32(
            np.linalg.norm(drone.position - target.position)
        )

        # Set desired state
        drone.desired.position = (
            fib_ref_points[iD, :] + target.position
        )
        drone.desired.radii_to_target = np.linalg.norm(
            drone.desired.position - target.position
        )
        chaser_list.append(drone)


# Guidance reference state on S^2, integrated by the outer loop (eq. (11)),
# initialized at each drone's initial direction relative to the target
q_ref_list = []
for iD in range(param.sim.nD):
    q_s = chaser_list[iD].position - target.position
    q_ref_list.append(q_s / np.linalg.norm(q_s))

target_true_hist = np.zeros([24, param.sim.Nsim])

# =============================================================================
# Reorder initial positions for agents to be closer to closer fib point
# =============================================================================
if param.sim.nD != 1:
    assignments, assigned_ref = assign_agents_to_fibonacci_points(
        chaser_list,
        target,
        fib_ref_points,
        param,
    )
else:
    assigned_ref = range(param.sim.nD)

# =============================================================================
# Logs initialization
# =============================================================================
position_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
perceived_target_hist = np.zeros([param.sim.nD, 6, param.sim.Nsim])
p_d_hist = np.zeros((param.sim.nD, 3, param.sim.Nsim))
radial_hist = np.zeros((param.sim.nD, param.sim.Nsim))
# Inter-agent distances: initialize as all unique agent pairs (i, j), i < j
agent_pairs = list(combinations(range(param.sim.nD), 2))
inter_agent_dist_hist = np.zeros((len(agent_pairs), param.sim.Nsim))

# =============================================================================
# Main simulation loop
# =============================================================================
for k in range(param.sim.Nsim):
    print(f"Computing simulation iteration: {k:04d}")
    current_time = param.sim.dTi*k

    # ---------------------------------------------------------
    # Move target
    # ---------------------------------------------------------
    target.desired.position = param.traj.ref_target_traj["pos"](current_time+param.sim.dTi)
    target.desired.velocity = param.traj.ref_target_traj["vel"](current_time+param.sim.dTi)
    drone_motion(target, param)

    # ---------------------------------------------------------
    # Active drones
    # ---------------------------------------------------------
    # Pairwise geodesic distances between guidance references (shared by
    # every drone's APF safeguard within this time step)
    geo_matrix = geo_distances(np.hstack(q_ref_list), 1, np.zeros([3, 1]))

    for iD, agent in enumerate(chaser_list):
        # ---------- Shift to sphere frame (back to origin) ----------

        q_s = agent.position - target.position
        agent.radii_to_target = np.linalg.norm(q_s)
        if agent.radii_to_target < 1e-8:
            continue

        # Desired reduced-attitude direction: fibonacci point relative to the
        # target, projected onto the unit sphere (q* = (p* - p_T)/||p* - p_T||)
        q_d = fib_ref_points[assigned_ref[iD], :]
        q_d = q_d / np.linalg.norm(q_d)

        # Position reference generation: integrate the guidance state (eq. (11))
        q_ref_list[iD], e_q = linear_motion_ref(q_ref_list[iD], q_d, param)

        # Safe guard collisions: APF displacement applied to the guidance
        # references (eq. (14)), so references deflect around each other
        if param.sim.nD != 1:
            q_ref_list[iD] = apf_safe_guard(iD, q_ref_list[iD], q_ref_list,
                                            min_safe_dist, param, geo_matrix)
        q_hat = q_ref_list[iD]

        # Update sphere radius reference (integrates its own state, eq. (15))
        agent.desired.radii_to_target = update_radius(agent.desired.radii_to_target, e_q, param)

        # ---------- Enforce S² constraint (local) ----------
        agent.desired.position = agent.desired.radii_to_target * q_hat + target.position

        agent = drone_motion(agent, param)

    #logs values
    for iD in range(param.sim.nD):
        agent = chaser_list[iD]
        position_history[iD,:, k:k+1] = agent.position
        p_d_hist[iD,:,k:k+1] = agent.desired.position
        radial_hist[iD, k] = agent.radii_to_target

    # Inter-agent pairwise distances at this timestep
    for p_idx, (i, j) in enumerate(agent_pairs):
        inter_agent_dist_hist[p_idx, k] = np.linalg.norm(
            chaser_list[i].position - chaser_list[j].position
        )

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
if param.display.verbose:
    data = {
        "position_history": position_history,
        "target_true_hist": target_true_hist,
        "radial_history": radial_hist,
        "inter_agent_dist_hist": inter_agent_dist_hist,
        "agent_pairs": agent_pairs,
        "removed_initial_pos": False,
        "p_d_hist": p_d_hist,
        "min_safe_dist": min_safe_dist,
        "colors": plt.cm.tab10(np.linspace(0, 1, param.sim.nD))
    }
    sim_data = SimulationData(**data)

    create_figures(sim_data, param)

