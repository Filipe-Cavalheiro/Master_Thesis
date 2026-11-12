import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Project Capture
# Bruno Guerreiro
# Filipe Cavalheiro

from Structs import Param, Voronoi, DroneState, SimulationData, Site
from voronoi_mesh_funcs import pre_compute_desired_pos
from delaunay_triangulation_network import delaunay_triangulation_network
from compute_desired_goal import compute_desired_goal
from Animation_display import create_figures
from actuate_control import actuate_control

# =============================================================================
# Simulation initialization
# =============================================================================
param = Param()
voronoi = Voronoi(param)
origin_p_d, swarm_center, voronoi = pre_compute_desired_pos(voronoi, param)
voronoi.sites[0].weight = 2.0
gfx = None

chaser_list: list[DroneState] = []
for i in range(param.sim.nD):
    drone = DroneState(param)
    drone.position = origin_p_d[:, i:i+1] + swarm_center
    drone.desired.goal = drone.position
    chaser_list.append(drone)

target = DroneState(param)
target.position = param.traj.initial_target_pos.copy()
target_true_hist = np.zeros([24, param.sim.Nsim])

active_mask = np.ones(param.sim.nD, dtype=bool)
active_ids = np.where(active_mask)[0]

position_history = np.zeros([param.sim.nD, 3, param.sim.Nsim])
p_d_hist = np.zeros((param.sim.nD, 3, param.sim.Nsim))

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
    # Active drones
    # ---------------------------------------------------------
    for idx, iD in enumerate(active_ids):
        agent = chaser_list[iD]
        agent.estimated.target_position = target.position

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

    voronoi.sites = [
        Site(site.index, site.weight)
        for site in voronoi.newSite
    ]

    #logs values
    for iD, agent in enumerate(chaser_list):
        position_history[iD,:, k:k+1] = agent.position
        p_d_hist[iD,:,k:k+1] = agent.desired.goal
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
    "p_d_hist": p_d_hist,
    "colors": plt.cm.tab10(np.linspace(0, 1, param.sim.nD)),
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
