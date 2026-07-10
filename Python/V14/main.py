import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation

# Project Capture
# Bruno Guerreiro
# Filipe Cavalheiro

from Structs import Param, Voronoi, DroneState, SimulationData
from voronoi_mesh_funcs import pre_compute_desired_pos
from target_motion import target_motion
from delaunay_triangulation_network import delaunay_triangulation_network
from estimate_target_variables import estimate_target_variables
from compute_desired_state import compute_desired_state
from actuate_control import actuate_control
from drone_3dfull_dyn import drone_3dfull_dyn
from Animation_display import create_figures
from helper_funcs import *
from init_obstacle_avoidance import init_obstacle_avoidance

# =============================================================================
# Simulation initialization
# =============================================================================
param = Param()
voronoi = Voronoi(param)

origin_p_d, swarm_center, voronoi = pre_compute_desired_pos(voronoi, param)
# else:
gfx = None

# =============================================================================
# Chaser initialization
# =============================================================================
chaser_list: list[DroneState] = []
for i in range(param.sim.nD):
    drone = DroneState(param)
    drone.position = origin_p_d[:, i:i+1] + swarm_center
    drone.desired.position = drone.position
    chaser_list.append(drone)

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

    old = target.position
    target_motion(
        target,
        param
    )

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
        chaser_i = chaser_list[iD]

        ema_estimated = (
            estimate_target_variables(
                target,
                chaser_i,
                ema_estimated,
                idx,
                laplacian,
                param
            )
        )

        neighbours = active_ids[connections[idx]]

        if chaser_i.estimated.target_position.size == 0:
            pass

        voronoi = compute_desired_state(
            k,
            iD,
            origin_p_d,
            voronoi,
            chaser_i,
            neighbours,
            param,
        )

        compute_horizon_to_desired_state

        chaser_list[iD] = actuate_control(chaser_list[iD], param)

    if np.isnan(origin_p_d[:, active_ids]).any():
        raise RuntimeError("A position value is NaN!")

    voronoi.sites = voronoi.newSite

    #logs values
    for iD in range(param.sim.nD):
        chaser_i = chaser_list[iD]
        position_history[iD,:, k:k+1] = chaser_i.position
        perceived_target_hist[iD,:,k:k+1] = np.concatenate(
            (chaser_i.estimated.target_position, 
             chaser_i.estimated.target_velocity
            ), axis=0
        )
        p_d_hist[iD,:,k:k+1] = chaser_i.desired.position
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
    "removed_initial_pos": False,
    "p_d_hist": p_d_hist,
    "perceived_target_hist": perceived_target_hist,
    "colors": plt.cm.tab10(np.linspace(0, 1, param.sim.nD)),
    "active_ids": active_ids,
    "numAgents_history": numAgents_history
}
sim_data = SimulationData(**data)

create_figures(sim_data, param)