import numpy as np
from voronoi_mesh_funcs import pos_based_mesh_transform, runCVT
from Structs_sim import Voronoi, Param
from Structs_agent import DroneState

def compute_desired_goal(
    k,
    iD,
    origin_p_d,
    voronoi: Voronoi,
    agent: DroneState,
    neighbours,
    param: Param,
):
    # Extract target state
    target_pos = agent.estimated.target_position
    if param.sim.measurement_dimension == 2:
        target_vel = agent.estimated.target_velocity
    else:
        target_vel = np.zeros([3,1])

    (voronoi.vertices, voronoi.faces, voronoi.mesh_pos
    ) = pos_based_mesh_transform(target_pos, voronoi, param)

    if k % param.sim.update_CVT_interval == 0:
        sites = [voronoi.sites[i] for i in [iD,*neighbours.tolist()]]
        voronoi.newSites[iD] = runCVT(voronoi, sites, agent, param)

    agent.desired.goal = voronoi.vertices[voronoi.sites[iD], :].reshape(-1,1)

    if target_pos[0] >= 0:
        agent.desired.velocity = target_vel
    else:
        agent.desired.velocity = np.array(
            [0.0, target_vel[1,0], target_vel[2,0]]
        ).reshape(-1,1)

    return agent, voronoi