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

    airspace = voronoi.airspace
    n_in, dist_surface, past_red, _ = airspace.wall_frame(target_pos)
    # Before the wall, or with the target inside the red zone, the formation
    # can't follow the target along the wall normal: drop that component.
    if dist_surface < 0 or past_red > 0:
        n = n_in.reshape(-1, 1)
        wall_vel = target_vel - (target_vel.T @ n) * n
    else:
        wall_vel = target_vel

    agent.desired.goal_velocity = wall_vel

    if param.sim.distribution == "fibonacci":
        goal = (origin_p_d[:,iD:iD+1] + target_pos).reshape(1, 3)
        agent.desired.goal = airspace.project(goal).reshape(-1, 1)
        agent.desired.velocity = wall_vel

    elif param.sim.distribution == "dijkstra":
        (voronoi.vertices, voronoi.faces, voronoi.mesh_pos
        ) = pos_based_mesh_transform(target_pos, voronoi, param)

        if k % param.sim.update_CVT_interval == 0:
            sites = [voronoi.sites[i] for i in [iD,*neighbours.tolist()]]
            voronoi.newSites[iD] = runCVT(voronoi, sites, agent, param)

        agent.desired.goal = voronoi.vertices[voronoi.sites[iD], :].reshape(-1,1)

        agent.desired.velocity = wall_vel

    else:
        raise ValueError(
            f"Unknown distribution type: {param.sim.distribution}"
        )

    return agent, voronoi