function [p_d, swarm_center, voronoi] = pre_compute_desired_pos(voronoi, Param)
%PRE_COMPUTE_DESIRED_POS Summary of this function goes here
%   Detailed explanation goes here

if strcmp(Param.distribuition, 'fibonacci')
    p_d = get_fibonacci_points(Param.nD, Param.sphere_radius, [0;0;0]);
    swarm_center = Param.initial_target_pos;

elseif strcmp(Param.distribuition, 'dijkstra')
    [voronoi.v_circle, voronoi.v_cone, voronoi.v_sphere] = pre_compute_meshes(1001,Param);
    
    [voronoi.vertices, voronoi.faces, voronoi.mesh_pos] = pos_based_mesh_transform( ...
        Param.initial_target_pos, ...
        voronoi.mesh_pos, ...
        voronoi.surf_direction, ...
        voronoi.v_circle, ...
        voronoi.v_cone, ...
        voronoi.v_sphere, ...
        voronoi.faces, ...
        Param);
    % get initial desires_final_pos (p_d)
    N = size(voronoi.vertices,1);
    voronoi.sites = randperm(N, Param.nD);
    swarm_center = voronoi.mesh_pos;
    p_d = voronoi.vertices(voronoi.sites(:), :)' - Param.initial_target_pos;
end

end

