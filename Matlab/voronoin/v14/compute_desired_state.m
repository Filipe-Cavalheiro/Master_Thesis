function [desired, voronoi] = compute_desired_state( ...
    k, iD, gfx, desired, origin_p_d, voronoi, neighbours, ...
    perceived_target_hist, Param)

% Extract target state once
target_state = perceived_target_hist{k, iD};
target_pos   = target_state(1:3);
desired.psi = 0;
desired.acc = zeros(3,1);
desired.jerk = zeros(3,1);

if Param.measurable_states == 2
    target_vel = target_state(4:6);
else
    target_vel = zeros(3,1);
end

desired.x_2dot = [0; 0; 0];
desired.x_3dot = [0; 0; 0];
desired.x_4dot = [0; 0; 0];

desired.b1 = [1; 0; 0];
desired.b1_dot = [0; 0; 0];
desired.b1_2dot = [0; 0; 0];

switch Param.distribuition

    case 'fibonacci'
        desired.pos(:,iD) = origin_p_d + target_pos;
        desired.vel = target_vel;

    case 'dijkstra'

        [voronoi.vertices, voronoi.faces, voronoi.mesh_pos] = ...
            pos_based_mesh_transform( ...
                target_pos, ...
                voronoi.mesh_pos, ...
                voronoi.surf_direction, ...
                voronoi.v_circle, ...
                voronoi.v_cone, ...
                voronoi.v_sphere, ...
                voronoi.faces, ...
                Param);

        if k == 1 || mod(k,100) == 0
            sites = voronoi.sites([iD, neighbours]);

            voronoi.newSite(iD) = runCVT(voronoi, sites, gfx);
        end

        desired.pos(:,iD) = voronoi.vertices(voronoi.sites(iD), :)';
        if target_pos >= 0
            desired.vel = target_vel;
        else
            desired.vel = [0; target_vel(2); target_vel(3)];
        end

    otherwise
        error('Unknown distribution type: %s', Param.distribuition);
end

end