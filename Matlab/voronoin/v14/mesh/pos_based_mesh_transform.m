function [vertices, faces, mesh_pos] = pos_based_mesh_transform(p, mesh_pos, surf_direction, v_circle, v_cone, v_sphere, faces, Param)
    

    persistent ran_u2_eq_1;

    if isempty(ran_u2_eq_1)
        ran_u2_eq_1 = false;
    end
    
    % Distance along surface normal
    dist_surface = dot(p, surf_direction);

    % Compute u_target
    if dist_surface < -(Param.sphere_radius/2)
        u_target = 0;
    elseif dist_surface > -(Param.sphere_radius/2)
        u_target = 0.5 + (dist_surface / Param.sphere_radius);
    elseif dist_surface >= Param.sphere_radius/2
        u_target = dist_surface / (Param.sphere_radius/2);
    else
        u_target = 2;
    end

    % Split stages
    if u_target <= 1
        u1 = u_target;
        u2 = 0;
        mesh_pos = [0; p(2); p(3)];
    else
        u1 = 1;
        u2 = min(u_target - 1,1);
        mesh_pos = [p(1) - Param.sphere_radius/2 + u2*Param.sphere_radius/2; p(2); p(3)];
    end

    % --- Circle → cone (operate per vertex) ---
    v_current = (1-u1).*v_circle + u1.*v_cone;
    
    if u2 > 0
        v_current = (1-u2).*v_current + u2.*v_sphere;
    end
    
    vertices = v_current + mesh_pos';

    if u2 == 1 && ~ran_u2_eq_1
        % Run this block ONLY once
        faces = convhull(vertices(:,1), vertices(:,2), vertices(:,3));
        ran_u2_eq_1 = true;

    elseif u2 ~= 1
        % Reset flag if we leave u2 == 1
        ran_u2_eq_1 = false;

        if u2 >= 0.2
            faces = convhull(vertices(:,1), vertices(:,2), vertices(:,3));
        else
            DT = delaunayTriangulation(v_circle(:,2), v_circle(:,3));
            faces = DT.ConnectivityList;
        end
    end
end