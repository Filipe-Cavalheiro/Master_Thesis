function [new_q_d, voronoi_vertices] = voronoi_update(iD, q_d ,neighbours, c_sphere_estimated, Rad)
%VORONOI_UPDATE Summary of this function goes here
%   Detailed explanation goes here
    % --- build local point set ---
    neigh_pos = [q_d{iD}'; cell2mat(q_d(neighbours))'];

    % ---------- Shift to sphere frame (back to origin)----------
    neigh_pos  = neigh_pos - c_sphere_estimated';
    neigh_pos = neigh_pos ./ vecnorm(neigh_pos, 2, 2);

    % compute convex hull
    K = convhull(neigh_pos);

    voronoi_vertices = [];

    for j = 1:size(K,1)
        % only keep faces that include the agent
        if any(K(j,:) == 1)
            p = neigh_pos(K(j,:),:);

            vertice = cross(p(2,:) - p(1,:), p(3,:) - p(1,:));
            vertice = vertice / norm(vertice);

            center_shifted = (q_d{iD} - c_sphere_estimated)';
            center_shifted = center_shifted./vecnorm(center_shifted, 2, 2);
            if dot(vertice, center_shifted) <= 0
                vertice = -vertice;
            end

            voronoi_vertices = [voronoi_vertices; vertice];
        end
    end

    center = (q_d{iD} - c_sphere_estimated)';
    center = center./vecnorm(center, 2, 2);

    [~,~,V] = svd(voronoi_vertices - center);
    proj = (voronoi_vertices - center)*V(:,1:2);

    angles = atan2(proj(:,2),proj(:,1));
    [~,idx] = sort(angles);

    voronoi_vertices = voronoi_vertices(idx,:);

    centroid_sum = [0 0 0];
    
    v0 = voronoi_vertices(1,:);

     for j = 2:size(voronoi_vertices,1)-1
        a = v0;
        b = voronoi_vertices(j,:);
        c = voronoi_vertices(j+1,:);
    
        A = spherical_triangle_area(a,b,c);
    
        tri_centroid = (a + b + c);
        tri_centroid = tri_centroid / norm(tri_centroid);
    
        centroid_sum = centroid_sum + A * tri_centroid;
     end

    if norm(centroid_sum) < 1e-8
        new_q_d = q_d{iD}; % fallback
    else
        new_q_d = c_sphere_estimated + Rad * (centroid_sum / vecnorm(centroid_sum, 2, 2))';
    end

    voronoi_vertices =  c_sphere_estimated' + Rad * (voronoi_vertices ./ vecnorm(voronoi_vertices, 2, 2));
end

