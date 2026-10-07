clear; clc; close all;

numAgents = 10;
agents = randn(numAgents, 3);
agents = agents ./ vecnorm(agents, 2, 2); 

% K contains the indices of the vertices for each triangular face
K = convhull(agents);

% Calculate Voronoi Vertices
% These are the corners of our Voronoi cells
voronoi_vertices = zeros(size(K,1), 3);
new_agents = zeros(size(agents));

for i = 1:size(K,1)
    p = agents(K(i,:), :);
    % The normal to the triangle is the Voronoi vertex on the sphere
    v_vert = cross(p(2,:) - p(1,:), p(3,:) - p(1,:));
    voronoi_vertices(i,:) = v_vert / norm(v_vert); 
end

% Visualization
figure('Color', 'w');
axis equal; 
view(3); 
grid off;
hold on;
title('Spherical Voronoi Partition');

[x_sphere, y_sphere, z_sphere] = sphere;
hSphere = surf(x_sphere, y_sphere, z_sphere);
hSphere.FaceColor = 'none';
hSphere.EdgeColor = 'b';
hSphere.EdgeAlpha = 0.1;

plot3(agents(:,1), agents(:,2), agents(:,3), 'k.', 'MarkerSize', 15);

colors = lines(numAgents);

% Draw each Voronoi Cell
for i = 1:numAgents
    % Find all faces of the convex hull that include this agent
    [faceIdx, ~] = find(K == i);
    
    % Get the Voronoi vertices corresponding to these faces
    cell_verts = voronoi_vertices(faceIdx, :);
    
    % To plot a patch, vertices must be ordered around the center
    % We project them to a 2D plane temporarily to sort by angle
    center = agents(i,:);
    [~, ~, V] = svd(cell_verts - center);
    proj = (cell_verts - center) * V(:, 1:2);
    angles = atan2(proj(:,2), proj(:,1));
    [~, sortIdx] = sort(angles);
    cell_verts = cell_verts(sortIdx, :);
    raw_centroid = mean(cell_verts, 1);
    spherical_centroid = raw_centroid / norm(raw_centroid);
    scatter3(spherical_centroid(1), spherical_centroid(2), spherical_centroid(3), ...
             40, 'r', 'filled', 'MarkerEdgeColor', 'k');
    
    % Draw the colored patch
    patch(cell_verts(:,1), cell_verts(:,2), cell_verts(:,3), ...
          colors(i,:), 'FaceAlpha', 1.0, 'EdgeColor', 'k', 'LineWidth', 1.5);
end