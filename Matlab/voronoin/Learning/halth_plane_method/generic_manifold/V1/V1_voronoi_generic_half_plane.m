clear; clc; close all;

% Parameters
Tend = 1.0;
dTi = 0.01;
Nsim = round(Tend/dTi);
numAgents = 8;
R = 1.0;
verbose = true;
nTheta = 100;    % Angular resolution
nR = 50;         % Radial resolution

agents_history = cell(Nsim+1,numAgents);

% % Generate circle points
% theta = 2*pi*rand(numAgents,1);
% radius = R * sqrt(rand(numAgents,1));
% x = radius .* cos(theta);
% y = radius .* sin(theta);
% z = zeros(size(x));
% agents = [x,y,z];
% 
% % Generate circle grid (polar coordinates)
% [theta, radius] = meshgrid(linspace(0, 2*pi, nTheta), linspace(0, R, nR));
% x_mesh = radius .* cos(theta);
% y_mesh = radius .* sin(theta);
% z_mesh = zeros(size(x_mesh));
% surf_mesh = [x_mesh(:), y_mesh(:), z_mesh(:)];

% Generate cone points
theta = 2*pi*rand(numAgents,1);
radius = R * sqrt(rand(numAgents,1));
x = radius .* cos(theta);
y = radius .* sin(theta);
z = R * (1 - radius/R);
agents = [x,y,z];

% agents =[-0.7109    0.4904    0.1354;
%    -0.7981   -0.1667    0.1840;
%     0.2699   -0.0381    0.7264;
%    -0.7786   -0.4667    0.0912;
%    -0.9023    0.0055    0.0972;
%     0.0161   -0.7439    0.2553;
%    -0.0188    0.6119    0.3871;
%    -0.5060    0.1951    0.4571];

% Generate cone grid (polar coordinates)
[theta, radius] = meshgrid(linspace(0, 2*pi, nTheta), linspace(0, R, nR));
x_mesh = radius .* cos(theta);
y_mesh = radius .* sin(theta);
z_mesh = R * (1 - radius/R);
surf_mesh = [x_mesh(:), y_mesh(:), z_mesh(:)];

% % Generate sphere points
% agents_unit = randn(numAgents,3);
% agents_unit = agents_unit ./ vecnorm(agents_unit,2,2);
% agents = agents_unit*R;
% 
% % Generate circle grid (polar coordinates)
% theta = 2*pi*rand(nTheta*nR,1);
% phi = acos(2*rand(nTheta*nR,1)-1);
% x_mesh = R*sin(phi).*cos(theta);
% y_mesh = R*sin(phi).*sin(theta);
% z_mesh = R*cos(phi);
% 
% surf_mesh = [x_mesh, y_mesh, z_mesh];

scale = 2*R; % length of line for visualization

for k = 1:Nsim
    fprintf("Computing simulation interation: %04d\n", k);

    agents_history{k} = agents;
    new_agents = zeros(numAgents,3);

    agents_surf_mesh = [[x_mesh(:); agents(:,1)],...
                    [y_mesh(:); agents(:,2)],...
                    [z_mesh(:); agents(:,3)]];

    agents_surf_mesh = uniquetol(agents_surf_mesh, 1e-10, 'ByRows', true);

    % Find columns that are all zeros
    zero_cols = all(agents_surf_mesh == 0, 1);
    
    % --- Delaunay triangulation network ---
    [connections, Laplacian, degree_matrix]  = Delaunay_triangulation_network(...
        agents_surf_mesh(:, ~zero_cols));
    G = graph(connections);

    % --- Half-plane Voronoi computation ---
    for iD = 1:numAgents
        voronoi_vertices = [];
        tangent_vec = [];

        if verbose
            fig = figure;
            hold on;
            axis equal;
            view(3);
            scatter3(agents_surf_mesh(:,1), agents_surf_mesh(:,2), agents_surf_mesh(:,3));
            scatter3(agents(:,1), agents(:,2), agents(:,3), 'r');
            scatter3(agents(iD,1), agents(iD,2), agents(iD,3), 'x', 'g');
        end
        
        % --- Loop over all agents to compute the Voronoi cell ---
        zero_cols = all(agents == 0, 1);
        DT = delaunay(agents(:, ~zero_cols));
        
        % Keep only triangles that contain iD
        DT_filtered = DT(any(DT == iD, 2), :);
        agents_neightbour_iD = unique(DT_filtered(:));
        agents_neightbour_iD(agents_neightbour_iD == iD) = []; 
        
        % figure;
        % axis equal;
        % triplot(DT_filtered, agents(:,1), agents(:,2));
        % hold on;
        % 
        % scatter(agents(iD,1), agents(iD,2), 'x', 'g');
        % 
        % 
        % % Number each point
        % for k = 1:size(agents,1)
        %     text(agents(k,1), agents(k,2), sprintf(' %d', k), ...
        %         'FontSize', 10, 'Color', 'b');
        % end
                
        % --- Loop over all agents to compute the Voronoi cell ---
        idx = 1;
        for j = agents_neightbour_iD'
            p1 = agents(iD,:);       % current agent
            p2 = agents(j,:);        % other agent
            
            % Find indices for the agents in the surface mesh
            [~, p1_idx] = ismember(p1, agents_surf_mesh, 'rows');
            [~, p2_idx] = ismember(p2, agents_surf_mesh, 'rows');
            
            % Shortest path along the mesh
            [path, path_len] = shortestpath(G, p1_idx, p2_idx);
            cumulative_dist = [0; cumsum(vecnorm(diff(agents_surf_mesh(path,:)),2,2))];
            half_len = cumulative_dist(end)/2;
        
            %[v, w] = unique( A, 'stable' );
            %duplicate_indices = setdiff( 1:numel(A), w );

            % Interpolate to find midpoint of the shortest path
            x_mid = interp1(cumulative_dist, agents_surf_mesh(path,1), half_len);
            y_mid = interp1(cumulative_dist, agents_surf_mesh(path,2), half_len);
            z_mid = interp1(cumulative_dist, agents_surf_mesh(path,3), half_len);
            mid = [x_mid, y_mid, z_mid];
            
            % Direction vector between agents
            dir_vec = p1 - p2;
            dir_vec = dir_vec / norm(dir_vec);
            
            % Surface normal
            idx_neighbors = knnsearch(agents_surf_mesh, mid, 'K', 10);
            neighbors = agents_surf_mesh(idx_neighbors,:);

            C = cov(neighbors);
            [V, D] = eig(C);
            
            [~, min_idx] = min(diag(D));
            surface_normal = V(:, min_idx)';
            surface_normal = surface_normal / norm(surface_normal);

            % Project vector onto tangent plane
            n_proj = dir_vec - dot(dir_vec, surface_normal) * surface_normal;
            n_proj = n_proj / norm(n_proj);
            
            % Tangent vector is perpendicular to both the normal and projected vector
            tangent = cross(surface_normal, n_proj);
            tangent = tangent / norm(tangent);
            
            voronoi_vertices(idx,:) = mid;
            tangent_vec(idx,:) = tangent;
            normal_vec(idx,:) = n_proj;
        
            idx = idx + 1;

            if verbose
                % --- Build a plane orthogonal to t (since t is your half-space normal) ---
                plane_scale = scale;

                % Get two orthonormal vectors spanning the plane
                B = null(n_proj);         % 3x2 matrix, columns span plane perpendicular to t
                v1 = B(:,1)';
                v2 = B(:,2)';

                % Create grid on the plane
                [s, u] = meshgrid(linspace(-plane_scale, plane_scale, 15));

                plane_pts = mid + s(:)*v1 + u(:)*v2;
                plane_pts = reshape(plane_pts, [size(s,1), size(s,2), 3]);

                % Plot plane
                surf(plane_pts(:,:,1), plane_pts(:,:,2), plane_pts(:,:,3), ...
                     'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'b');
                scatter3(mid(1), mid(2), mid(3), 'x', 'r');
                scatter3(agents(j, 1), agents(j, 2), agents(j, 3), 'x', 'r');
            end
        end
                
        % --- Masking logic to select Voronoi cell ---
        P = surf_mesh;
        
        % Initialize mask to include all points
        mask = true(size(P,1),1);
        
        % Loop through the half-planes and refine the mask
        for j = 1:size(voronoi_vertices,1)
            center = voronoi_vertices(j,:);
            normal_vec_j = normal_vec(j,:);
        
            % Ensure normal points *away* from agent iD
            to_agent = agents(iD,:) - center;
            if dot(normal_vec_j, to_agent) < 0
                normal_vec_j = -normal_vec_j;
            end
        
            mask = mask & ((P - center) * normal_vec_j' > 0);
        end

        P(~mask, :) = [];
                
        % Compute the final centroid of the Voronoi cell
        centroid = mean(P);
        if verbose
            scatter3(P(:,1), P(:, 2), P(:, 3), 'yellow', 'filled')
            scatter3(agents(:,1), agents(:,2), agents(:,3), 'r');
            scatter3(centroid(1), centroid(2), centroid(3), 'x', 'k');  % Plot the centroid of the Voronoi cell
            set(fig, 'CurrentCharacter', char(0));
            while true
                pause(0.1);
                ch = get(fig, 'CurrentCharacter');
                if ch == 'q'
                    break;
                end
            end
            close(fig);
        end

        k_NN = 10; % number of neighbors
        % Find k nearest mesh points
        diffs = surf_mesh - centroid;
        [~, idxs] = mink(vecnorm(diffs,2,2), k_NN);
        neighbors = surf_mesh(idxs,:);
        
        % Fit plane using PCA
        C = mean(neighbors);
        X = neighbors - C;
        
        [~,~,V] = svd(X, 'econ');
        normal = V(:,3); % smallest singular value direction
        
        % Project centroid onto plane
        projected = centroid - dot(centroid - C, normal') * normal';
        new_agents(iD,:) = projected;  % Update the agent's new position
        if isnan(projected)
            error('NaN detected! Stopping program.');
        end
    end
    agents = new_agents;
end

%%
figure;
hold on;
axis equal;
grid on;
view(3);

colors = lines(numAgents); % color per agent

for i = 1:numAgents
    traj = [];
    
    for k = 1:Nsim
        if size(agents_history{k},1) >= i
            traj = [traj; agents_history{k}(i,:)];
        end
    end
    
    plot3(traj(:,1), traj(:,2), traj(:,3), '-', 'Color', colors(i,:));
    scatter3(traj(end,1), traj(end,2), traj(end,3), 50, colors(i,:), 'filled');
end

% draw sphere
[xs,ys,zs] = sphere(50);
surf(xs,ys,zs,'FaceAlpha',0.05,'EdgeColor','none');

title('Agent trajectories over time');

%%
figure;
axis equal;
grid on;
view(3);
hold on;

%surf(x_mesh, y_mesh, z_mesh,'FaceAlpha',0.05,'EdgeColor','none');

h = scatter3([],[],[],50,'filled');

for k = 1:Nsim
    A = agents_history{k};
    
    set(h, 'XData', A(:,1), 'YData', A(:,2), 'ZData', A(:,3));
    title(['Time step: ', num2str(k)]);
    
    drawnow;
    pause(0.05);
end

% %%
% figure;
% hold on;
% 
% % Draw semi-transparent sphere
% [XS, YS, ZS] = sphere(100);   
% surf(R*XS, R*YS, R*ZS, ...
%     'FaceAlpha', 1.0, ...
%     'EdgeColor', 'none', ...
%     'FaceColor', [0.7647 0.7647 0.7647]); 
% 
% % Plot points
% A = agents_history{Nsim};
% scatter3(A(:,1), A(:,2), A(:,3), 50, 'k', 'filled');
% 
% % Formatting for paper
% axis equal
% axis off
% set(gca, 'Visible', 'off')
% set(gcf, 'Color', 'w');          % white background
% set(gca, 'Position', [0 0 1 1]); % fill figure
% 
% view(3);
% print2pdf("Lloyd_half_plane_normal.pdf", 1);
% view(90,0);
% print2pdf("Lloyd_half_plane_side.pdf", 1);
% view(0,90);
% print2pdf("Lloyd_half_plane_top.pdf", 1);