clear; clc; close all;

Tend = 1;
dTi = 0.01;
Nsim = round(Tend/dTi);
Nsim = 100;

numAgents = 30;
R = 1.0;

agents_unit = randn(numAgents,3);
agents_unit = agents_unit ./ vecnorm(agents_unit,2,2);
agents = agents_unit*R;
 
agents_history = cell(Nsim+1,numAgents);
cells_history = cell(Nsim,numAgents);
numAgents_history = zeros(Nsim,1);
verbose = false;
update_interval = 2;

for k = 1:Nsim
    %start = clock;
    fprintf("Computing simulation interation: %04d\n", k);
    % if k == 100
    %     removeIdx = randperm(size(agents,1), round(numAgents/2));
    %     agents(removeIdx,:) = [];
    %     numAgents = size(agents,1);
    % end
    numAgents_history(k) = numAgents;

    agents_history{k} = agents;
    new_agents = zeros(numAgents,3);
    
    if numAgents < 4
        connections = ones(numAgents);
        connections = connections - diag(ones(numAgents,1));
    else
        % --- Delaunay triangulation network ---
        [connections, Laplacian, degree_matrix]  = ....
            Delaunay_triangulation_network(agents);
    end
    % --- Half-plane Voronoi computation ---
    for iD = 1:numAgents
        voronoi_vertices = [];
        tangent_vec = [];

        neighbours = find(connections(iD,:) > 0);
        
        % Include the agent itself
        neigh_idx = [iD neighbours];
        neigh_pos = agents(neigh_idx,:) ./ vecnorm(agents(neigh_idx,:),2,2);
                
        % Clip sphere points by each neighbor's half-plane
        for j = 2:length(neigh_idx)
            p1 = neigh_pos(1,:);        % current agent
            p2 = neigh_pos(j,:);        % neighbor
            
            % Geodesic midpoint
            theta = acos(dot(p1, p2));
            m = (sin((1-0.5)*theta)*p1 + sin(0.5*theta)*p2) / sin(theta);

            n = cross(p1,p2);
            n = n./norm(n);

            tangent = cross(n, m);   % perpendicular to both normal and midpoint
            tangent = tangent / norm(tangent);

            voronoi_vertices(j-1,:) = m;
            tangent_vec(j-1,:) = tangent;
        end
       
        % figure;
        % view(3);
        % axis equal;
        % hold on;
        % [xs,ys,zs] = sphere(60);
        % 
        % surf(xs,ys,zs,'FaceColor','none','EdgeColor','b','EdgeAlpha',0.1)
        % scatter3(voronoi_vertices(:,1), voronoi_vertices(:,2), voronoi_vertices(:,3), 'k');
        % scatter3(neigh_pos(:,1), neigh_pos(:,2), neigh_pos(:,3), 'r', 'filled');
        % 
        % edge_pts = [];
        % for j = 2:length(neigh_idx)
        %     p1 = neigh_pos(1,:);        % current agent
        %     p2 = neigh_pos(j,:);        % neighbor
        % 
        %     arc = geodesic_arc(p1, p2, 50);  % 50 points along the arc
        %     edge_pts = [edge_pts; arc; NaN NaN NaN];  % add NaN to break line between arcs
        % end
        % 
        % plot3(edge_pts(:,1), edge_pts(:,2), edge_pts(:,3), 'k', 'LineWidth', 1.5);
        % 
        % % --- Plot planes perpendicular to tangent plane at Voronoi vertices ---        
        % for j = 1:size(voronoi_vertices,1)
        %     center = voronoi_vertices(j,:);
        %     t = tangent_vec(j,:);
        % 
        %     % Find two vectors perpendicular to t to span the plane
        %     if abs(t(3)) < 0.9
        %         v1 = cross(t, [0 0 1]);
        %     else
        %         v1 = cross(t, [0 1 0]);
        %     end
        %     v1 = v1 / norm(v1);
        %     v2 = cross(t, v1);
        %     v2 = v2 / norm(v2);
        % 
        %     % Create local plane coordinates
        %     [u,v] = meshgrid(linspace(-0.3, 0.3, 10));
        %     X = center(1) + u*v1(1) + v*v2(1);
        %     Y = center(2) + u*v1(2) + v*v2(2);
        %     Z = center(3) + u*v1(3) + v*v2(3);
        % 
        %     % Plot the perpendicular plane
        %     surf(X, Y, Z, 'FaceAlpha', 0.4, 'FaceColor', 'r', 'EdgeColor', 'none');
        % end

        %%
        figure;
        view(3);
        axis equal;
        hold on;
        [xs,ys,zs] = sphere(60);

        surf(xs,ys,zs,'FaceColor','none','EdgeColor','b','EdgeAlpha',0.1)
        scatter3(voronoi_vertices(:,1), voronoi_vertices(:,2), voronoi_vertices(:,3), 'k');
        scatter3(neigh_pos(:,1), neigh_pos(:,2), neigh_pos(:,3), 'r', 'filled');

        edge_pts = [];
        for j = 2:length(neigh_idx)
            p1 = neigh_pos(1,:);        % current agent
            p2 = neigh_pos(j,:);        % neighbor

            arc = geodesic_arc(p1, p2, 50);  % 50 points along the arc
            edge_pts = [edge_pts; arc; NaN NaN NaN];  % add NaN to break line between arcs
        end

        plot3(edge_pts(:,1), edge_pts(:,2), edge_pts(:,3), 'k', 'LineWidth', 1.5);
        
        % Points on sphere (centered at origin)
        [X,Y,Z] = sphere(50);
        P = [X(:), Y(:), Z(:)];
        
        % Initialize logical mask (all true)
        mask = true(size(P,1),1);
        
        % --- Apply all plane constraints ---
        for j = 1:size(voronoi_vertices,1)
            center = voronoi_vertices(j,:);
            t = tangent_vec(j,:);
            
            % Update mask: keep only points satisfying ALL conditions
            mask = mask & ((P - center) * t' < 0);
        end
        
        % Reshape mask to grid
        mask = reshape(mask, size(X));
        
        % Apply mask
        X(~mask) = NaN;
        Y(~mask) = NaN;
        Z(~mask) = NaN;
        
        %surf(X,Y,Z);
        Cx = 0; Cy = 0; Cz = 0;
        total_area = 0;
        
        [nr, nc] = size(X);
        
        for i = 1:nr-1
            for j = 1:nc-1
                
                % Skip if any corner is NaN
                pts = [
                    X(i,j),   Y(i,j),   Z(i,j);
                    X(i+1,j), Y(i+1,j), Z(i+1,j);
                    X(i,j+1), Y(i,j+1), Z(i,j+1);
                    X(i+1,j+1), Y(i+1,j+1), Z(i+1,j+1)
                ];
                
                if any(isnan(pts(:)))
                    continue;
                end
                
                % Triangle 1
                A = pts(1,:);
                B = pts(2,:);
                C = pts(3,:);
                
                area1 = norm(cross(B-A, C-A)) / 2;
                centroid1 = (A + B + C) / 3;
                
                % Triangle 2
                A = pts(4,:);
                B = pts(3,:);
                C = pts(2,:);
                
                area2 = norm(cross(B-A, C-A)) / 2;
                centroid2 = (A + B + C) / 3;
                
                % Accumulate
                Cx = Cx + area1*centroid1(1) + area2*centroid2(1);
                Cy = Cy + area1*centroid1(2) + area2*centroid2(2);
                Cz = Cz + area1*centroid1(3) + area2*centroid2(3);
                
                total_area = total_area + area1 + area2;
            end
        end
        
        centroid = [Cx, Cy, Cz] / total_area;
        new_agents(iD,:) = centroid./norm(centroid);
    end
    agents = new_agents;
    %times(k) = etime(clock, start);
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

[xs,ys,zs] = sphere(50);
surf(xs,ys,zs,'FaceAlpha',0.05,'EdgeColor','none');

h = scatter3([],[],[],50,'filled');

for k = 1:Nsim
    A = agents_history{k};
    
    set(h, 'XData', A(:,1), 'YData', A(:,2), 'ZData', A(:,3));
    title(['Time step: ', num2str(k)]);
    
    drawnow;
    pause(0.05);
end

%%
figure;
hold on;

% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(R*XS, R*YS, R*ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]); 

% Plot points
A = agents_history{Nsim};
scatter3(A(:,1), A(:,2), A(:,3), 50, 'k', 'filled');

% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure

view(3);
print2pdf("Lloyd_half_plane_normal.pdf", 1);
view(90,0);
print2pdf("Lloyd_half_plane_side.pdf", 1);
view(0,90);
print2pdf("Lloyd_half_plane_top.pdf", 1);