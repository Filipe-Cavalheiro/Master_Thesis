clear; clc; close all;

Tend = 1.5;
dTi = 0.01;
Nsim = round(Tend/dTi);

numAgents = 4;
R = 5.0;

% Small-world parameters
k_neighbours = 20;
gamma = 0.2;
comm_range = 1;

agents_unit = randn(numAgents,3);
agents_unit = agents_unit ./ vecnorm(agents_unit,2,2);
agents = agents_unit*R;
 
agents_history = cell(Nsim+1,numAgents);
cells_history = cell(Nsim,numAgents);
numAgents_history = zeros(Nsim,1);

for k = 1:3

    if k == 100
        removeIdx = randperm(size(agents,1), round(numAgents/2));
        agents(removeIdx,:) = [];
        numAgents = size(agents,1);
    end
    numAgents_history(k) = numAgents;

    agents_history{k} = agents;
    new_agents = zeros(numAgents,3);

    % --- SMALL WORLD NETWORK ---
    [connections, Laplacian, degree_matrix]  = build_small_world_network(...
        agents, k_neighbours, gamma, comm_range);
    
    % Cells and centroids
    for i = 1:numAgents
        neighbours = find(connections(i,:) > 0);
        
        % --- build local point set ---
        neigh_idx = [i neighbours];
        neigh_pos = agents(neigh_idx,:);
        neigh_pos = neigh_pos ./ vecnorm(neigh_pos, 2, 2);
        
        % compute convex hull
        K = convhull(neigh_pos);
        
        % % --- plot points ---
        % figure;
        % hold on;
        % axis equal;
        % view(3);
        % 
        % [xs,ys,zs] = sphere(60);
        % surf(xs,ys,zs,'FaceColor','none','EdgeColor','b','EdgeAlpha',0.1)
        % 
        % %plot all neighbors
        % plot3(neigh_pos(:,1), neigh_pos(:,2), neigh_pos(:,3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        % 
        % %--- annotate points with their index ---
        % for idx = 1:size(neigh_pos,1)
        %     text(neigh_pos(idx,1), neigh_pos(idx,2), neigh_pos(idx,3), sprintf('%d', idx), ...
        %         'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k', 'HorizontalAlignment', 'right');
        % end
        % 
        % %--- plot convex hull ---
        % trisurf(K, neigh_pos(:,1), neigh_pos(:,2), neigh_pos(:,3), ...
        %     'FaceAlpha', 0.3, 'EdgeColor', 'k');
        
        % --- compute outward-pointing normals for faces including the agent ---
        voronoi_vertices = [];
        
        for j = 1:size(K,1)
            if any(K(j,:) == 1) % only faces including the agent
                p = neigh_pos(K(j,:),:);
                
                % normal vector
                v = cross(p(2,:) - p(1,:), p(3,:) - p(1,:));
                v = v / norm(v);
                
                % Only keep faces whose normals point outward from the sphere 
                % the other ones are "inside" planes 
                center_shifted = agents(i,:)./vecnorm(agents(i,:), 2, 2);
                if dot(v, center_shifted) <= 0 
                    v = -v; % flip outward
                end

                voronoi_vertices = [voronoi_vertices; v]; 
            
            end
        end
        
        %perpendicular vector to agent, tangent to sphere surface
        center = agents(i,:) ./ vecnorm(agents(i,:), 2, 2);

        [~,~,V] = svd(voronoi_vertices - center);
        proj = (voronoi_vertices - center)*V(:,1:2);

        % this reorganization is done for graphical viewing purposes
        angles = atan2(proj(:,2),proj(:,1));
        [~,idx] = sort(angles);

        voronoi_vertices = voronoi_vertices(idx,:);

        cells_history{k,i} = voronoi_vertices;

        centroid_sum = [0 0 0];
        
        v0 = voronoi_vertices(1,:);

         for j = 2:size(voronoi_vertices,1)-1
            a = v0;
            b = voronoi_vertices(j,:);
            c = voronoi_vertices(j+1,:);
            
            % Spherical Excess Formula (L'Huilier formula)
            A = spherical_triangle_area(a,b,c);
            
            % Area is calculated for varying R no need to also scale this
            tri_centroid_dir = (a + b + c);
            tri_centroid_dir = tri_centroid_dir / norm(tri_centroid_dir);
        
            centroid_sum = centroid_sum + A * tri_centroid_dir;
        end
        
        true_centroid = R*centroid_sum ./ vecnorm(centroid_sum, 2, 2);
       
        new_agents(i,:) = true_centroid;
        %hold off;
    end
    agents = new_agents;
end

%%
maxAgents = numAgents_history(1);
figure('Color','w');
axis equal
view(3)
grid off
hold on
title_handle = title('');

[xs,ys,zs] = sphere(60);
surf(R*xs,R*ys,R*zs,'FaceColor','none','EdgeColor','b','EdgeAlpha',0.1)

colors = lines(maxAgents);

% agent plot
plot3(NaN,NaN,NaN,'Color',[0.4 0.8 0.4],'LineWidth',1);
hAgents = plot3(agents_history{1},agents_history{1},agents_history{1},...
    'k.','MarkerSize',25);

% ---- Initial draw ----
    for i = 1:maxAgents
        hEdges(i) = plot3(NaN,NaN,NaN,'Color',colors(i,:),'LineWidth',1.5);
    end
    
for k = 1:1
    numAgents = numAgents_history(k);
    agents = agents_history{k};

    if (k > 1) && (numAgents_history(k) ~= numAgents_history(k-1))
        % Reset all cells
        for i = 1:maxAgents
           set(hEdges(i),'XData',NaN,'YData',NaN, 'ZData', NaN);
        end
    end
   
    % update agents
    set(hAgents,'XData',agents(:,1),...
        'YData',agents(:,2),...
        'ZData',agents(:,3));

    % set(title_handle,'String',...
    %     sprintf('Voronoi S2 (Iter %d)',k));

    % Update geodesic edges
    for i = 1:numAgents

        verts = cells_history{k,i};
        edge_pts = [];

        for j = 1:size(verts,1)
            p1 = verts(j,:);
            p2 = verts(mod(j,size(verts,1))+1,:);
            arc = geodesic_arc(p1,p2,20, R);
            edge_pts = [edge_pts; arc];
        end

        set(hEdges(i),...
            'XData',edge_pts(:,1),...
            'YData',edge_pts(:,2),...
            'ZData',edge_pts(:,3));
    end

    drawnow;
    pause(0.1);
end