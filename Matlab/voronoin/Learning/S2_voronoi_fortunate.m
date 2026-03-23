clear; clc; close all;

Tend = 3;
dTi = 0.01;
Nsim = round(Tend/dTi);

numAgents = 25;

agents = randn(numAgents,3);
agents = agents ./ vecnorm(agents,2,2);

agents_history = zeros(numAgents,3,Nsim);
cells_history = cell(Nsim,numAgents);

for k = 1:Nsim
    agents_history(:,:,k) = agents;

    K = convhull(agents);
    numFaces = size(K,1);

    voronoi_vertices = zeros(numFaces,3);

    % Voronoi vertices
    for i = 1:numFaces
        p = agents(K(i,:),:);
        v = cross(p(2,:) - p(1,:), p(3,:) - p(1,:));
        voronoi_vertices(i,:) = v / norm(v);
    end

    new_agents = zeros(numAgents,3);

    % Cells and centroids
    for i = 1:numAgents

        faceIdx = any(K==i,2);
        cell_verts = voronoi_vertices(faceIdx,:);

        center = agents(i,:);

        [~,~,V] = svd(cell_verts - center);
        proj = (cell_verts - center)*V(:,1:2);

        angles = atan2(proj(:,2),proj(:,1));
        [~,idx] = sort(angles);

        cell_verts = cell_verts(idx,:);

        cells_history{k,i} = cell_verts;

        total_area = 0;
        centroid_sum = [0 0 0];
        
        v0 = cell_verts(1,:);
        
        for j = 2:size(cell_verts,1)-1
            a = v0;
            b = cell_verts(j,:);
            c = cell_verts(j+1,:);
        
            A = spherical_triangle_area(a,b,c);
        
            tri_centroid = (a + b + c);
            tri_centroid = tri_centroid / norm(tri_centroid);
        
            centroid_sum = centroid_sum + A * tri_centroid;
            total_area = total_area + A;
        end
        
        true_centroid = centroid_sum / norm(centroid_sum);
        
        new_agents(i,:) = true_centroid;

    end

    agents = new_agents;
end

%%
%% Visualization (Geodesic Voronoi on S2)

figure('Color','w');
axis equal
view(3)
grid off
hold on
title_handle = title('');

% Sphere
[xs,ys,zs] = sphere(60);
surf(xs,ys,zs,'FaceColor','none','EdgeColor','b','EdgeAlpha',0.1)

% Agent points
hAgents = plot3(agents_history(:,1,1),...
                agents_history(:,2,1),...
                agents_history(:,3,1),...
                'k.','MarkerSize',25);

% Line handles
hEdges = gobjects(numAgents,1);

% ---- Initial draw ----
for i = 1:numAgents
    verts = cells_history{1,i};

    edge_pts = [];

    for j = 1:size(verts,1)
        p1 = verts(j,:);
        p2 = verts(mod(j,size(verts,1))+1,:);

        % geodesic interpolation
        omega = acos(dot(p1,p2));
        if abs(omega) < 1e-6
            arc = repmat(p1,20,1);
        else
            t = linspace(0,1,20)';
            arc = (sin((1-t)*omega).*p1 + sin(t*omega).*p2) / sin(omega);
        end

        edge_pts = [edge_pts; arc; NaN NaN NaN];
    end

    hEdges(i) = plot3(edge_pts(:,1),...
                      edge_pts(:,2),...
                      edge_pts(:,3),...
                      'LineWidth',1.5);
end

% ---- Animation loop ----
for k = 1:Nsim

    % Update agents
    set(hAgents,...
        'XData',agents_history(:,1,k),...
        'YData',agents_history(:,2,k),...
        'ZData',agents_history(:,3,k));

    set(title_handle,'String',...
        sprintf('Voronoi S2 (Iter %d)',k));

    % Update geodesic edges
    for i = 1:numAgents

        verts = cells_history{k,i};
        edge_pts = [];

        for j = 1:size(verts,1)
            p1 = verts(j,:);
            p2 = verts(mod(j,size(verts,1))+1,:);
            arc = geodesic_arc(p1,p2,20);
            edge_pts = [edge_pts; arc; NaN NaN NaN];
        end

        set(hEdges(i),...
            'XData',edge_pts(:,1),...
            'YData',edge_pts(:,2),...
            'ZData',edge_pts(:,3));
    end

    drawnow;
end


