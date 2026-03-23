clear; clc; close all;

% PARAMETERS
numAgents = 15;
iterations = 1000;

% Random points on the unit sphere
agents = randn(numAgents,3);
agents = agents ./ vecnorm(agents,2,2);

% store positions for animation
history = zeros(numAgents,3,iterations);

% ================= COMPUTATION =================
for k = 1:iterations

    % Compute convex hull
    K = convhull(agents(:,1),agents(:,2),agents(:,3));

    % Voronoi vertices (face normals)
    numFaces = size(K,1);
    V = zeros(numFaces,3);

    for i = 1:numFaces
        p1 = agents(K(i,1),:);
        p2 = agents(K(i,2),:);
        p3 = agents(K(i,3),:);

        n = cross(p2-p1, p3-p1);
        n = n / norm(n);

        V(i,:) = n;
    end

    newAgents = zeros(size(agents));

    % Compute centroids
    for i = 1:numAgents

        faces = any(K==i,2);
        verts = V(faces,:);

        if isempty(verts)
            newAgents(i,:) = agents(i,:);
            continue
        end

        c = mean(verts,1);
        c = c / norm(c); % project to sphere

        newAgents(i,:) = c;

    end

    agents = newAgents;

    % store for animation
    history(:,:,k) = agents;

end

%% ================= ANIMATION =================
figure;
view(3);
grid on;
axis equal;
hold on;

% draw sphere
[xs,ys,zs] = sphere(50);
surf(xs,ys,zs,'FaceAlpha',0.1,'EdgeColor','none')

% initial points
pts = history(:,:,1);

points = scatter3(pts(:,1),pts(:,2),pts(:,3),60,'r','filled');

for k = 1:iterations

    pts = history(:,:,k);

    set(points, ...
        'XData', pts(:,1), ...
        'YData', pts(:,2), ...
        'ZData', pts(:,3));

    title(['Iteration ', num2str(k)])
    drawnow;
    pause(0.01);
end