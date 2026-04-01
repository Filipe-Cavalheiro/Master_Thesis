clear; clc; close all;

% PARAMETERS
numAgents = 15;
iterations = 1000;
removeAt = iterations/2; % iteration to remove agents
numRemove = 5;           % number of agents to remove

% Random points on the unit sphere
agents = randn(numAgents,3);
agents = agents ./ vecnorm(agents,2,2);

% store positions for animation (maximum possible size)
history = cell(iterations,1);

% ================= COMPUTATION =================
for k = 1:iterations

    % Remove agents halfway
    if k == removeAt
        removeIdx = randperm(numAgents, numRemove);
        agents(removeIdx,:) = [];
        numAgents = size(agents,1);
    end

    % Compute convex hull
    if numAgents >= 4 % Convex hull requires at least 4 points in 3D
        K = convhull(agents(:,1),agents(:,2),agents(:,3));
    else
        K = [];
    end

    % Voronoi vertices (face normals)
    if ~isempty(K)
        numFaces = size(K,1);
        V = zeros(numFaces,3);

        for iFace = 1:numFaces
            p1 = agents(K(iFace,1),:);
            p2 = agents(K(iFace,2),:);
            p3 = agents(K(iFace,3),:);

            n = cross(p2-p1, p3-p1);
            n = n / norm(n);

            V(iFace,:) = n;
        end
    end

    newAgents = zeros(size(agents));

    % Compute centroids
    for i = 1:numAgents

        if isempty(K)
            newAgents(i,:) = agents(i,:);
            continue
        end

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
    history{k} = agents;

end

%% ================= ANIMATION =================
figure('Position',[100 100 700 600]);
view(3);
grid on;
axis equal;
hold on;

% draw sphere
[xs,ys,zs] = sphere(50);
surf(xs,ys,zs,'FaceAlpha',0.1,'EdgeColor','none')

% initial points
pts = history{1};
points = scatter3(pts(:,1),pts(:,2),pts(:,3),60,'r','filled');

% create UI label
agentLabel = uicontrol('Style','text', ...
                       'String', ['Agents: ', num2str(size(pts,1))], ...
                       'FontSize',12, 'FontWeight','bold', ...
                       'Position',[20 20 120 30]); % [left bottom width height]

for k = 1:iterations
    pts = history{k};

    set(points, ...
        'XData', pts(:,1), ...
        'YData', pts(:,2), ...
        'ZData', pts(:,3));

    % update label
    set(agentLabel, 'String', ['Agents: ', num2str(size(pts,1))]);

    title(['Iteration ', num2str(k)])
    drawnow;
    pause(0.01);
end