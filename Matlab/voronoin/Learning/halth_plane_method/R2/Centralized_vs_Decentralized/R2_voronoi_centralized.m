clear; clc; close all;

% PARAMETERS

numAgents = 8;

world_X = 10;
world_Y = 10;

Tend = 2;
dTi  = 0.01;

Nsim = round(Tend/dTi);

% DOMAIN

domain = polyshape( ...
    [0 world_X world_X 0], ...
    [0 0 world_Y world_Y]);

% INITIAL CONDITIONS

S = load("agent_position.mat");
agents = S.agents(1:numAgents,:);

% HISTORY

agents_history = cell(Nsim+1,1);
cells_history  = cell(Nsim,numAgents);

agents_history{1} = agents;

% ARTIFICIAL BOUNDARY POINTS
% These force bounded Voronoi cells

buffer = max(world_X,world_Y);

boundaryPts = [
    -buffer -buffer
     world_X/2 -buffer
     world_X+buffer -buffer

     world_X+buffer world_Y/2

     world_X+buffer world_Y+buffer
     world_X/2      world_Y+buffer
    -buffer         world_Y+buffer

    -buffer world_Y/2
];

% LLOYD ITERATIONS

fprintf('Running Lloyd relaxation...\n');

for k = 1:Nsim

    % Add boundary points

    allPts = [agents; boundaryPts];

    % Voronoi tessellation

    dt = delaunayTriangulation(allPts);

    [V,R] = voronoiDiagram(dt);

    new_agents = agents;

    % Process only REAL agents
    for i = 1:numAgents

        region = R{i};

        verts = V(region,:);

        % Remove infinite vertices if any remain

        finiteMask = all(isfinite(verts),2);
        verts = verts(finiteMask,:);

        if size(verts,1) < 3
            continue
        end

        % Create polygon

        cell_poly = polyshape(verts);

        % Clip to domain

        cell_poly = intersect(cell_poly,domain);

        % Store for animation

        cells_history{k,i} = cell_poly;

        % Compute centroid

        if ~isempty(cell_poly.Vertices)

            [cx,cy] = centroid(cell_poly);

            new_agents(i,:) = [cx cy];
        end
    end

    % Update agents

    agents = new_agents;

    agents_history{k+1} = agents;

    if mod(k,20)==0
        fprintf('Iteration %d / %d\n',k,Nsim);
    end
end

fprintf('Simulation complete\n');

% ============================================================
% PREALLOCATE TRAJECTORIES
% ============================================================

trajX = zeros(Nsim+1,numAgents);
trajY = zeros(Nsim+1,numAgents);

for k = 1:Nsim+1

    A = agents_history{k};

    trajX(k,:) = A(:,1)';
    trajY(k,:) = A(:,2)';
end

% ============================================================
% ANIMATION
% ============================================================

figure( ...
    'units','normalized', ...
    'outerposition',[0 0 1 1]);

ax = axes;

hold(ax,'on');
axis(ax,'equal');

xlim(ax,[0 world_X]);
ylim(ax,[0 world_Y]);

box(ax,'on');

xlabel(ax,'x');
ylabel(ax,'y');

% Plot domain
plot(ax,domain, ...
    'FaceColor','none', ...
    'EdgeColor','k', ...
    'LineWidth',2);

% Initial positions
initial_agents = agents_history{1};

colors = lines(numAgents);

cell_plots    = gobjects(numAgents,1);
traj_lines    = gobjects(numAgents,1);
agent_scatter = gobjects(numAgents,1);

for i = 1:numAgents

    % Voronoi polygons
    cell_plots(i) = plot( ...
        polyshape(), ...
        'FaceColor',colors(i,:), ...
        'FaceAlpha',0.15, ...
        'EdgeColor',colors(i,:), ...
        'LineWidth',1.2);

    % Agent marker
    agent_scatter(i) = scatter(ax, ...
        initial_agents(i,1), ...
        initial_agents(i,2), ...
        80, ...
        colors(i,:), ...
        'filled');

    % Trajectory
    traj_lines(i) = plot(ax, ...
        trajX(1,i), ...
        trajY(1,i), ...
        'Color',colors(i,:), ...
        'LineWidth',1.5);
end

% Animate
fprintf('Animating...\n');

for k = 1:Nsim

    agents = agents_history{k};

    % Update Voronoi cells
    for i = 1:numAgents

        pgon = cells_history{k,i};

        if isempty(pgon)
            continue
        end

        set(cell_plots(i), 'Shape', pgon);
    end

    % Update agents + trajectories
    for i = 1:numAgents

        set(agent_scatter(i), ...
            'XData', agents(i,1), ...
            'YData', agents(i,2));

        set(traj_lines(i), ...
            'XData', trajX(1:k,i), ...
            'YData', trajY(1:k,i));
    end

    drawnow;
end

fprintf('Done\n');

print2pdf("voronoi_centralized_comparison.pdf",1);