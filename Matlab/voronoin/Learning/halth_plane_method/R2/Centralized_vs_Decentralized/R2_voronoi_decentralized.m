% VORONOI_SMALLWORLD_CONSENSUS - Agents move to Voronoi centers while
% maintaining small-world communication for consensus on their values
clear; clc; close all;

% PARAMETERS
numAgents = 8;
world_X = 10;
world_Y = 10;
Tend = 2;           % seconds
dTi = 0.01;         % inner-loop and simulation sampling period
Nsim = round(Tend/dTi)+1;

% INITIALIZATION
% Random initialization of agents
agents = load("agent_position.mat");
agents = agents.agents(1:numAgents,:);

domain = polyshape([0 world_X world_X 0],[0 0 world_Y world_Y]);

% Store agents positions over time
agents_history = cell(Nsim+1,numAgents);
agents_history{1} = agents;

numAgents_history = zeros(Nsim,1);

% Store cell information for each iteration
cells_history = cell(Nsim, numAgents);
cell_area_history = cell(Nsim, numAgents);

% COMPUTATION
fprintf('Computing Voronoi relaxation with small-world consensus...\n');
for k = 1:Nsim 
    numAgents_history(k) = numAgents;

    % --- SMALL-WORLD CONSENSUS STEP ---
    % Update connections based on current positions
    %if mod(k, update_interval) == 1 || k == 1
    [connections, Laplacian, degree_matrix]  = Delaunay_triangulation_network(agents);
    %end
    connections_history{k} = connections;
        
    % Voronoi
    voronoi_centers = agents;

    % Find neighbors of agent i (where connections(i,:) is non-zero)
    % as well it self
    for current_agent = 1:numAgents
        
        p_i = agents(current_agent,:);
        cell_poly = domain;
        
        neighbors = find(connections(current_agent,:) > 0);
    
        for j = neighbors
            
            p_j = agents(j,:);
            
            mid = (p_i + p_j)/2;
            n = p_j - p_i;
            
            L = world_X*world_Y*10;
            
            v = [-n(2) n(1)];
            v = v / norm(v);
            
            p1 = mid + L*v;
            p2 = mid - L*v;
            
            hp = polyshape([p1(1) p2(1) p2(1)-n(1)*L p1(1)-n(1)*L],...
                           [p1(2) p2(2) p2(2)-n(2)*L p1(2)-n(2)*L]);
            
            cell_poly = intersect(cell_poly,hp);
        end
        
        if ~isempty(cell_poly.Vertices)
            cell_area_history{k, current_agent} = area(cell_poly);
            cells_history{k,current_agent} = cell_poly;
            [cx,cy] = centroid(cell_poly);
            voronoi_centers(current_agent,:) = [cx cy];
        end
    end
    
    % --- MOVE AGENTS ---
    % Here we simply update positions to the new Voronoi centers
    agents = voronoi_centers;
    
    % Store history
    agents_history{k+1} = agents;
    
    % Progress indicator
    if mod(k, 10) == 0
        fprintf('Iteration %d/%d complete\n', k, Nsim);
    end
end

fprintf('Computation complete. Starting animation...\n');

%% ANIMATION - Combined view
figure('units','normalized','outerposition',[0 0 1 1])

% Create subplot ONCE
ax1 = subplot(2,2,1:2);

hold(ax1,'on');
axis(ax1,'equal')

xlim(ax1,[0 world_X])
ylim(ax1,[0 world_Y])

plot(ax1,domain,'FaceColor','none','LineWidth',2,'EdgeColor','k')

% Initial positions
initial_agents = agents_history{1};
numAgents = size(initial_agents,1);

% Plot initial positions as crosses
scatter(ax1, ...
    initial_agents(:,1), ...
    initial_agents(:,2), ...
    100,'kx','LineWidth',2);

% Current agent positions
agent_scatter = scatter(ax1,NaN,NaN,80,'r','filled');

% Store trajectory history
trajX = cell(numAgents,1);
trajY = cell(numAgents,1);

% Create trajectory lines
traj_lines = gobjects(numAgents,1);

for i = 1:numAgents
    trajX{i} = initial_agents(i,1);
    trajY{i} = initial_agents(i,2);

    traj_lines(i) = plot(ax1, ...
        trajX{i}, trajY{i}, ...
        'b-','LineWidth',1.5);
end

% Animation loop
for k = 1:Nsim

    agents = agents_history{k};

    % Update current positions
    set(agent_scatter, ...
        'XData',agents(:,1), ...
        'YData',agents(:,2));

    % Append trajectory history
    for i = 1:numAgents

        trajX{i}(end+1) = agents(i,1);
        trajY{i}(end+1) = agents(i,2);

        set(traj_lines(i), ...
            'XData',trajX{i}, ...
            'YData',trajY{i});
    end

    drawnow
end

print2pdf("voronoi_decentralized_comparison.pdf",1);