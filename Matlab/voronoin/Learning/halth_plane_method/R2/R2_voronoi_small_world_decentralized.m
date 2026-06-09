% VORONOI_SMALLWORLD_CONSENSUS - Agents move to Voronoi centers while
% maintaining small-world communication for consensus on their values
clear; clc; close all;

% PARAMETERS
numAgents = 9;
world_X = 10;
world_Y = 10;
Tend = 2;           % seconds
dTi = 0.01;         % inner-loop and simulation sampling period
Nsim = round(Tend/dTi)+1;

% Small-world network parameters
k = 4;                  % Number of nearest neighbors (initial local connections)
gamma = 0.2;            % Rewiring probability (0 = only local, 1 = fully random)
comm_range = 1;         % Maximum communication distance (relative to domain size)
update_interval = 10;    % Update connections every N steps
k_neighbours = 8;

% INITIALIZATION
% Random initialization of agents
agents = [world_X*rand(numAgents,1), world_Y*rand(numAgents,1)];

% Random initial values for consensus
agent_values = rand(numAgents, 1);

domain = polyshape([0 world_X world_X 0],[0 0 world_Y world_Y]);

% Store agents positions over time
agents_history = cell(Nsim+1,numAgents);
agents_history{1} = agents;

% Store agent values over time
values_history = cell(Nsim+1,numAgents);
values_history{1} = agent_values;

numAgents_history = zeros(Nsim,1);

% Store cell information for each iteration
cells_history = cell(Nsim, numAgents);
cell_area_history = cell(Nsim, numAgents);

% Store connection history for visualization
connections_history = cell(Nsim, 1);

% COMPUTATION
fprintf('Computing Voronoi relaxation with small-world consensus...\n');
for k = 1:Nsim 

    if k == 50
        removeIdx = randperm(size(agents,1), round(numAgents/3));
        agents(removeIdx,:) = [];
        agent_values(removeIdx,:) = [];
        numAgents = size(agents,1);  % update agent count
    end
    numAgents_history(k) = numAgents;

    % --- SMALL-WORLD CONSENSUS STEP ---
    % Update connections based on current positions
    %if mod(k, update_interval) == 1 || k == 1
    [connections, Laplacian, degree_matrix]  = build_small_world_network(agents, k_neighbours, gamma, comm_range);
    %end
    connections_history{k} = connections;
    
    % Run consensus on agent values (only with connected neighbors)
    agent_values = consensus_step(agent_values, Laplacian);
    
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
    values_history{k+1} = agent_values;
    
    % Progress indicator
    if mod(k, 10) == 0
        fprintf('Iteration %d/%d complete\n', k, Nsim);
    end
end

fprintf('Computation complete. Starting animation...\n');

%% ANIMATION - Combined view
figure('units','normalized','outerposition',[0 0 1 1])

% TOP PANEL
subplot(2,2,1:2)
hold on;
axis equal
xlim([0 world_X])
ylim([0 world_Y])

plot(domain,'FaceColor','none','LineWidth',2,'EdgeColor','k')

maxAgents = max(numAgents_history);

% Voronoi patches
vor_patches = gobjects(maxAgents,1);
for i=1:maxAgents
    vor_patches(i) = patch(NaN,NaN,[0.85 0.9 1], ...
        'EdgeColor','b','LineWidth',0.5);
end

title_handle = title('');

% Network lines (preallocate handles)
net_lines = gobjects(maxAgents,maxAgents);

for i=1:maxAgents
    for j=i+1:maxAgents
        net_lines(i,j) = plot(NaN,NaN,'Color',[0.4 0.8 0.4],'LineWidth',1);
    end
end

% Agent scatter
agent_scatter = scatter(NaN,NaN,80,'r','filled');

% BOTTOM LEFT PANEL
subplot(2,2,3)
hold on;

area_lines = gobjects(maxAgents,1);

for i=1:maxAgents
    area_lines(i) = plot(NaN,NaN,'LineWidth',1.5);
end

xlim([0 Nsim])
ylim([0 100])
xlabel('Iteration')
ylabel('Value')
title('Area Progress')

% BOTTOM RIGHT PANEL
subplot(2,2,4)
hold on;

traj_lines = gobjects(maxAgents,1);

for i=1:maxAgents
    traj_lines(i) = plot(NaN,NaN,'LineWidth',1.5);
end

xlim([0 Nsim])
ylim([0 1])
xlabel('Iteration')
ylabel('Value')
title('Consensus Progress')

for k = 1:Nsim

    agents = agents_history{k};
    cells  = cells_history(k,:);
    connections = connections_history{k};

    % UPDATE TOP PANEL
    subplot(2,2,1:2)

    % Update Voronoi cells
    for i=1:length(cells)
        if ~isempty(cells{i})
            v = cells{i}.Vertices;
            set(vor_patches(i),'XData',v(:,1),'YData',v(:,2));
        else
            set(vor_patches(i),'XData',NaN,'YData',NaN);
        end
    end

    % Reset all network lines
    for i = 1:maxAgents
        for j = i+1:maxAgents
            set(net_lines(i,j),'XData',NaN,'YData',NaN);
        end
    end

    % Update network connections
    for i=1:size(connections,1)
        for j=i+1:size(connections,2)
            if connections(i,j)
                set(net_lines(i,j), ...
                    'XData',[agents(i,1) agents(j,1)], ...
                    'YData',[agents(i,2) agents(j,2)]);
            else
                set(net_lines(i,j),'XData',NaN,'YData',NaN);
            end

        end
    end

    set(title_handle,'String',...
        sprintf('Voronoi Relaxation + Small-World (Iter %d)',k));

    % Update agents
    set(agent_scatter,'XData',agents(:,1),'YData',agents(:,2));

    % UPDATE BOTTOM LEFT PANEL
    subplot(2,2,3)
    
    for i = 1:maxAgents
    
        traj = NaN(k,1);
    
        for t = 1:k
            if ~isempty(cell_area_history{t,i})
                traj(t) = cell_area_history{t,i};
            end
        end
    
        set(area_lines(i),'XData',0:k-1,'YData',traj);
    
    end
        
    % UPDATE BOTTOM RIGHT PANEL
    subplot(2,2,4)

    for i=1:maxAgents

        traj = NaN(k,1);

        for t=1:k
            if ~isempty(values_history{t}) && length(values_history{t})>=i
                traj(t) = values_history{t}(i);
            end
        end

        set(traj_lines(i),'XData',0:k-1,'YData',traj);

    end

    drawnow limitrate
    pause(0.05);
end

%% ANIMATION - Combined view
figure('units','normalized','outerposition',[0 0 1 1])

% TOP PANEL
hold on;
axis equal
xlim([0 world_X])
ylim([0 world_Y])

plot(domain,'FaceColor','none','LineWidth',2,'EdgeColor','k')

maxAgents = max(numAgents_history);

% Voronoi patches
vor_patches = gobjects(maxAgents,1);
for i=1:maxAgents
    vor_patches(i) = patch(NaN,NaN,[0.85 0.9 1], ...
        'EdgeColor','b','LineWidth',0.5);
end

%title_handle = title('');

% Network lines (preallocate handles)
net_lines = gobjects(maxAgents,maxAgents);

for i=1:maxAgents
    for j=i+1:maxAgents
        net_lines(i,j) = plot(NaN,NaN,'Color',[0.4 0.8 0.4],'LineWidth',1);
    end
end

% Agent scatter
agent_scatter = scatter(NaN,NaN,80,'r','filled');

for k = 1:Nsim

    agents = agents_history{k};
    cells  = cells_history(k,:);
    connections = connections_history{k};

    % Update Voronoi cells
    for i=1:length(cells)
        if ~isempty(cells{i})
            v = cells{i}.Vertices;
            set(vor_patches(i),'XData',v(:,1),'YData',v(:,2));
        else
            set(vor_patches(i),'XData',NaN,'YData',NaN);
        end
    end

    % Reset all network lines
    for i = 1:maxAgents
        for j = i+1:maxAgents
            set(net_lines(i,j),'XData',NaN,'YData',NaN);
        end
    end

    % Update network connections
    for i=1:size(connections,1)
        for j=i+1:size(connections,2)
            if connections(i,j)
                set(net_lines(i,j), ...
                    'XData',[agents(i,1) agents(j,1)], ...
                    'YData',[agents(i,2) agents(j,2)]);
            else
                set(net_lines(i,j),'XData',NaN,'YData',NaN);
            end

        end
    end

    % set(title_handle,'String',...
    %     sprintf('Voronoi Relaxation + Small-World (Iter %d)',k));

    % Update agents
    set(agent_scatter,'XData',agents(:,1),'YData',agents(:,2));

    drawnow limitrate
    pause(0.05);
end