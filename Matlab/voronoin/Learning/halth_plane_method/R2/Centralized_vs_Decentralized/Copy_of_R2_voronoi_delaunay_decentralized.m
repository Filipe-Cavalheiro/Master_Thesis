% VORONOI_SMALLWORLD_CONSENSUS - Agents move to Voronoi centers while
% maintaining small-world communication for consensus on their values
clear; clc; close all;

% PARAMETERS
numAgents = 30;
world_X = 10;
world_Y = 10;
Tend = 2;           % seconds
dTi = 0.01;         % inner-loop and simulation sampling period
Nsim = round(Tend/dTi)+1;

% INITIALIZATION
% Random initialization of agents
agents = load("agent_position.mat");
agents = agents.agents;

domain = polyshape([0 world_X world_X 0],[0 0 world_Y world_Y]);

% Store agents positions over time
agents_history = cell(Nsim+1,numAgents);
agents_history{1} = agents;

numAgents_history = zeros(Nsim,1);

% Store cell information for each iteration
cells_history = cell(Nsim, numAgents);
cell_area_history = cell(Nsim, numAgents);

% Store connection history for visualization
connections_history = cell(Nsim, 1);

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

maxAgents = max(numAgents_history);

% Voronoi patches
vor_patches = gobjects(maxAgents,1);

for i = 1:maxAgents
    vor_patches(i) = patch(ax1,NaN,NaN,[0.85 0.9 1], ...
        'EdgeColor','b', ...
        'LineWidth',0.5);
end

title_handle = title(ax1,'');

% Agent scatter
agent_scatter = scatter(ax1,NaN,NaN,80,'r','filled');

% Animation loop
for k = 1:Nsim
    agents = agents_history{k};
    cells = cells_history(k,:);
    connections = connections_history{k};

    % Update Voronoi cells
    for i = 1:length(cells)

        if isgraphics(vor_patches(i))

            if ~isempty(cells{i})
                v = cells{i}.Vertices;

                set(vor_patches(i), ...
                    'XData',v(:,1), ...
                    'YData',v(:,2));

            else
                set(vor_patches(i), ...
                    'XData',NaN, ...
                    'YData',NaN);
            end
        end
    end

    % Update title
    set(title_handle,'String', ...
        sprintf('Voronoi Relaxation + Small-World (Iter %d)',k));

    % Update agents
    set(agent_scatter, ...
        'XData',agents(:,1), ...
        'YData',agents(:,2));

    drawnow
end