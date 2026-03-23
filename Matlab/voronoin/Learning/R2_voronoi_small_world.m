% VORONOI_SMALLWORLD_CONSENSUS - Agents move to Voronoi centers while
% maintaining small-world communication for consensus on their values
clear; clc; close all;

% PARAMETERS
numAgents = 9;
world_X = 10;
world_Y = 10;
Tend = 1;           % seconds
dTi = 0.01;         % inner-loop and simulation sampling period
Nsim = round(Tend/dTi)+1;

% Small-world network parameters
k = 4;                  % Number of nearest neighbors (initial local connections)
gamma = 0.2;            % Rewiring probability (0 = only local, 1 = fully random)
comm_range = 0;         % Maximum communication distance (relative to domain size)
update_interval = 10;    % Update connections every N steps
k_neighbours = 4;

% INITIALIZATION
% Random initialization of agents
agents = [world_X*rand(numAgents,1), world_Y*rand(numAgents,1)];

% Random initial values for consensus
agent_values = rand(numAgents, 1);

domain = polyshape([0 world_X world_X 0],[0 0 world_Y world_Y]);

% Store agents positions over time
agents_history = zeros(Nsim+1, numAgents, 2);
agents_history(1,:,:) = agents;

% Store agent values over time
values_history = zeros(Nsim+1, numAgents);
values_history(1,:) = agent_values;

% Store cell information for each iteration
cells_history = cell(Nsim, numAgents);

% Store connection history for visualization
connections_history = cell(Nsim, 1);

% COMPUTATION
fprintf('Computing Voronoi relaxation with small-world consensus...\n');
for k = 1:Nsim 
    % --- SMALL-WORLD CONSENSUS STEP ---
    % Update connections based on current positions
    if mod(k, update_interval) == 1 || k == 1
        [connections, Laplacian, degree_matrix]  = build_small_world_network(agents, k_neighbours, gamma, comm_range);
    end
    connections_history{k} = connections;
    
    % Run consensus on agent values (only with connected neighbors)
    agent_values = consensus_step(agent_values, Laplacian);
    
    % Voronoi
    voronoi_centers = zeros(numAgents,2);
    
    for i = 1:numAgents
        cell_poly = domain;
        p_i = agents(i,:);
        
        for j = 1:numAgents
            
            if j == i
                continue
            end
            
            p_j = agents(j,:);
            
            mid = (p_i + p_j)/2;
            n = p_j - p_i;
            
            %L must be bigger than the domain
            L = world_X*world_Y*10;
            
            v = [-n(2) n(1)];
            v = v / norm(v);
            
            p1 = mid + L*v;
            p2 = mid - L*v;
            
            hp = polyshape([p1(1) p2(1) p2(1)-n(1)*L p1(1)-n(1)*L],...
                           [p1(2) p2(2) p2(2)-n(2)*L p1(2)-n(2)*L]);
            
            cell_poly = intersect(cell_poly,hp);
        end
        
        if isempty(cell_poly.Vertices)
            continue
        end
        
        cells_history{k,i} = cell_poly;
        
        [cx,cy] = centroid(cell_poly);
        voronoi_centers(i,:) = [cx cy];
    end

    % --- MOVE AGENTS ---
    % Here we simply update positions to the new Voronoi centers
    agents = voronoi_centers;
    
    % Store history
    agents_history(k+1,:,:) = agents;
    values_history(k+1,:) = agent_values;
    
    % Progress indicator
    if mod(k, 10) == 0
        fprintf('Iteration %d/%d complete\n', k, Nsim);
    end
end

fprintf('Computation complete. Starting animation...\n');

%% ANIMATION - Combined view
figure('Position', [50, 50, 1400, 600]);

for k = 1:Nsim
    % Clear figure for animation
    clf;
    
    % Left subplot: Voronoi cells with small-world network
    subplot(1, 2, 1);
    plot_voronoi_with_network(agents_history(k,:,:), cells_history(k,:), ...
                              connections_history{k}, domain, world_X, world_Y, k);
    
    % Right subplot: Consensus progress
    subplot(1, 2, 2);
    plot_consensus_with_voronoi(values_history(1:k+1,:), k, numAgents);
    
    drawnow;
    pause(0.1);
end

% FINAL VISUALIZATIONS
% Plot agent trajectories
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
plot_trajectories(agents_history, domain, world_X, world_Y, numAgents);

subplot(1, 2, 2);
plot_consensus_evolution(values_history, Nsim, numAgents);

% Show final network structure
figure;
plot_final_network(agents_history(end,:,:), connections_history{end}, domain, world_X, world_Y);

fprintf('Done!\n');

function [connections, Laplacian, degree_matrix] = build_small_world_network(positions, k_neighbours, gamma, comm_range)
    % BUILD_SMALL_WORLD_NETWORK - Creates connections using Watts-Strogatz model
    % Returns adjacency matrix, Laplacian matrix, and degree matrix
    
    % Handle different input formats
    if ndims(positions) == 3
        positions = squeeze(positions);
    end
    
    N_agents = size(positions, 1);
    
    % Step 1: Create adjacency matrix (connections)
    % Start with no connections
    A = false(N_agents, N_agents); 
    
    % For each agent, find k nearest neighbors
    dist_matrix = squareform(pdist(positions));
    for i = 1:N_agents
        [~, idx] = sort(dist_matrix(i, :));
        neighbors = idx(2:min(k_neighbours+1, N_agents));
        A(i, neighbors) = true;
    end
    
    % Step 2: Rewire connections with probability gamma
    for i = 1:N_agents
        current_neighbors = find(A(i, :));
        
        for j = 1:length(current_neighbors)
            neighbor = current_neighbors(j);
            
            if i < neighbor && rand() < gamma
                A(i, neighbor) = false;
                
                available = setdiff(1:N_agents, [i, current_neighbors]);
                
                if ~isempty(available)
                    new_neighbor = available(randi(length(available)));
                    
                    if dist_matrix(i, new_neighbor) <= comm_range
                        A(i, new_neighbor) = true;
                    else
                        in_range = find(dist_matrix(i, :) <= comm_range & dist_matrix(i, :) > 0);
                        in_range = setdiff(in_range, [i, current_neighbors]);
                        
                        if ~isempty(in_range)
                            new_neighbor = in_range(randi(length(in_range)));
                            A(i, new_neighbor) = true;
                        else
                            A(i, neighbor) = true;
                        end
                    end
                end
            end
        end
    end
    
    % Ensure symmetry (undirected graph)
    A = A | A';
    
    % Step 3: Compute degree matrix and Laplacian
    % Degree matrix D (diagonal)
    degrees = sum(A, 2);
    degree_matrix = diag(degrees);
    
    % Laplacian matrix L = D - A
    % Convert A to double for Laplacian calculation
    A_double = double(A);
    Laplacian = degree_matrix - A_double;
    
    % Return adjacency matrix as connections (for backward compatibility)
    connections = A;
    
    % Optional: Display Laplacian properties
    % if N_agents <= 10  % Only for small networks to avoid clutter
    %     fprintf('\n--- Laplacian Matrix Properties ---\n');
    %     fprintf('Size: %dx%d\n', size(Laplacian,1), size(Laplacian,2));
    %     fprintf('Rank: %d\n', rank(Laplacian));
    %     fprintf('Trace: %f\n', trace(Laplacian));
    % 
    %     % Check if Laplacian is positive semi-definite
    %     eigenvalues = eig(Laplacian);
    %     fprintf('Smallest eigenvalue: %e\n', min(eigenvalues));
    %     % fprintf('Second smallest eigenvalue (algebraic connectivity): %f\n', ...
    %     %         sort(eigenvalues, 'ascend')(2));
    % end
end

function new_values = consensus_step(values, Laplacian, dt)
    % CONSENSUS_STEP_LAPLACIAN - Consensus using Laplacian dynamics
    % dx/dt = -L x
    % Discretized: x_new = x - dt * L * x
    
    if nargin < 3
        dt = 0.1;  % Time step
    end
    
    % Continuous-time consensus
    new_values = values - dt * Laplacian * values;
    
    % Ensure values stay bounded (optional)
    new_values = max(0, min(1, new_values));
end

function plot_voronoi_with_network(agents, cells, connections, domain, world_X, world_Y, iteration)
    % PLOT_VORONOI_WITH_NETWORK - Visualize Voronoi cells and communication network
    
    % Extract current agent positions
    if ndims(agents) == 3
        current_agents = squeeze(agents);
    else
        current_agents = agents;
    end
    
    % Plot domain boundary
    plot(domain, 'FaceColor', 'none', 'LineWidth', 2, 'EdgeColor', 'k');
    hold on;
    axis equal;
    xlim([0 world_X]);
    ylim([0 world_Y]);
    
    % Plot all Voronoi cells
    for i = 1:length(cells)
        if ~isempty(cells{i})
            % Color cells based on agent values (if we had them at this timestep)
            % For now, use light blue
            plot(cells{i}, 'FaceColor', [0.85 0.9 1], 'EdgeColor', 'b', 'LineWidth', 0.5);
        end
    end
    
    % Plot communication network connections
    if ~isempty(connections)
        for i = 1:size(connections, 1)
            for j = i+1:size(connections, 2)
                if connections(i, j)
                    plot([current_agents(i,1), current_agents(j,1)], ...
                         [current_agents(i,2), current_agents(j,2)], ...
                         'g-', 'LineWidth', 1, 'Color', [0.4 0.8 0.4]);
                end
            end
        end
    end
    
    % Plot agents
    scatter(current_agents(:,1), current_agents(:,2), 80, 'r', 'filled');
    
    % Add agent numbers
    for i = 1:size(current_agents, 1)
        text(current_agents(i,1)+0.2, current_agents(i,2)+0.2, num2str(i), ...
             'FontSize', 8, 'FontWeight', 'bold', 'Color', 'k');
    end
    
    title(sprintf('Voronoi Relaxation + Small-World Network (Iter %d, γ=0.2)', iteration));
    xlabel('world_X Position');
    ylabel('world_Y Position');
    hold off;
end

function plot_consensus_with_voronoi(values_history, current_iter, numAgents)
    % PLOT_CONSENSUS_WITH_VORONOI - Show consensus progress
    
    % Extract values up to current iteration
    values = values_history(1:current_iter+1, :);
    
    % Plot all agent value trajectories
    plot(0:current_iter, values, 'LineWidth', 1.5);
    hold on;
    
    xlim([0, size(values_history, 1)-1]);
    ylim([-0.1 1.1]);
    xlabel('Iteration');
    ylabel('Agent Values');
    title('Consensus on Agent Values');
    grid on;
    
    % Calculate and display convergence metrics
    current_values = values(end, :);
    spread = std(current_values);
    range_val = max(current_values) - min(current_values);
    
    text(current_iter*0.7, 0.9, sprintf('Std: %.4f\nRange: %.4f', spread, range_val), ...
         'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', 'w');
    
    % Add vertical line at current iteration
    plot([current_iter, current_iter], [-0.1, 1.1], 'k--', 'LineWidth', 1);
    
    hold off;
end

function plot_trajectories(agents_history, domain, world_X, world_Y, numAgents)
    % PLOT_TRAJECTORIES - Show agent movement paths
    
    plot(domain, 'FaceColor', 'none', 'LineWidth', 2, 'EdgeColor', 'k');
    hold on;
    axis equal;
    xlim([0 world_X]);
    ylim([0 world_Y]);
    
    % Plot trajectories for each agent
    colors = lines(numAgents);
    for i = 1:numAgents
        trajectory = squeeze(agents_history(:,i,:));
        plot(trajectory(:,1), trajectory(:,2), '.-', 'Color', colors(i,:), 'MarkerSize', 8);
    end
    
    % Mark start and end positions
    start_agents = squeeze(agents_history(1,:,:));
    final_agents = squeeze(agents_history(end,:,:));
    
    scatter(start_agents(:,1), start_agents(:,2), 50, 'g', 'filled', 'MarkerEdgeColor', 'k');
    scatter(final_agents(:,1), final_agents(:,2), 80, 'r', 'filled', 'MarkerEdgeColor', 'k');
    
    legend('Domain', 'Trajectories', 'Start', 'Final', 'Location', 'eastoutside');
    title('Agent Trajectories during Voronoi Relaxation');
    xlabel('world_X Position');
    ylabel('world_Y Position');
    hold off;
end

function plot_consensus_evolution(values_history, Nsim, numAgents)
    % PLOT_CONSENSUS_EVOLUTION - Show how consensus metrics evolved
    
    % Calculate spread over time
    spread = std(values_history, 0, 2);
    
    % Plot
    subplot(2, 1, 1);
    plot(0:Nsim, values_history, 'LineWidth', 1);
    xlabel('Iteration');
    ylabel('Agent Values');
    title('Consensus Evolution');
    grid on;
    xlim([0 Nsim]);
    
    subplot(2, 1, 2);
    plot(0:Nsim, spread, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Standard Deviation');
    title('Convergence Metric');
    grid on;
    xlim([0 Nsim]);
    
    % Add convergence info
    final_spread = spread(end);
    text(Nsim*0.7, max(spread)*0.8, sprintf('Final Std: %.4f', final_spread), ...
         'FontSize', 11, 'FontWeight', 'bold');
end

function plot_final_network(agents, connections, domain, world_X, world_Y)
    % PLOT_FINAL_NETWORK - Show the final small-world network structure
    
    % Extract final agent positions
    if ndims(agents) == 3
        final_agents = squeeze(agents);
    else
        final_agents = agents;
    end
    
    % Plot domain
    plot(domain, 'FaceColor', 'none', 'LineWidth', 2, 'EdgeColor', 'k');
    hold on;
    axis equal;
    xlim([0 world_X]);
    ylim([0 world_Y]);
    
    % Plot connections
    if ~isempty(connections)
        for i = 1:size(connections, 1)
            for j = i+1:size(connections, 2)
                if connections(i, j)
                    % Color based on connection type (local vs long-range)
                    dist = norm(final_agents(i,:) - final_agents(j,:));
                    if dist < 2  % Arbitrary threshold for "local"
                        line_color = [0.4 0.6 1];  % Blue for local
                    else
                        line_color = [1 0.4 0.4];  % Red for long-range shortcuts
                    end
                    plot([final_agents(i,1), final_agents(j,1)], ...
                         [final_agents(i,2), final_agents(j,2)], ...
                         '-', 'LineWidth', 1, 'Color', line_color);
                end
            end
        end
    end
    
    % Plot agents
    scatter(final_agents(:,1), final_agents(:,2), 100, 'k', 'filled');
    
    title('Final Small-World Network Structure');
    xlabel('world_X Position');
    ylabel('world_Y Position');
    
    % Add legend
    plot([-1 -1], [-1 -1], 'b-', 'LineWidth', 2);
    plot([-1 -1], [-1 -1], 'r-', 'LineWidth', 2);
    legend('Local Connections', 'Long-range Shortcuts', 'Location', 'eastoutside');
    
    % Calculate network statistics
    num_connections = sum(connections(:))/2;
    avg_degree = 2*num_connections / size(final_agents, 1);
    
    text(1, world_Y-1, sprintf('Total Connections: %d\nAvg Degree: %.2f', ...
         num_connections, avg_degree), 'FontSize', 11, 'FontWeight', 'bold', ...
         'BackgroundColor', 'w');
    
    hold off;
end