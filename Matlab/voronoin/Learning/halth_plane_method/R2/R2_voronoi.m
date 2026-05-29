clear; clc; close all;

Tend = 1;           
dTi = 0.01;         
Nsim = round(Tend/dTi);
world_X = 10;
world_Y = 10;

numAgents = 30;

% Initial positions
agents = 10*rand(numAgents,2);

% Square domain
domain = polyshape([0 world_X world_X 0],[0 0 world_Y world_Y]);

agents_history = cell(Nsim,1);
cells_history = cell(Nsim,numAgents);
numAgents_history = zeros(Nsim,1);

fprintf('Computing Voronoi relaxation without voronoin...\n');

%   Compute
for k = 1:Nsim
    numAgents_history(k) = numAgents;
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
    
    agents = voronoi_centers;
    
    agents_history{k} = agents;
    
end

fprintf('Simulation finished\n');

%% ANIMATION

figure;
axis equal;
hold on;
xlim([0 world_X]);
ylim([0 world_Y]);

% Plot domain
plot(domain, 'FaceColor','none','LineWidth',2);

% Initial agent positions
agents = agents_history{1};
hScatter = scatter(agents(:,1), agents(:,2), 60, 'filled');

% Agent counter label
agentLabel = uicontrol('Style','text', ...
                       'String',['Agents: ', num2str(size(agents,1))], ...
                       'FontSize',12,'FontWeight','bold', ...
                       'Position',[20 20 120 30]);

% Maximum number of agents possible
maxAgents = size(cells_history,2);

% Initialize cell graphics
hCells = gobjects(maxAgents,1);

for i = 1:maxAgents
    hCells(i) = patch(NaN, NaN, 'b', ...
        'FaceAlpha',0.2, ...
        'EdgeColor','b');
end

% Animation loop
for k = 1:Nsim
    
    currentAgents = numAgents_history(k);
    
    % Update Voronoi cells
    for i = 1:maxAgents
        
        if i <= currentAgents && ~isempty(cells_history{k,i})
            
            cell_poly = cells_history{k,i};
            [x,y] = boundary(cell_poly);
            
            set(hCells(i),'XData',x,'YData',y);
            
        else
            set(hCells(i),'XData',NaN,'YData',NaN);
        end
        
    end
    
    % Update agent positions
    agents = agents_history{k};
    set(hScatter,'XData',agents(:,1),'YData',agents(:,2));

    % Update label
    set(agentLabel,'String',['Agents: ', num2str(currentAgents)]);
    
    title(['Iteration ', num2str(k)]);
    
    drawnow;
end