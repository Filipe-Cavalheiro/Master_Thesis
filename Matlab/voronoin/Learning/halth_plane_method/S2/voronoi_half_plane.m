clear; clc; close all;

Tend = 1;
dTi = 0.01;
Nsim = round(Tend/dTi);

numAgents = 30;
R = 1.0;
tol = 5e-3;  % tolerance for "same location"
verbose = false;

% ---------------------------------------------------------------
% Precompute the sphere sampling grid and its triangle connectivity
% ONCE. This never changes across agents/timesteps/runs, so there is
% no reason to recompute it 300,000 times as the original code did.
% ---------------------------------------------------------------
[X, Y, Z] = sphere(50);
P = [X(:), Y(:), Z(:)];
[nr, nc] = size(X);

[I, J] = ndgrid(1:nr-1, 1:nc-1);
idxA = sub2ind(size(X), I(:),   J(:));
idxB = sub2ind(size(X), I(:)+1, J(:));
idxC = sub2ind(size(X), I(:),   J(:)+1);
idxD = sub2ind(size(X), I(:)+1, J(:)+1);

A0 = P(idxA,:); B0 = P(idxB,:); C0 = P(idxC,:); D0 = P(idxD,:);

nRuns = 100;
half_plane_times = zeros(1, nRuns);
convergence_iteration = zeros(1, nRuns);
converged = 0;
not_converged = 0;

for run = 1:nRuns
    converged_bool = false;
    agents_unit = randn(numAgents,3);
    agents_unit = agents_unit ./ vecnorm(agents_unit,2,2);
    agents = agents_unit * R;

    agents_history   = cell(Nsim+1, 1);
    numAgents_history = zeros(Nsim,1);

    tic;

    for k = 1:Nsim
        numAgents_history(k) = numAgents;
        agents_history{k} = agents;
        new_agents = zeros(numAgents,3);
    
        % --- Half-plane Voronoi computation ---
        for iD = 1:numAgents
            current_agent = agents(iD,:);
            other_agents  = agents([1:iD-1, iD+1:numAgents], :);
            Nother = size(other_agents,1);
    
            p1 = repmat(current_agent, Nother, 1);
            p2 = other_agents;
            cosTheta = sum(p1 .* p2, 2);
            cosTheta = min(max(cosTheta, -1), 1);
            theta = acos(cosTheta);
            m = (sin(0.5*theta).*p1 + sin(0.5*theta).*p2) ./ sin(theta);
            n = cross(p1, p2, 2);
            n = n ./ vecnorm(n,2,2);
            tangent = cross(n, m, 2);
            tangent = tangent ./ vecnorm(tangent,2,2);
    
            offsets = sum(m .* tangent, 2);
            Dmat = P * tangent';
            mask = all(Dmat < offsets', 2);
    
            validCell = mask(idxA) & mask(idxB) & mask(idxC) & mask(idxD);
            cross1 = cross(B0 - A0, C0 - A0, 2);
            area1  = vecnorm(cross1,2,2) / 2;
            centroid1 = (A0 + B0 + C0) / 3;
            cross2 = cross(C0 - D0, B0 - D0, 2);
            area2  = vecnorm(cross2,2,2) / 2;
            centroid2 = (D0 + C0 + B0) / 3;
            area1(~validCell) = 0;
            area2(~validCell) = 0;
            total_area = sum(area1) + sum(area2);
            Cxyz = sum(area1.*centroid1 + area2.*centroid2, 1);
            centroid = Cxyz / total_area;
            new_agents(iD,:) = centroid ./ norm(centroid);
        end
    
        % ============ CONVERGENCE CHECK (once per timestep, AFTER all agents updated) ============
        if max(vecnorm(new_agents - agents, 2, 2)) < tol
            agents = new_agents;
            agents_history{k+1} = agents;
            convergence_iteration(run) = k;
            converged = converged + 1;
            converged_bool = true;
            fprintf('Converged after %d iterations.\n', k);
            break;  
        end
    
        agents = new_agents;
    end

    half_plane_times(run) = toc;
    fprintf("Run %03d/%03d done in %.3f s\n", run, nRuns, half_plane_times(run));
    if converged_bool == false
        not_converged = not_converged + 1;
    end
    %sphere_uniformity(agents);
end

fprintf("Mean time per run: %.3f s\n", mean(half_plane_times));
fprintf("Mean necessary iterations for convergence: %.3f \n", mean(convergence_iteration));
fprintf("converged %d times \n", converged)
fprintf("did not converged %d times \n", not_converged)

if verbose == false
    return;
end

%%
figure;
hold on;
axis equal;
grid on;
view(3);

colors = lines(numAgents); % color per agent

for i = 1:numAgents
    traj = [];
    
    for k = 1:Nsim
        if size(agents_history{k},1) >= i
            traj = [traj; agents_history{k}(i,:)];
        end
    end
    
    plot3(traj(:,1), traj(:,2), traj(:,3), '-', 'Color', colors(i,:));
    scatter3(traj(end,1), traj(end,2), traj(end,3), 50, colors(i,:), 'filled');
end

% draw sphere
[xs,ys,zs] = sphere(50);
surf(xs,ys,zs,'FaceAlpha',0.05,'EdgeColor','none');

title('Agent trajectories over time');

%%
figure;
axis equal;
grid on;
view(3);
hold on;

[xs,ys,zs] = sphere(50);
surf(xs,ys,zs,'FaceAlpha',0.05,'EdgeColor','none');

h = scatter3([],[],[],50,'filled');

for k = 1:Nsim
    A = agents_history{k};
    
    set(h, 'XData', A(:,1), 'YData', A(:,2), 'ZData', A(:,3));
    title(['Time step: ', num2str(k)]);
    
    drawnow;
    pause(0.05);
end

%%
figure;
hold on;

% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(R*XS, R*YS, R*ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]); 

% Plot points
A = agents_history{Nsim};
scatter3(A(:,1), A(:,2), A(:,3), 50, 'k', 'filled');

% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure

view(3);
print2pdf("Lloyd_half_plane_normal.pdf", 1);
view(90,0);
print2pdf("Lloyd_half_plane_side.pdf", 1);
view(0,90);
print2pdf("Lloyd_half_plane_top.pdf", 1);