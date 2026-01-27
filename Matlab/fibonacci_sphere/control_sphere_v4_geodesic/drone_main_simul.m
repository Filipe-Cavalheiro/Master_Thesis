% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD

fib_ref_points= get_fibonacci_points(Param.nD, Param.sphere_radius);
fib_ref_points = fib_ref_points';
min_safe_dist = get_min_safe_distance(Param.nD, Param.sphere_radius);

clearvars dist_vals sorted_dists sorted_idx distances;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MPC Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = eye(3);
B = eye(3);
C = eye(3);
nx = size(B, 1);
nu = size(B, 2);
ny = size(C, 1);

% MPC data
N = 30; 
P = eye(nx);
Q = eye(nx);
R = eye(nu);

F = eye(nx);
G = zeros(nx, N*nu);
Gaux = [];
for n = 1:N
    F = [F; A^n];
    Gaux = [A^(n-1)*B Gaux];
    G = [G; Gaux zeros(nx, (N-n)*nu)];
end

H = kron(eye(N+1), C);
F_bar = H*F;
G_bar = H*G;
Q_bar = blkdiag(kron(eye(N), Q), P);
R_bar = kron(eye(N), R);

R_til = G_bar'*Q_bar*G_bar+R_bar;
S_til = G_bar'*Q_bar;

Ky = R_til^(-1)*S_til;
K = Ky*F_bar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y_bar = kron(ones(N+1, 1), fib_ref_points);
for i = 1:Param.nD
    ref_pos{i} = x{i}(1:3,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("intereation:%d\n", k);

    % parse the position from cells to array
    for i = 1:Param.nD
        current_pos(:, i) = x{i}(1:3,k);
    end

    % Project points on the sphere
    points_on_sphere = projectToSphere(current_pos', Param.sphere_radius);
    %Compute geodesic distances between all pairs
    current_geo_distances = geo_distances(points_on_sphere, Param.sphere_radius);

    for iD = 1:Param.nD
        
        % get state vector
        pos{iD} = x{iD}(1:3,k);
        v{iD} = x{iD}(4:6,k);
        R = reshape(x{iD}(7:15,k),3,3);
        om = x{iD}(16:18,k);

        % Unconstrained MPC
        % U = -K(iD*3-2:iD*3,1:3)*current_pos(:, iD) + Ky(iD*3-2:iD*3,1:nx*(N+1))*Y_bar(:, iD);        
        % ref_pos{iD} = A*current_pos(:, iD) + B*U(1:nu);

        % remove distance to self
        distances = current_geo_distances(iD, :);
        [sorted_dists, sorted_idx] = sort(distances);
        neighbor_indices = sorted_idx(1:(Param.nD-1));
        dist_vals = sorted_dists(1:(Param.nD-1));
        
        artificial_potential_vector = zeros(3,1);
        for i = 1:Param.nD-1
            if dist_vals(i) > min_safe_dist
                continue;
            end
            A_point = points_on_sphere(iD, :)';
            B_point = points_on_sphere(sorted_idx(i), :)';

            % Ensure normalized to sphere radius R
            A_point = (A_point / norm(A_point)) * Param.sphere_radius;
            B_point = (B_point / norm(B_point)) * Param.sphere_radius;
            % Direction vector from A to B (parallel direction)
            AB_direction =  A_point - B_point;
            AB_direction = AB_direction / norm(AB_direction);
            
            artificial_potential_vector = artificial_potential_vector + A_point + (AB_direction) *(min_safe_dist-dist_vals(i));
        end
        % After finishing the loop make sure the vector is clamped
        vec_mag = norm(artificial_potential_vector);
        % If the vector exceeds min_safe_dist, scale it down
        if vec_mag > min_safe_dist
            artificial_potential_vector = (artificial_potential_vector / vec_mag) * min_safe_dist;
        end
        % Apply the corrected vector
        ref_pos{iD}(:,k) = artificial_potential_vector;
        
        %project back to sphere the ref
        ref_pos{iD}(:,k) = (ref_pos{iD}(:,k) / norm(ref_pos{iD}(:,k))) * Param.sphere_radius;
        
        % get reference
        %p_d = p_ref{iD}(:,k);
        p_d =  ref_pos{iD}(:,k);
        %v_d = v_ref{iD}(:,k);
        v_d = zeros(3,1);
        %a_d = a_ref{iD}(:,k);
        a_d = zeros(3,1);
        %j_d = j_ref{iD}(:,k);
        j_d = zeros(3,1);
        %psi_d = psi_ref{iD}(:,k);
        psi_d = 0;
        %dpsi_d = dpsi_ref{iD}(:,k);
        dpsi_d = 0;

        % mellinger controller
        [T{iD}(:,k),tau{iD}(:,k),e_p] = drone_mellinger_ctrl(pos{iD},v{iD},R,om,Param,p_d,psi_d,xiep{iD}(:,k),v_d,dpsi_d,a_d,j_d);
        
        % integrate position error
        xiep{iD}(:,k+1) = xiep{iD}(:,k) + Param.dTi*e_p;
        
        % nonlinear drone model (continuous time)
        [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(v{iD},R,om,T{iD}(:,k),tau{iD}(:,k),Param);
        
        % discretization 
        pp = pos{iD} + Param.dTi*dot_p;
        vp = v{iD} + Param.dTi*dot_v;
        Rp = rot_integrate(R,om,Param.dTi);
        omp = om + Param.dTi*dot_om;
        x{iD}(:,k+1) = [pp;vp;reshape(Rp,[],1);omp];

        % auxiliary drone attitude computation from rotation matrix
        lbd{iD}(:,k) = R2Euler(R);

        if abs(1-norm(Rp'*Rp))>1e-4
            warning(['Problems with rotation matrix integration... stoping simulation: t = ' num2str(k*Param.dTi) ' s.']);
            break;
        end
    
    end

    if abs(1-norm(Rp'*Rp))>1e-4
        break;
    end
end

% trim vectors to same length
clear pos;
for iD = 1:Param.nD
    pos{iD} = x{iD}(1:3,:);
end

%drone_show_data;

figure; hold on; grid on; axis equal
[x, y, z] = sphere;
x = x * Param.sphere_radius;
y = y * Param.sphere_radius;
z = z * Param.sphere_radius;

% Draw reference sphere
hSphere = surf(x, y, z);
hSphere.FaceAlpha = 0.3; 
hSphere.EdgeColor = 'none';
hSphere.FaceColor = [0.8627,0.8627,0.8627];

% Setup colors and plot handles for agents
colors = lines(Param.nD);
for iD = 1:Param.nD
    scatter3( ...
        pos{iD}(1,1), pos{iD}(2,1), pos{iD}(3,1), ...
        50, colors(iD,:), 'x', 'LineWidth', 2, ...
        'HandleVisibility','off');

    scatter3( ...
        ref_pos{iD}(1,1), ref_pos{iD}(2,1), ref_pos{iD}(3,1), ...
        50, colors(iD,:), 'o', 'LineWidth', 2, ...
        'HandleVisibility','off');
end

hInit = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'x', 'LineWidth', 2);

hProj = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'o', 'LineWidth', 2);

legend([hInit, hProj], ...
    {'Initial Positions', 'Initial Position Projection'});

lgd = legend;
lgd.AutoUpdate = 'off';

for iD = 1:Param.nD
    % Initialize line and marker objects
    hTraj{iD} = plot3(NaN, NaN, NaN, 'Color', colors(iD,:), 'LineWidth', 1);
    hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    
    hrefTraj{iD} = plot3(NaN, NaN, NaN, 'Color', colors(iD,:), 'LineWidth', 1);
    href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
end

xlabel('x'); ylabel('y'); zlabel('z');
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure
view(3);

% Animation loop
nSteps = size(pos{1}, 2);  % number of time steps
for k = 1:nSteps-1
    for iD = 1:Param.nD
        % Update trajectory up to current time
        set(hTraj{iD}, 'XData', pos{iD}(1, 1:k), ...
                       'YData', pos{iD}(2, 1:k), ...
                       'ZData', pos{iD}(3, 1:k));
        % Update agent position marker
        set(hAgent{iD}, 'XData', pos{iD}(1, k), ...
                        'YData', pos{iD}(2, k), ...
                        'ZData', pos{iD}(3, k));

        % Update ref position marker
        set(hrefTraj{iD}, 'XData', ref_pos{iD}(1, 1:k), ...
                        'YData', ref_pos{iD}(2, 1:k), ...
                        'ZData', ref_pos{iD}(3, 1:k));

        % Update ref position marker
        set(href{iD}, 'XData', ref_pos{iD}(1, k), ...
                        'YData', ref_pos{iD}(2, k), ...
                        'ZData', ref_pos{iD}(3, k));
    end
    drawnow;
    pause(0.01); % adjust speed of animation
end