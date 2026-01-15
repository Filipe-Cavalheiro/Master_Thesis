% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD

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
N = 5; 
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

for i = 1:Param.nD
    ref_pos{i} = x{i}(1:3,1);
end
current_pos = zeros(3, Param.nD);  % preallocation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("intereation:%d\n", k);
    for iD = 1:Param.nD
        % get state vector
        pos{iD} = x{iD}(1:3,k);
        v{iD} = x{iD}(4:6,k);
        R = reshape(x{iD}(7:15,k),3,3);
        om = x{iD}(16:18,k);

        for i = 1:Param.nD
            current_pos(:, i) = x{i}(1:3,k);
        end

        current_radius = Param.sphere_radius + (Param.end_radius - Param.sphere_radius) * (k - 1) / (Nsim - 1);

        fib_ref_points= get_fibonacci_points(Param.nD, current_radius);
        fib_ref_points = fib_ref_points';

        % Compute geodesic distances between all pairs
        ref_cart_distances = cartisian_distances(fib_ref_points');

        for i = 1:Param.nD
            distances = ref_cart_distances(i, :);
            [sorted_dists, sorted_idx] = sort(distances);
            dist_vals(i,:) = sorted_dists(1:Param.K_neighbour);
        end
        
        % Compute average distance per point
        ref_min_distance = min(dist_vals, [], "all"); % min across matrix
        min_safe_dist = ref_min_distance*1;

        Y_bar = kron(ones(N+1, 1), fib_ref_points);

        %Compute geodesic distances between all pairs
        current_cart_distances = cartisian_distances(current_pos');

        % Unconstrained MPC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = -K(iD*3-2:iD*3,1:3)*current_pos(:, iD) + Ky(iD*3-2:iD*3,1:nx*(N+1))*Y_bar(:, iD);        
        ref_pos{iD} = A*current_pos(:, iD) + B*U(1:nu);

        % Constraint of safe distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Find K nearest neighbors for point iD
        distances = current_cart_distances(iD, :);
        [sorted_dists, sorted_idx] = sort(distances);
        neighbor_indices = sorted_idx(1:Param.K_neighbour); % store neighbor indices
        dist_vals = sorted_dists(1:Param.K_neighbour);

        for i = 1:Param.K_neighbour
            if dist_vals(i) > min_safe_dist
                continue; 
            end
        
            A_point = current_pos(:, iD);
            B_point = current_pos(:, sorted_idx(i));
        
            A_norm = A_point / norm(A_point);
            B_norm = B_point / norm(B_point);
        
            % Tangent direction from A away from B (great-circle direction)
            tangent_dir = B_norm - dot(B_norm, A_norm) * A_norm;  % projection of B onto tangent plane at A
            tangent_dir = -tangent_dir / norm(tangent_dir);  % direction away from B on the sphere
        
            % Move A slightly along this tangent direction
            delta = (min_safe_dist - dist_vals(i));  % how much too close
            ref_pos{iD} = ref_pos{iD} + tangent_dir * delta;
        end
        % ref_pos{iD} = (ref_pos{iD} / norm(ref_pos{iD})) * Param.sphere_radius;

        % get reference
        %p_d = p_ref{iD}(:,k);
        p_d =  ref_pos{iD};
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

% --- Plot ---
% 3D trajectories
figure; hold on; grid on; axis equal
[x,y,z] = sphere;
x = x*Param.sphere_radius;
y = y*Param.sphere_radius;
z = z*Param.sphere_radius;

hSphere = surf(x,y,z);
hSphere.FaceAlpha = 0.3;   % Set transparency (0 = fully transparent, 1 = opaque)
hSphere.EdgeColor = 'none'; % Optional: remove mesh lines for smoother look

scatter3(fib_ref_points(1, :), fib_ref_points(2, :), fib_ref_points(3, :)); % start
colors = lines(Param.nD);
for iD = 1:Param.nD
    plot3(squeeze(pos{iD}(1, :)), squeeze(pos{iD}(2, :)), squeeze(pos{iD}(3, :)), ...
        'Color', colors(iD,:), 'LineWidth',0.5);
    plot3(pos{iD}(1, 1), pos{iD}(2, 1), pos{iD}(3, 1), 'o', 'Color', colors(iD,:), 'MarkerSize',6); % start
    plot3(pos{iD}(1, end), pos{iD}(2, end), pos{iD}(3, end), 'x', 'Color', colors(iD,:), 'MarkerSize',6); % end
end
view(3);
xlabel('x'); ylabel('y'); zlabel('z');
title('3D Trajectories of 9 Agents (Case 1)');
hold off;
