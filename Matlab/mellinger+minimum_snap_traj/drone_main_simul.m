% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)
% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;

% --- DRONE INITIAL CONDITIONS ---
for iD = 1:Param.nD
    x{iD} = zeros(18,Nsim+1); % [pos; vel; R(9); omega]
    x{iD}(7:15,1) = reshape(eye(3),[],1); % initial rotation
    xiep{iD} = zeros(3,Nsim+1);           % integrated position error
end

% TRAJECTORY GENERATION (Minimum Snap)
p0 = [0;0;0];
pf = [10;10;10];
T  = Nsim*Param.dTi;

% Function to generate minimum snap trajectory
%expanded_traj = minimum_snap_trajectory_3D(p0, pf, T, Param.dTi);
expanded_traj  = [linspace(0, 10, Nsim); linspace(0, 10, Nsim); linspace(0, 10, Nsim)];

% LOGS
pos_hist = zeros(3,Nsim);

% SIMULATION LOOP
for k = 1:Nsim
    fprintf("Iteration: %d\n", k);
    for iD = 1:Param.nD
        
        % GET STATE VECTOR
        pos{iD} = x{iD}(1:3,k);
        v{iD}   = x{iD}(4:6,k);
        R       = reshape(x{iD}(7:15,k),3,3);
        om      = x{iD}(16:18,k);

        pos_hist(:,k) = pos{iD};
        
        % GET REFERENCE
        p_d = expanded_traj(:, k);  % desired position
        v_d = zeros(3,1);           % desired velocity
        a_d = zeros(3,1);           % desired acceleration
        j_d = zeros(3,1);           % desired jerk
        psi_d  = 0;
        dpsi_d = 0;

        % Mellinger controller
        [Torque_ctrl,tau,e_p] = drone_mellinger_ctrl(pos{iD},v{iD},R,om,Param,p_d,psi_d,xiep{iD}(:,k),v_d,dpsi_d,a_d,j_d);
        
        % Integrate position error
        xiep{iD}(:,k+1) = xiep{iD}(:,k) + Param.dTi*e_p;
        
        % Nonlinear drone dynamics 
        [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(v{iD},R,om,Torque_ctrl,tau,Param);
        
        % Discretization
        pp  = pos{iD} + Param.dTi*dot_p;
        vp  = v{iD}   + Param.dTi*dot_v;
        Rp  = rot_integrate(R,om,Param.dTi);
        omp = om + Param.dTi*dot_om;
        x{iD}(:,k+1) = [pp; vp; reshape(Rp,[],1); omp];
        
        % Safety check for rotation matrix
        if abs(1-norm(Rp'*Rp))>1e-4
            warning(['Problems with rotation matrix integration... stopping simulation at t = ' num2str(k*Param.dTi) ' s.']);
            break;
        end
    end
    
    if abs(1-norm(Rp'*Rp))>1e-4
        break;
    end
end

%% --- PLOT TRAJECTORY ---
figure;
hold on;
plot3(expanded_traj(1,:), expanded_traj(2,:), expanded_traj(3,:), 'b-', 'LineWidth', 2);
plot3(pos_hist(1,:), pos_hist(2,:), pos_hist(3,:), 'r-', 'LineWidth', 2);
plot3(pos_hist(1,1), pos_hist(2,1), pos_hist(3,1), 'gx','MarkerFaceColor','g'); % start
plot3(pos_hist(1,end), pos_hist(2,end), pos_hist(3,end), 'rx','MarkerFaceColor','r'); % end
grid on;
xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
title('3D Minimum Snap Trajectory of Drone');
legend('desired trajectory', 'obtained','Start','End');
view(3); % 3D view
axis equal;

%% --- MINIMUM SNAP TRAJECTORY FUNCTION ---
function traj = minimum_snap_trajectory_3D(p0, pf, T, dt)
    % Generate 3D minimum snap trajectory
    % p0, pf: 3x1 vectors
    % T: total time
    % dt: time step

    t0 = 0;
    traj_time = 0:dt:T;
    traj = zeros(length(traj_time),3);

    % Polynomial matrix (7th-order)
    A = [...
        1 t0 t0^2 t0^3 t0^4 t0^5 t0^6 t0^7;
        0 1 2*t0 3*t0^2 4*t0^3 5*t0^4 6*t0^5 7*t0^6;
        0 0 2 6*t0 12*t0^2 20*t0^3 30*t0^4 42*t0^5;
        0 0 0 6 24*t0 60*t0^2 120*t0^3 210*t0^4;
        1 T T^2 T^3 T^4 T^5 T^6 T^7;
        0 1 2*T 3*T^2 4*T^3 5*T^4 6*T^5 7*T^6;
        0 0 2 6*T 12*T^2 20*T^3 30*T^4 42*T^5;
        0 0 0 6 24*T 60*T^2 120*T^3 210*T^4];

    for dim = 1:3
        % Boundary conditions per axis
        bc = [p0(dim); 0; 0; 0; pf(dim); 0; 0; 0];
        
        % Solve for polynomial coefficients
        coef = A\bc;

        % Evaluate polynomial over trajectory time
        for k = 1:length(traj_time)
            t = traj_time(k);
            traj(k,dim) = polyval(flip(coef'), t);
        end
    end
end