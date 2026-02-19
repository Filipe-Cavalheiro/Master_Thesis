% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fibonacci sphere values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fib_ref_points = get_fibonacci_points(Param.nD, Param.sphere_radius, Param.initial_target_pos);
min_safe_dist = get_min_safe_distance(Param.nD, Param.sphere_radius)*0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign each agent to nearest ref point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute geodesic distances between all pairs
pos_on_sphere = projectToSphere(initial_pos, Param.sphere_radius, Param.initial_target_pos);
geo_distances_all = zeros(Param.nD, Param.nD);
for i = 1:Param.nD
    for j = 1:Param.nD
        a = fib_ref_points(:,i);
        b = pos_on_sphere(:,j);
        dot_product = dot(a-Param.initial_target_pos, b-Param.initial_target_pos);
        geo_distances_all(i, j) = Param.sphere_radius * acos(dot_product / (Param.sphere_radius^2));
    end
end

%the theoretical maximum two points will be apart is pi so 10pi is just to
%force full one-to-one assignment
costofnonassignment = 10*pi*Param.sphere_radius;
[assignments, unassignedrows, unassignedcolumns] = ...
    assignmunkres(geo_distances_all, costofnonassignment);

if ~isempty(unassignedrows) || ~isempty(unassignedcolumns)
    disp("ERROR: impossible to assing 1-to-1");
    exit;
end

organized_initial_pos = zeros(3,Param.nD);
organized_Rad = zeros(1,Param.nD);
for i = 1:Param.nD
    organized_initial_pos(:,i) = initial_pos(:,assignments(i,1));
    organized_Rad(i) = Param.r0(assignments(i,1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iD = 1:Param.nD
    % initial values
    Rad{iD} = organized_Rad(iD);
    chaser{iD}(1:3,1) = organized_initial_pos(:,iD);
    q{iD} = organized_initial_pos(:,iD);
    omega{iD} = [0; 0; 0];
    
    % desired values
    q_d{iD} = projectToSphere(fib_ref_points(:,iD), Param.sphere_radius, Param.initial_target_pos);
    
    omega_d{iD} = [0; 0; 0];
end


% Storage
for iD = 1:Param.nD
    q_hist{iD} = zeros(3, Nsim);
    Rad_hist{iD} = zeros(1, Nsim);
    q_d_hist{iD} = zeros(3, Nsim);
end
c_sphere = zeros(3,Nsim);
target_hist = zeros(18,Nsim);

current_pos = zeros(3,Param.nD);
for i = 1:Param.nD
    current_pos(:, i) = q{i};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target drone initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial target state
target.x = zeros(18,1);
target.x(1:3)   = Param.initial_target_pos;    % position
target.x(4:6)   = [0;0;0];                     % velocity
target.x(7:15)  = reshape(eye(3),[],1);        % rotation matrix
target.x(16:18) = [0;0;0];                     % angular velocity

% Integral error (if you want to reuse Mellinger)
target.xiep = zeros(3,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("Computing simulation interation: %04d\n", k);
    
    %move target
    t = (k-1)*Param.dTi;
    p_d_target = target_ref(t);
    [target, c_sphere(:,k)] = target_motion(target, p_d_target, Param);
    target_hist(:,k) = target.x;

    for iD = 1:Param.nD
        % get state vector
        pos{iD} = chaser{iD}(1:3,k);
        v{iD} = chaser{iD}(4:6,k);
        R = reshape(chaser{iD}(7:15,k),3,3);
        om = chaser{iD}(16:18,k);
        
        % Position reference generation
        [q{iD}, e_q] = linear_motion_ref(Rad{iD}, pos{iD},omega{iD},q_d{iD},omega_d{iD},c_sphere(:,k), Param);
            
        % Position reference shift to safe guard collisions
        current_pos(:, iD) = q{iD};
        q{iD} = AFP_safe_guard(iD, current_pos, min_safe_dist, Param.sphere_radius, Rad{iD}, c_sphere(:,k), Param);
        
        % update radius
        Rad{iD} = update_radius(Rad{iD}, e_q, q{iD}, q_d{iD}, chaser{iD}(:,k), target.x,Param);
            
        %shift q_d to new sphere pos
        if k > 1
            % update radius of q_d
            q_d{iD} = q_d{iD} + c_sphere(:,k)  - c_sphere(:,k-1);
        end

        % update radius of q_d
        q_d{iD} = projectToSphere(q_d{iD}, Rad{iD}, c_sphere(:,k));

        % --- Store
        Rad_hist{iD}(k) = Rad{iD};
        q_hist{iD}(:,k) = q{iD};
        q_d_hist{iD}(:,k) = q_d{iD};

        % get reference
        p_d =  q{iD};
        v_d = zeros(3,1);
        a_d = zeros(3,1);
        j_d = zeros(3,1);
        psi_d = 0;
        dpsi_d = 0;
    
        % mellinger controller
        [T{iD}(:,k),tau{iD}(:,k),e_p] = drone_mellinger_ctrl(pos{iD},...
            v{iD},R,om,Param,p_d,psi_d,xiep{iD}(:,k),v_d,dpsi_d,a_d,j_d);
        
        % integrate position error
        xiep{iD}(:,k+1) = xiep{iD}(:,k) + Param.dTi*e_p;
        
        % nonlinear drone model (continuous time)
        [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(v{iD},R,om,T{iD}(:,k),tau{iD}(:,k),Param);
        
        % discretization
        pp = pos{iD} + Param.dTi*dot_p;
        vp = v{iD} + Param.dTi*dot_v;
        Rp = rot_integrate(R,om,Param.dTi);
        omp = om + Param.dTi*dot_om;
        chaser{iD}(:,k+1) = [pp;vp;reshape(Rp,[],1);omp];
    
        % auxiliary drone attitude computation from rotation matrix
        lbd{iD}(:,k) = R2Euler(R);
    
        if abs(1-norm(Rp'*Rp))>1e-4
            warning(['Problems with rotation matrix integration... stoping simulation: t = ' num2str(k*Param.dTi) ' s.']);
            break;
        end
    end
end

%%
% figure;
% hold on;
% grid on;
% 
% t = (0:Nsim-1) * Param.dTi;
% 
% plot(t, Rad_hist{1}, 'LineWidth', 3);
% plot(t, vecnorm(target_hist(1:3,:)-chaser{1}(1:3,1:end-1), 2, 1), 'LineWidth', 3);
% plot(t, vecnorm(target_hist(4:6,:), 2, 1) ...
%        - vecnorm(chaser{1}(4:6,2:end), 2, 1), 'LineWidth', 3);
% 
% xlabel('Time [s]');
% lgd = legend( ...
%     'Radial history', ...
%     'Position difference (norm)', ...
%     'Velocity difference (norm)' ...
% );
% 
% lgd.FontSize = 12; 
% lgd.Interpreter = 'latex';

animate_simulation(chaser, q_hist, q_d_hist, fib_ref_points, c_sphere, Param, Nsim);