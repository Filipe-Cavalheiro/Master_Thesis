% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD;

sim_timer_start = tic;             % --- start overall computation timer ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fibonacci sphere values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%calculate Fibonacci points for the unit sphere (the radius is applied
%after
fib_ref_points= get_fibonacci_points(Param.nD, Param.sphere_radius);
fib_ref_points = fib_ref_points';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign each agent to nearest ref point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute geodesic distances between all pairs
pos_on_sphere = projectToSphere(initial_pos, Param.sphere_radius);
pos_on_sphere = pos_on_sphere(:, 1:Param.nD);
geo_distances_all = zeros(Param.nD, Param.nD);
for i = 1:Param.nD
    for j = 1:Param.nD
        a = fib_ref_points(:,i);
        b = pos_on_sphere(:,j);
        dot_product = dot(a, b);
        geo_distances_all(i, j) = Param.sphere_radius * acos(dot_product / (Param.sphere_radius^2));
    end
end

%the theoretical maximum two points will be apart is pi so 10xpi is just to
%force full one-to-one assignment
costofnonassignment = 10*pi;
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

% %target motion reference
% target_fun = @(t) [ ...
%     (t<=1.5).*3*t + (t>1.5 & t<=3).*3*t + (t>3).*9 ; ...
%     (t<=1.5).*0    + (t>1.5 & t<=3).*(-3*(t-1.5)) + (t>3).*(-4.5) ; ...
%     zeros(size(t)) ];

target_fun = @(t) [ ...
    0 ; ...
    0 ; ...
    0];

for iD = 1:Param.nD
    % initial values
    Rad{iD} = organized_Rad(iD);
    x{iD}(1:3,1) = organized_initial_pos(:,iD);
    q{iD} = organized_initial_pos(:,iD);
    omega{iD} = [0; 0; 0];
    
    % desired values
    q_d{iD} = fib_ref_points(:,iD)/norm(fib_ref_points(:,iD))*Rad{iD};
    
    omega_d{iD} = [0; 0; 0];
end

% Storage
for iD = 1:Param.nD
    q_hist{iD} = zeros(3, Nsim);
    Rad_hist{iD} = zeros(1, Nsim);
    q_d_hist{iD} = zeros(3, Nsim);
end
c_sphere = zeros(3,Nsim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performance metrics initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CONV_THRESHOLD = 0.1;              % [m] convergence tolerance per agent
convergence_time = NaN;            % will hold the time (s) when all agents
                                    % are simultaneously within threshold
convergence_found = false;
error_hist = zeros(Param.nD, Nsim); % ||pos - q_d|| per agent per step
conv_error_hist = zeros(Param.nD, Nsim); % ||pos - p_final|| error vs each agent's FINAL lattice point

final_ref = zeros(3, Param.nD);
for iD = 1:Param.nD
    final_ref(:, iD) = fib_ref_points(:,iD);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("Computing simulation interation: %04d\n", k);
    
    t = (k-1)*Param.dTi;
    c_sphere(:,k) = target_fun(t);

    for iD = 1:Param.nD
        % get state vector
        pos{iD} = x{iD}(1:3,k);
        v{iD} = x{iD}(4:6,k);
        R = reshape(x{iD}(7:15,k),3,3);
        om = x{iD}(16:18,k);

        % Position reference generation
        q{iD} = linear_motion_ref(Rad{iD}, pos{iD},omega{iD},q_d{iD},omega_d{iD},c_sphere(:,k), Param);

        % Reduce q_d radius
        [Rad{iD}, q_d{iD}] = update_radius(Rad{iD}, q{iD}, q_d{iD}, c_sphere(:,k), Param);

        % --- Store ---
        Rad_hist{iD}(k) = Rad{iD};
        q_hist{iD}(:,k) = q{iD};
        q_d_hist{iD}(:,k) = q_d{iD};

        % --- Track per-agent tracking error for convergence measurement
        error_hist(iD,k) = norm(pos{iD} - q_d{iD});

        % --- Convergence error against the FIXED final lattice point
        conv_error_hist(iD,k) = norm(pos{iD} - final_ref(:,iD));

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
        
        % % discretization
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

    % --- Check convergence: all agents within threshold OF THEIR FINAL TARGET
    if ~convergence_found && all(conv_error_hist(:,k) < CONV_THRESHOLD)
        convergence_time = (k-1)*Param.dTi;
        convergence_found = true;
    end
end

sim_computation_time = toc(sim_timer_start);  % --- stop overall computation timer ---

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performance metrics reporting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
final_error_per_agent = error_hist(:,Nsim-1);       % error at final sim step
final_error_max  = max(final_error_per_agent);
final_error_mean = mean(final_error_per_agent);
conv_final_error_per_agent = conv_error_hist(:,Nsim-1);       % error at final sim step

% --- Final average pairwise distance between agents (Euclidean, 3D)
final_pos = zeros(3, Param.nD);
for iD = 1:Param.nD
    final_pos(:, iD) = x{iD}(1:3, Nsim-1);
end
pairwise_dists = pdist(final_pos');   % Euclidean distance between all unique pairs
final_avg_agent_distance = mean(pairwise_dists);

fprintf('\n--- Simulation Performance Metrics (Lyapunov)---\n');
if convergence_found
    fprintf('Convergence time (all agents < %.2f m): %.3f s\n', CONV_THRESHOLD, convergence_time);
else
    fprintf('Convergence time (all agents < %.2f m): NOT reached within %.3f s\n', CONV_THRESHOLD, Nsim*Param.dTi);
    fprintf('Final error per agent (objective tracking)[m]:\n');
    disp(conv_error_hist(:,end)');
end
fprintf('Final error per agent (tracking goal) [m]:\n');
disp(conv_final_error_per_agent');
fprintf('Final error per agent (controller reference tracking error)[m]:\n');
disp(final_error_per_agent');
fprintf('Final error (controller reference tracking error) (max)  : %.4f m\n', final_error_max);
fprintf('Final error (controller reference tracking error) (mean) : %.4f m\n', final_error_mean);
fprintf('Final average pairwise distance between agents: %.4f m\n', final_avg_agent_distance);
fprintf('Total computation time for %.1f s simulation: %.3f s\n', Nsim*Param.dTi, sim_computation_time);
fprintf('---------------------------------------\n');

%%
animate_simulation(x, q_hist, q_d_hist, c_sphere, fib_ref_points, Param, Nsim)