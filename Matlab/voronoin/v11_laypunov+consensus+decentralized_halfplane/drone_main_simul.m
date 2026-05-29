% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD;
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iD = 1:Param.nD
    % initial values
    Rad{iD} = Param.r0(iD);
    chaser{iD}(1:3,1) = initial_pos(:,iD);
    q{iD} = initial_pos(:,iD);
    omega{iD} = [0; 0; 0];
    
    % desired values
    q_d{iD} = projectToSphere(q{iD}, Rad{iD}, Param.initial_target_pos);
    
    omega_d{iD} = [0; 0; 0];
end

estimated_target_position = zeros(18,Param.nD);

% Storage
for iD = 1:Param.nD
    q_hist{iD} = zeros(3, Nsim);
    Rad_hist{iD} = zeros(1, Nsim);
    q_d_hist{iD} = zeros(3, Nsim);
end

c_sphere = zeros(3,Nsim);
target_true_hist = zeros(18,Nsim);
perceived_target_hist = cell(Nsim, Param.nD);
agents_history = cell(Nsim+1,Param.nD);
cells_history = cell(Nsim,Param.nD);
numAgents_history = zeros(Nsim,1);
avgAgentVal_history = zeros(Param.nD,Nsim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target drone initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial target state
target.x = zeros(18,1);
target.x(1:3)   = Param.initial_target_pos;    % position
target.x(4:6)   = [0;0;0];                     % velocity
target.x(7:15)  = reshape(eye(3),[],1);        % rotation matrix
target.x(16:18) = [0;0;0];                     % angular velocity

% Integral error
target.xiep = zeros(3,1);

ema_estimated = repmat(target.x, 1,Param.nD);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("Computing simulation interation: %04d\n", k);

    % drop some agents mid simulation
    if k == round(500)
        removeIdx = randperm(size(q_d,2), round(3));
        q_d(removeIdx) = [];
        new_q_d(removeIdx) = [];
        ema_estimated(:,removeIdx) = [];
        estimated_target_position(:,removeIdx) = [];
        Param.nD = size(q_d,2);
    end
    numAgents_history(k) = Param.nD;
    
    % move actual target
    t = (k-1)*Param.dTi;
    p_d_target = target_ref(t);
    [target, c_sphere(:,k)] = target_motion(target, p_d_target, Param);
    target_true_hist(:,k) = target.x;
    
    % --- SMALL WORLD NETWORK ---
    [connections,connection_Laplacian,~] = Delaunay_triangulation_network(cell2mat(q_d)');
    
    for iD = 1:Param.nD
        % get state vector
        pos{iD} = chaser{iD}(1:3,k);
        v{iD} = chaser{iD}(4:6,k);
        R = reshape(chaser{iD}(7:15,k),3,3);
        om = chaser{iD}(16:18,k);

        % get target estimated position
        noisey_target = target.x + [Param.noise_std*randn(3,1); zeros(length(target.x)-3, 1)];
        % exponential moving average filter
        ema_estimated(:,iD) = (1-Param.ema_alpha)*ema_estimated(:,iD) + Param.ema_alpha*noisey_target;
        % consensus update
        estimated_target_position(:,iD) = ema_estimated(:,iD) - Param.dTi*ema_estimated*connection_Laplacian(:,iD);
        c_sphere_estimated = estimated_target_position(1:3,iD);
        perceived_target_hist{k,iD} = estimated_target_position(:,iD);

        % Position reference generation
        [q{iD}, e_q] = linear_motion_ref(Rad{iD}, pos{iD},omega{iD},q_d{iD},omega_d{iD}, c_sphere_estimated, Param);
           
        % update radius os sphere
        Rad{iD} = update_radius(Rad{iD}, e_q, q{iD}, q_d{iD}, chaser{iD}(:,k), estimated_target_position(:,iD), Param);
        
        neighbours = find(connections(iD,:) > 0);
        [new_q_d{iD}, cell_verts] = voronoi_update_halfplane(iD, q_d, neighbours, c_sphere_estimated, Rad{iD});
        cells_history{k,iD}  = cell_verts;

        % --- build local point set ---
        neigh_pos = [q_d{iD}'; cell2mat(q_d(neighbours))'];
        neigh_pos_sphere = projectToSphere(neigh_pos', Param.sphere_radius, target.x(1:3));
        agents_dists = geo_distances(neigh_pos_sphere, Param.sphere_radius, target.x(1:3));
        sorted_agents_dists = sort(agents_dists,2);
        sorted_agents_dists(:,end) = [];
        avgAgentVal_history(iD,k) = mean(mean(sorted_agents_dists,2));
    
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
    q_d = new_q_d;
end

%%
figure;
hold on;
grid on;

t = (0:Nsim-1) * Param.dTi;

plot(t, Rad_hist{1}, 'LineWidth', 3);
plot(t, vecnorm(target_true_hist(1:3,:)-chaser{1}(1:3,1:end-1), 2, 1), 'LineWidth', 3);
plot(t, vecnorm(target_true_hist(4:6,:), 2, 1) ...
       - vecnorm(chaser{1}(4:6,2:end), 2, 1), 'LineWidth', 3);

xlabel('Time [s]');
lgd = legend( ...
    'Radial history', ...
    'Position difference (norm)', ...
    'Velocity difference (norm)' ...
);

lgd.FontSize = 12; 
lgd.Interpreter = 'latex';

print2pdf("decentralized_results_drone_1.pdf", 1)
%%
figure;
hold on;
grid on;
axis equal;
view(3);

% Plot true trajectory
plot3(target_true_hist(1,:), target_true_hist(2,:), target_true_hist(3,:), ...
      'k', 'LineWidth', 3);

[Ks, Is] = size(perceived_target_hist);

for i = 1:Is
    x = nan(1, Ks);
    y = nan(1, Ks);
    z = nan(1, Ks);

    for k = 1:Ks
        if ~isempty(perceived_target_hist{k,i})
            est = perceived_target_hist{k,i}; % expected 18x1
            x(k) = est(1);
            y(k) = est(2);
            z(k) = est(3);
        end
    end

    % Plot trajectory (NaNs automatically skipped)
    plot3(x, y, z, 'LineWidth', 2);

    % Find valid indices (where drone exists)
    valid_idx = find(~isnan(x));

    if ~isempty(valid_idx)
        % Start point
        scatter3(x(valid_idx(1)), y(valid_idx(1)), z(valid_idx(1)), ...
                 80, 'filled');

        % End point
        scatter3(x(valid_idx(end)), y(valid_idx(end)), z(valid_idx(end)), ...
                 80, 'd', 'filled');
    end
end

xlabel('X');
ylabel('Y');
zlabel('Z');

title('Consensus Tracking of Moving Target (EMA + Noisy Measurements)');

% Optional: dynamic legend
legend_entries = [{'True target'}, ...                  
    arrayfun(@(i) sprintf('Drone %d est', i), 1:Is, 'UniformOutput', false)];
legend(legend_entries{:});

%%
figure;
grid on;
hold on;

for iD = 1:Param.nD
    plot(1:Nsim, avgAgentVal_history(iD,:))
end

%%
animate_simulation(chaser, q_hist, q_d_hist, c_sphere, cells_history, numAgents_history, Param);
