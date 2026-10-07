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
    if k == round(Nsim/3)
        removeIdx = randperm(size(q_d,2), round(Param.nD/3));
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

legend_size = 22;
legend_size_dim = 16;

%% Target Estimation
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

%% Distance to target for all drones
t = (0:Nsim-1) * Param.dTi;                  % [s]

figure('Color','w','Position',[100 100 900 600]);
hold on; grid on; box on;

colors = lines(Param.nD);
dist_target = zeros(Param.nD, Nsim);
for iD = 1:Param.nD
    dist_target(iD,:) = vecnorm(target_true_hist(1:3,:) - chaser{iD}(1:3,1:Nsim), 2, 1);
    plot(t, dist_target(iD,:), 'Color', colors(iD,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('Drone %d', iD));
end

% Mean over all drones
plot(t, mean(dist_target,1), 'k--', 'LineWidth', 3, 'DisplayName', 'Mean');

% Sphere radius reference (constant)
plot(t, Param.sphere_radius*ones(size(t)), 'r:', 'LineWidth', 3, ...
     'DisplayName', '$r^\star$');

set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex');
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', legend_size);
ylabel('Distance to target [m]', 'Interpreter', 'latex', 'FontSize', legend_size);

lgd = legend('Location', 'best');
lgd.FontSize = legend_size_dim;
lgd.Interpreter = 'latex';

print2pdf("radial_hist_12_agents", 1)

%% Complete trajectory over 30 s
t_end = 30;                                  % [s]
t_vec = (0:Nsim-1) * Param.dTi;
idx   = find(t_vec <= t_end);                % samples within 0-30 s

figure('Color','w','Position',[100 100 900 700]);
hold on; grid on; box on; axis equal; view(3);

% True target trajectory
h_target = plot3(target_true_hist(1,idx), ...
                 target_true_hist(2,idx), ...
                 target_true_hist(3,idx), ...
                 'k', 'LineWidth', 3);

% Chaser trajectories
colors = lines(Param.nD);
for iD = 1:Param.nD
    pos_hist = chaser{iD}(1:3, idx);         % chaser has Nsim+1 columns
    h_ch = plot3(pos_hist(1,:), pos_hist(2,:), pos_hist(3,:), ...
                 'Color', colors(iD,:), 'LineWidth', 1.5);
    if iD == 1
        h_chaser = h_ch;                     % keep only one legend entry
    else
        set(h_ch, 'HandleVisibility', 'off');
    end

    % chaser start (circle) and end (diamond)
    scatter3(pos_hist(1,1),   pos_hist(2,1),   pos_hist(3,1),   60, ...
             colors(iD,:), 'filled', 'HandleVisibility','off');
    scatter3(pos_hist(1,end), pos_hist(2,end), pos_hist(3,end), 60, ...
             colors(iD,:), 'd', 'filled', 'HandleVisibility','off');
end
scatter3(target_true_hist(1,end), target_true_hist(2,end), ...
    target_true_hist(3,end), 60, 'k', 'd', 'filled', 'HandleVisibility','off');

% Axes: visible and labelled
set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex');
xlabel('$x$ [m]', 'Interpreter', 'latex', 'FontSize', legend_size);
ylabel('$y$ [m]', 'Interpreter', 'latex', 'FontSize', legend_size);
zlabel('$z$ [m]', 'Interpreter', 'latex', 'FontSize', legend_size);

% Legend: handles and labels must match in number
lgd = legend([h_target h_chaser], ...
    {'Target (true)', 'Chasers'}, ...
    'Interpreter', 'latex', 'FontSize', legend_size_dim, 'Location', 'best');

print2pdf("trajectory_12_agents", 1)
%% Tracking error: drone position vs. Mellinger reference q{iD}
t_vec = (0:Nsim-1) * Param.dTi;              % [s]

err_norm = zeros(Param.nD, Nsim);
for iD = 1:Param.nD
    pos_hist = chaser{iD}(1:3, 1:Nsim);      % chaser has Nsim+1 columns
    e_track  = pos_hist - q_hist{iD};        % 3 x Nsim, position - reference
    err_norm(iD,:) = vecnorm(e_track, 2, 1);
end

figure('Color','w','Position',[100 100 900 600]);
hold on; grid on; box on;

colors = lines(Param.nD);
for iD = 1:Param.nD
    plot(t_vec, err_norm(iD,:), 'Color', colors(iD,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Drone %d', iD));
end
plot(t_vec, mean(err_norm,1), 'k--', 'LineWidth', 3, ...
     'DisplayName', 'Mean');

set(gca, 'FontSize', 14, 'LineWidth', 0.5, 'TickLabelInterpreter', 'latex');
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', legend_size);
ylabel('$\|\mathbf{p}_i - \mathbf{q}_i\|$ [m]', 'Interpreter', 'latex', 'FontSize', legend_size);
legend('Interpreter', 'latex', 'FontSize', legend_size_dim, 'Location', 'best');

print2pdf("tracking_error_12_agents", 1)

%% Inter-agent Euclidean distance
t_vec = (0:Nsim-1) * Param.dTi;              % [s]
nD    = Param.nD;

pairs = nchoosek(1:nD, 2);                   % all unique drone pairs
nP    = size(pairs, 1);

dist_pairs = zeros(nP, Nsim);
for p = 1:nP
    i = pairs(p,1);
    j = pairs(p,2);
    dist_pairs(p,:) = vecnorm(chaser{i}(1:3,1:Nsim) - chaser{j}(1:3,1:Nsim), 2, 1);
end

figure('Color','w','Position',[100 100 900 600]);
hold on; grid on; box on;

% Every pair (thin, light grey, single legend entry)
h_all = plot(t_vec, dist_pairs(1,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 1, ...
             'DisplayName', 'Individual pairs');
for p = 2:nP
    plot(t_vec, dist_pairs(p,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 1, ...
         'HandleVisibility', 'off');
end

% Summary statistics across pairs
plot(t_vec, max(dist_pairs,[],1),  'r-',  'LineWidth', 2.5, 'DisplayName', 'Maximum');
plot(t_vec, mean(dist_pairs,1),    'k--', 'LineWidth', 3,   'DisplayName', 'Mean');
plot(t_vec, min(dist_pairs,[],1),  'b-',  'LineWidth', 2.5, 'DisplayName', 'Minimum');

set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex');
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', legend_size);
ylabel('$\|\mathbf{p}_i - \mathbf{p}_j\|$ [m]', 'Interpreter', 'latex', 'FontSize', legend_size);
legend('Interpreter', 'latex', 'FontSize', legend_size_dim, 'Location', 'best');

print2pdf("inter_agent_dist_12_agents", 1)

%% Separation / uniformity of desired positions q_d over time
t_vec = (0:Nsim-1) * Param.dTi;              % [s]

ratio_hist = zeros(1,Nsim);
cv_hist    = zeros(1,Nsim);
meanNN     = zeros(1,Nsim);
minNN      = zeros(1,Nsim);
maxNN      = zeros(1,Nsim);
riesz_hist = zeros(1,Nsim);

for k = 1:Nsim
    % 3 x nD matrix of desired positions at step k
    Qd_k = zeros(3, Param.nD);
    for iD = 1:Param.nD
        Qd_k(:,iD) = q_d_hist{iD}(:,k);
    end

    r = sphere_uniformity(Qd_k, ...
            'Center', target_true_hist(1:3,k), ...
            'Radius', Param.sphere_radius);

    ratio_hist(k) = r.ratio_min_mean;
    cv_hist(k)    = r.cv_nn;
    meanNN(k)     = r.mean_nn;
    minNN(k)      = r.min_nn;
    maxNN(k)      = r.max_nn;
    riesz_hist(k) = r.riesz_energy;
end

figure('Color','w','Position',[100 100 900 800]);

% (1) nearest-neighbour separation in metres
subplot(3,1,1); hold on; grid on; box on;
fill([t_vec fliplr(t_vec)], [minNN fliplr(maxNN)], [0.85 0.85 0.85], ...
     'EdgeColor','none','DisplayName','Min--Max');
plot(t_vec, meanNN, 'k--', 'LineWidth', 3, 'DisplayName', 'Mean');
plot(t_vec, minNN,  'b-',  'LineWidth', 2, 'DisplayName', 'Minimum');
plot(t_vec, maxNN,  'r-',  'LineWidth', 2, 'DisplayName', 'Maximum');
set(gca,'FontSize',14,'LineWidth',1.5,'TickLabelInterpreter','latex');
ylabel('$d_{NN}$ [m]','Interpreter','latex','FontSize',legend_size);
legend('Interpreter','latex','FontSize',legend_size_dim,'Location','best');

% (2) uniformity ratio
subplot(3,1,2); hold on; grid on; box on;
plot(t_vec, ratio_hist, 'LineWidth', 2.5);
set(gca,'FontSize',14,'LineWidth',1.5,'TickLabelInterpreter','latex');
ylabel('$d_{NN,min}/d_{NN,mean}$','Interpreter','latex','FontSize',legend_size);
ylim([0 1.05]);

% (3) coefficient of variation
subplot(3,1,3); hold on; grid on; box on;
plot(t_vec, cv_hist, 'LineWidth', 2.5);
set(gca,'FontSize',14,'LineWidth',1.5,'TickLabelInterpreter','latex');
xlabel('Time [s]','Interpreter','latex','FontSize',legend_size);
ylabel('CV$(d_{NN})$','Interpreter','latex','FontSize',legend_size);

print2pdf("qd_separation_12_agents", 1)
%%
% animate_simulation(chaser, q_hist, q_d_hist, c_sphere, cells_history, numAgents_history, Param);
