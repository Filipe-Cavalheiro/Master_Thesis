% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all; 
addpath(genpath(pwd)); % load all sub folders
drone_init;            % initialize variables
clearvars iD;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state.vertices = [];
state.faces = [];
state.sites = [];

% get mesh
[v_circle, v_cone, v_sphere] = pre_compute_meshes(1001,Param);

mesh_pos = [0,0,0];
surf_direction = [1, 0, 0];

[state.vertices, state.faces, mesh_pos] = pos_based_mesh_transform( ...
    Param.initial_target_pos, ...
    mesh_pos, ...
    surf_direction, ...
    v_circle, ...
    v_cone, ...
    v_sphere, ...
    state.faces, ...
    Param);

% get initial desires_final_pos (p_d)
N = size(state.vertices,1);
state.sites = randperm(N, Param.nD);
p_d = state.vertices(state.sites(:), :)';

hFig = figure;
hold on; 
view(3);
axis equal;
camlight;
lighting gouraud;
shading interp
lightangle(-45,30)
h.AmbientStrength = 0.3;
h.DiffuseStrength = 0.8;
h.SpecularStrength = 0.9;
h.SpecularExponent = 25;

gfx.surf = trisurf(state.faces, state.vertices(:,1), state.vertices(:,2), state.vertices(:,3), ...
    zeros(size(state.vertices,1),1), ...
    'EdgeColor','none');
gfx.verbose = true;
gfx.sites = scatter3(nan, nan, nan, 60, 'g', 'x');
gfx.voronoi_edges = scatter3(nan, nan, nan, 60, 'r', 'x');

p_d_hist = cell(Param.nD, 1);
for iD = 1:Param.nD
    % initial values
    chaser{iD}(1:3,1) = p_d(:,iD);

    % Storage
    p_d_hist{iD} = zeros(3, Nsim);
end
estimated_target_variables = zeros(18,Param.nD);

target_true_hist = zeros(18,Nsim);
perceived_target_hist = cell(Nsim, Param.nD);
agents_history = cell(Nsim+1,Param.nD);
numAgents_history = zeros(Nsim,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target drone initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial target data.state
target.x = zeros(18,1);
target.x(1:3)   = Param.initial_target_pos;    % position
target.x(4:6)   = [0;0;0];                     % velocity
target.x(7:15)  = reshape(eye(3),[],1);        % rotation matrix
target.x(16:18) = [0;0;0];                     % angular velocity
target.xiep = zeros(3,1);                      % Integral error

ema_estimated = repmat(target.x, 1,Param.nD);
newSite = zeros(1, Param.nD);

active_mask = true(1, Param.nD);  % all active initially
active_ids = find(active_mask);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("Computing simulation interation: %04d\n", k);

    % drop some agents mid simulation
    if k == Param.nD_drop_time
        removeIdx_local = randperm(length(active_ids), round(Param.nD_drop));
        removeIdx = active_ids(removeIdx_local);
        active_mask(removeIdx) = false;
        active_ids = find(active_mask);
        ema_estimated(:, removeIdx) = [];
    end
    
    % move actual target
    t = (k-1)*Param.dTi;
    p_d_target = target_ref(Param.ref_target_traj, target.x(1:3));
    target = target_motion(target, p_d_target, Param);
    target_true_hist(:,k) = target.x;
    
    % --- SMALL WORLD NETWORK ---
    [connections,connection_Laplacian,~] = Delaunay_triangulation_network(p_d(:, active_ids)');

    % --- Voronoi update (centralized) ---
    % if mod(k,20) == 0 || k == 1
    %     for iD = 1:Param.nD
    %         tmp_sites = state.sites;
    %         [tmp_sites(iD), tmp_sites(1)] = deal(tmp_sites(1), tmp_sites(iD));
    %         newSite(iD) = runCVT(state, tmp_sites, gfx);
    %     end
    %     state.sites = newSite;
    % end
    % p_d = state.vertices(state.sites, :)';
    
    
    inactive_ids = find(~active_mask);
    for iD = inactive_ids
        % inactive drones will just fall from the sky
        p_d_hist{iD}(:,k) = NaN;
        pos = chaser{iD}(1:3,k);
        if pos(3) <= 0
            chaser{iD}(:,k+1) = [[pos(1); pos(2); 0]; zeros(15,1)];
        else
            v = chaser{iD}(4:6,k);
            R = reshape(chaser{iD}(7:15,k),3,3);
            om = chaser{iD}(16:18,k);
            [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(v,R,om,0,zeros(3,1),Param);
            % discretization
            pp = pos + Param.dTi*dot_p;
            vp = v + Param.dTi*dot_v;
            Rp = rot_integrate(R,om,Param.dTi);
            omp = om + Param.dTi*dot_om;
            chaser{iD}(:,k+1) = [pp;vp;reshape(Rp,[],1);omp];
        end
    end
        
    for iD = active_ids
        % get data.state vector
        pos = chaser{iD}(1:3,k);
        v = chaser{iD}(4:6,k);
        R = reshape(chaser{iD}(7:15,k),3,3);
        om = chaser{iD}(16:18,k);

        % get target estimated position
        noisey_target = target.x + [Param.noise_std*randn(3,1); zeros(length(target.x)-3, 1)];
        % exponential moving average filter
        laplacian_idx = find(active_ids == iD);
        ema_estimated(:,laplacian_idx) = (1-Param.ema_alpha)*ema_estimated(:,laplacian_idx) + Param.ema_alpha*noisey_target;
        % consensus update
        estimated_target_variables(:,iD) = ema_estimated(:,laplacian_idx)...
            - Param.dTi*ema_estimated*connection_Laplacian(:,laplacian_idx);
        estimated_target_position = estimated_target_variables(1:3,iD);
        perceived_target_hist{k,iD} = estimated_target_variables(:,iD);

        [state.vertices, state.faces, mesh_pos] = pos_based_mesh_transform( ...
            estimated_target_position, ...
            mesh_pos, ...
            surf_direction, ...
            v_circle, ...
            v_cone, ...
            v_sphere, ...
            state.faces, ...
            Param);

        neighbours = active_ids(connections(laplacian_idx,:));
        if mod(k,100) == 0 || k == 1
            sites = state.sites([iD, neighbours]);
            newSite(iD) = runCVT(state, sites, gfx);
        end
        p_d(:, iD) = state.vertices(state.sites(iD), :)';   

        % Store
        p_d_hist{iD}(:,k) = p_d(:, iD);

        % get reference
        drone_desired_pos =  pos+(p_d(:, iD)-pos)/norm(p_d(:, iD)-pos)*0.3;
        v_d = zeros(3,1);
        a_d = zeros(3,1);
        j_d = zeros(3,1);
        psi_d = 0;
        dpsi_d = 0;
    
        % mellinger controller
        [T,tau,e_p] = drone_mellinger_ctrl(pos,...
            v, R, om, Param, drone_desired_pos, psi_d,...
            xiep{iD}(:,k), v_d, dpsi_d, a_d, j_d);
        
        % integrate position error
        xiep{iD}(:,k+1) = xiep{iD}(:,k) + Param.dTi*e_p;
        
        % nonlinear drone model (continuous time)
        [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(v,R,om,T,tau,Param);
        
        % discretization
        pp = pos + Param.dTi*dot_p;
        vp = v + Param.dTi*dot_v;
        Rp = rot_integrate(R,om,Param.dTi);
        omp = om + Param.dTi*dot_om;
        chaser{iD}(:,k+1) = [pp;vp;reshape(Rp,[],1);omp];
        
        if abs(1-norm(Rp'*Rp))>1e-4
            warning(['Problems with rotation matrix integration... stoping simulation: t = ' num2str(k*Param.dTi) ' s.']);
            break;
        end
    end

    if any(isnan(p_d(:, active_ids)))
        save('log_vars.mat');
        error('A position value is NaN! exiting');
    end
    state.sites = newSite;
end

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

% dynamic legend
legend_entries = [{'True target'}, ...                  
    arrayfun(@(i) sprintf('Drone %d est', i), 1:Is, 'UniformOutput', false)];
legend(legend_entries{:});

%%
close all;
animate_simulation(chaser, target_true_hist(1:3,:),...
     numAgents_history, state.vertices, ...
     p_d_hist, Param);

% clean path after it runs so that I can modify folders
%rmpath(genpath(pwd))