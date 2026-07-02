% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all; 
addpath(genpath(pwd)); % load all sub folders

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Param = Param_init();

voronoi.vertices = [];
voronoi.faces = [];
voronoi.sites = [];
voronoi.mesh_pos = [];
voronoi.newSite = zeros(1, Param.nD);
voronoi.surf_direction = [1, 0, 0];

[origin_p_d, swarm_center, voronoi] = pre_compute_desired_pos(voronoi, Param);

if strcmp(Param.distribuition, 'dijkstra')
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
    
    gfx.surf = trisurf(voronoi.faces, voronoi.vertices(:,1), voronoi.vertices(:,2), voronoi.vertices(:,3), ...
        zeros(size(voronoi.vertices,1),1), ...
        'EdgeColor','none');
    gfx.verbose = true;
    gfx.sites = scatter3(nan, nan, nan, 60, 'g', 'x');
    gfx.voronoi_edges = scatter3(nan, nan, nan, 60, 'r', 'x');
else
    gfx = NaN;
end

p_d_hist = cell(Param.nD, 1);

% Preallocate cell array
chaser = cell(1, Param.nD);

% Template struct
template = struct( ...
    'pos',      zeros(3, Param.Nsim+1), ...
    'vel',      zeros(3, Param.Nsim+1), ...
    'attitude', zeros(9, Param.Nsim+1), ...
    'ang_vel',  zeros(3, Param.Nsim+1), ...
    'ei',       zeros(3, Param.Nsim+1), ...
    'eI',       zeros(3, Param.Nsim+1));

% Allocate all drones at once
[chaser{:}] = deal(template);
for iD = 1:Param.nD
    chaser{iD}.pos(:,1) = origin_p_d(:,iD) + swarm_center;
    desired.pos(:,iD) = chaser{iD}.pos(:,1);
    chaser{iD}.vel(:,1) = [0;0;0];
    chaser{iD}.attitude(:,1) = reshape(Euler2R([0;0;0]),[],1);
    chaser{iD}.ang_vel(:,1) = [0;0;0];
    chaser{iD}.ei(:,1) = [0;0;0];
    chaser{iD}.eI(:,1) = [0;0;0];
    p_d_hist{iD} = zeros(3, Param.Nsim);
end

target_true_hist = zeros(24,Param.Nsim);
estimated_target_variables = zeros(3,Param.nD);
perceived_target_hist = cell(Param.Nsim, Param.nD);
numAgents_history = zeros(Param.Nsim,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target drone initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial target state
target = deal(template);
target.pos   = Param.initial_target_pos;    % position
target.vel   = [0;0;0];                     % velocity
target.attitude  = reshape(eye(3),[],1);        % rotation matrix
target.ang_vel = [0;0;0];                     % angular velocity
target.ei = zeros(3,1);
target.eI = zeros(3,1);

ema_estimated = repmat(target.pos, 1,Param.nD);

active_mask = true(1, Param.nD);  % all active initially
active_ids = find(active_mask);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Param.Nsim
    Sim.k = k;
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
    [p_d_target, v_d_target] = target_ref(Param.dTi*k, Param.ref_target_traj, Param);
    target = target_motion(target, p_d_target, v_d_target, Param);
    tmp = cell2mat(struct2cell(target)');
    target_true_hist(:,k) = tmp(:);
    
    % --- SMALL WORLD NETWORK ---
    [connections,connection_Laplacian,~] = Delaunay_triangulation_network(desired.pos(:, active_ids)');
    inactive_ids = find(~active_mask);
    for iD = inactive_ids
        % inactive drones will just fall from the sky
        p_d_hist{iD}(:,k) = NaN;
        pos = chaser{iD}.pos(1:3,k);
        if pos(3) <= 0
            chaser{iD}.pos(:,k+1) = [pos(1); pos(2); 0];
            chaser{iD}.vel(:,k+1) = zeros(3,1);
            chaser{iD}.attitude(:,k+1) = zeros(9,1);
            chaser{iD}.ang_vel(:,k+1) = zeros(3,1);
        else
            vel = chaser{iD}.vel(:,k);
            attitude = reshape(chaser{iD}.attitude(:,k),3,3);
            om = chaser{iD}.ang_vel(:,k);
            [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(vel,attitude,om,0,zeros(3,1),Param);
            % discretization
            chaser{iD}.pos(:,k+1) = pos + Param.dTi*dot_p;
            chaser{iD}.vel(:,k+1) = vel + Param.dTi*dot_v;
            Rp = rot_integrate(attitude,om,Param.dTi);
            chaser{iD}.attitude(:,k+1) = reshape(Rp,[],1);
            chaser{iD}.ang_vel(:,k+1) = om + Param.dTi*dot_om;
        end
    end

    for iD = active_ids
        Sim.iD = iD;
        % get data.state vector
        pos = chaser{iD}.pos(:,k);
        vel = chaser{iD}.vel(:,k);
        attitude = reshape(chaser{iD}.attitude(:,k),3,3);
        om = chaser{iD}.ang_vel(:,k);
        
        % Estimate target position
        active_ids_idx = find(active_ids == iD);
        [estimated_target_variables(:,iD), ema_estimated] = estimate_target_variables(...
            target, iD, ema_estimated, ...
            active_ids_idx, connection_Laplacian, Param);
        perceived_target_hist{k,iD} = [estimated_target_variables(:,iD); target_true_hist(4:18,k)];

        % get desired chaser state
        neighbours = active_ids(connections(active_ids_idx,:));
        [desired, voronoi] = compute_desired_state(k, iD, gfx, ...
            desired, ...
            origin_p_d(:,iD), voronoi, ...
            neighbours, ...
            perceived_target_hist, ...
            Param);
        p_d_hist{iD}(:,k) = desired.pos(:,iD); % Store

        chaser{iD} = actuate_control(chaser{iD}, desired, Sim, Param);
    end

    if any(isnan(origin_p_d(:, active_ids)))
        save('log_vars.mat');
        error('A position value is NaN! exiting');
    end
    voronoi.sites = voronoi.newSite;
end

%% animations PDF and mp4
close all; clc;

Param.Nsim

data.chaser = chaser;
data.target_position = target_true_hist;
data.removed_initial_pos = false;
data.p_d_hist = p_d_hist; 
data.perceived_target_hist = perceived_target_hist;
data.colors = lines(Param.nD);
data.active_ids = active_ids;
data.numAgents_history = numAgents_history;

create_figures(data, Param);
% clean path after it runs so that I can modify folders
rmpath(genpath(pwd))