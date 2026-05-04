% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)

% initialization
clear all;

% Model and simulation parameters
Param.Tend = 30;
Param.dTi = 0.01;  % inner-loop and simulation sampling period
Nsim = round(Param.Tend/Param.dTi)+1;
Param.Nsim = Nsim;
Param.g = 9.81;     % earth gravity
Param.nD = 12; % number of drones
Param.sphere_radius = 2.0;
Param.nD_drop = 3;
Param.nD_drop_time = 500;

% target parameters
t = linspace(0, 2*pi, 200);
Param.ref_target_traj = [-5 + 15*sin(t);...
             15*cos(t);...
             5*ones(1,length(t))];   % target trajectory
Param.initial_target_pos = Param.ref_target_traj(:,1);

% exponential moving target parameters
Param.ema_alpha = 0.2;
Param.noise_std = 0.05; % a 5% error on reading target position

% animation
Param.verbose = false;
Param.animation.animate = false;
Param.animation.trail_length = 150;
Param.animation.step = 10;
% === Create PDF ===
Param.animation.create_pdf = true;
% === Video settings ===
Param.animation.save_to_mp4 = false;
Param.animation.frameStep = 10;
Param.animation.sim_dt = Param.dTi * Param.animation.frameStep;
Param.animation.videoFPS = 1 / Param.animation.sim_dt;   

% M690B drone
% (guessing parameters! needs identification or estimation from geometry)
% from https://doi.org/10.23919/ACC45564.2020.9147948 we have: "The quadrotor’s mass is M = 1.73Kg and its arms length is L = 0.2m. The moments of inertia matrix used in the nonlinear model (3) is considered diagonal with values: Ixx0=Iyy0=0.03Kg.m2 and Izz0=0.04Kg.m2. "
% from https://dspace.mit.edu/handle/1721.1/106777 we have as the result of a bifilar pendulum experiment: "mass 𝑚 = 1.28 kg; inertia (𝐽𝑥𝑥, 𝐽𝑦𝑦, 𝐽𝑧𝑧) = (6.9𝐸−3, 7.0𝐸−3, 12.4𝐸−3) Ns2; propeller diameter 0.165 m; propeller positions [±0.117 ±0.117 −0.012] m
% from https://doi.org/10.1016/j.conengprac.2010.02.008 we have: m = 4.34±5×10−3 kg; IXX = 0.0820±0.0025 kg m2, IYY = 0.0845±0.0029 kg m2, IZZ = 0.1377±0.0059 kg m2; rotor radius r = 0.165 ±0.5×10−3   m; arm length 0.157 m, height 0.035 m.
% from https://doi.org/10.1109/CEIT.2018.8751873 we have: m = 7.293 kg; Ixx = 0.13, Iyy = 0.25, Izz = 0.55 kg m^2; propeller radius 0.41 or 0.2 m; arm radius 0.55 or 0.45 m.
Param.m = 5;        % drone mass
Param.I = diag([2e-2,2e-2,3e-2]);  % inertia tensor
Param.D = 0.00;     % frame drag coeficient
Param.kp = diag([150,150,150]);
Param.kv = diag([15,15,15]);
Param.ki = diag([0,0,0]);
Param.kR = diag([8,8,8]);
Param.kom= diag([0.5,0.5,0.5]);

% initialize variables for all drones:
for iD = 1:Param.nD
    % set initial conditions
    p0{iD} = [0;0;0];
    v0{iD} = [0;0;0];
    psi0{iD} = pi/20*((Param.nD-1)/2-iD+1);
    R0{iD} = Euler2R([0;0;psi0{iD}]);
    om0{iD} = [0;0;0];
    chaser{iD} = zeros(18,Nsim+1);
    xiep{iD} = zeros(3,Nsim+1);
    chaser{iD}(:,1) = [p0{iD};v0{iD};reshape(R0{iD},[],1);om0{iD}];
end
