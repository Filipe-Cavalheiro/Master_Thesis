% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)

% inicializations

clear all;

% Simulation parameters
Param.Tend = 30;
Param.dTi = 0.001;
Param.Nsim = round(Param.Tend/Param.dTi)+1;

% EasyGlider drone
% put parameters into structure
Param.g = 9.81; % earth gravity
Param.rho=1.225;
Param.m = 1.9;
Param.S = 0.32;
Param.c_ = 0.3;
Param.b = 1.2;
Ix = 8.94e-2;
Iy = 1.44e-1;
Iz = 1.62e-1;
Ixz = 1.4e-2;
Param.I = [Ix 0 -Ixz; 0 Iy 0; -Ixz 0 Iz];

Param.omega = 509; % rotor angular velocity
Param.Ie = 1.3e-4; % rotor inertia

Param.Cm0 = 0.135;
Param.Cma = -1.5;
Param.Cmde = -1.13;
Param.cmdota = -10.4;
Param.Cmq = -50.8;
%Param.Cmq = -1;

Param.Cnb = 0.0344;
Param.Cnda = -0.012;
Param.Cndr = -0.0345;
Param.Cnp = -0.075;
Param.Cnr = -0.411;

Param.Clb = -0.04;
Param.Clda = 0.0677;
Param.Cldr = 0.0168;
Param.Clp = -0.414;
Param.Clr = 0.399;

%% Controller gains

% Param.mu_phi = 1.5;
% Param.mu_theta = 0.7;
% Param.mu_psi = 1;
% Param.mu_q = 1;
% Param.mu_p = 3;
% Param.mu_r = 0.05;

Param.mu_phi = 10;
Param.mu_theta = 5;
Param.mu_psi = 5;
Param.mu_p = 20;
Param.mu_q = 15;
Param.mu_r = 7;

% Param.de = deg2rad(9.1e-2);
% Param.da = deg2rad(1.01e-2);
% Param.dr = deg2rad(-6.7e-2);

%% initialize variables for all drones:
Param.om0 = [0;0;0];
phi0 = deg2rad(-6);
theta0 = deg2rad(-6);
psi0 = deg2rad(-6);
%Param.psi0 = deg2rad(0);
lbd0 = [phi0;theta0;psi0];
Param.R0 = Euler2R(lbd0);

%% set reference
phi_b = deg2rad(5);
theta_b = deg2rad(5);
psi_b = deg2rad(5);
Param.lbd_b = [phi_b;theta_b;psi_b];
Param.v_air = [17;0.03;0.37];
Param.alpha = deg2rad(1.24);
Param.beta = deg2rad(0.1);

