% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)

% inicializations

clear all;

% get boat parameters
P = boat_param();

% Model and simulation parameters
P.Tend = 30;
P.dTi = 0.01;
P.Nsim = round(P.Tend/P.dTi)+1;
P.nD = 1;         % number of boats
P.dh = 0.05;      % safety height difference between drones
P.Rad = 0.5;      % radius of circle
P.omn = 2*pi/20;  % rotation frequency
P.dphase = -pi/12;% ref circle angular difference between drones

P.ref_mode = 2; % reference: 1 - line reference; 2 - circle
P.ctrl_mode = 1; % 0 - open loop; 1 - V and yaw PID control.



% % Gains for nonlinear controller TBD
%direction
P.kyaw_delta = 2;  % **Redução extrema para suavizar ao máximo (antes: 0.35 → 0.05)**
P.kom_delta = 1;   % **Diminuído quase ao mínimo para evitar oscilações (antes: 0.18 → 0.01)**
P.kdp_delta = 1;   % **Baixado para minimizar resposta exagerada (antes: 0.3 → 0.05)**
P.kdv_delta = 0;   % **Amortecimento mínimo para evitar qualquer oscilação (antes: 0.1 → 0.01)**

% surge
P.kp_n = 0.0;  % **Resposta proporcional extremamente reduzida**
P.kpi_n = 0.00; % **Termo integral muito menor para evitar crescimento acumulativo**
P.kV_n = 10.0;  % **Diminuição drástica da influência da velocidade**
P.kVi_n = 5; % **Amortecimento maior no erro acumulado**






% Param.kp = diag([8,8,6]);
% Param.kv = diag([5,5,5]);
% Param.ki = diag([0,0,0]);
% Param.kR = diag([8,8,8]);
% Param.kom= diag([0.5,0.5,0.5]);

% initial conditions
P.eta_ref_static = [0.5;0.5;pi/3];

% initialize variables for all drones:
t = 0:P.dTi:P.Tend;
nt = length(t);
nx = 6;
nctrl = 2;
nctrl_aux = 13;
for iD = 1:P.nD

    % set initial conditions
    eta0{iD} = [0;0.2*((P.nD-1)/2-iD+1);-pi/6];
    nu0{iD} = [1;0;0];
    x{iD} = zeros(nx,P.Nsim);
    eta{iD} = zeros(3,P.Nsim);
    nu{iD} = zeros(3,P.Nsim);
    xie{iD} = zeros(4,P.Nsim);
    ctrl{iD} = zeros(nctrl,P.Nsim);
    ctrl_aux{iD} = zeros(nctrl_aux,P.Nsim);
    x{iD}(:,1) = [eta0{iD};nu0{iD}];

    if iD == 1 % Set main initial condition
        P.eta0 = eta0{iD};
        P.nu0 = nu0{iD};
    end
    
    [eta_ref{iD},nu_ref{iD}] = get_boat_ref(t,P);
    eta_ref_all{iD} = [eta_ref{iD};nu_ref{iD}];

end

Param = P;