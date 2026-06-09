% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)

% inicializations

clear all;

% Model and simulation parameters
Param.Tend = 10;
Param.dTi = 0.01;  % inner-loop and simulation sampling period
Nsim = round(Param.Tend/Param.dTi)+1;
Param.g = 9.81;     % earth gravity
Param.nD = 3; % number of drones
Param.sphere_radius = 5;  % sphere radius
Param.k_rad = 0.7; % radius reduction gain

% Outer-loop parameters
Param.c = 0.3;                             % Nm (s/rad)
Param.J = diag([0.0294 0.0305 0.0495]);    % kg m^2
Param.tau = zeros(3,1);
Param.K_r = diag([70 70 70]);
Param.K_w = diag([7.056 7.320 11.88]) * 1e-1;

% reference parameters
Param.p_ref_static = [1;1;3];
Param.psi_ref_static = pi/3;

Param.vz_d = 0.1;
Param.dh = 0.05;      % safety height difference between drones
Param.Rad = 5;        % radius of circle
Param.omn = 2*pi/20;  % rotation frequency
Param.dphase = -pi/12;% ref circle angular difference between drones

Param.ref_mode = 2; % reference: 1 - square wave; 2 - circle

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

if Param.nD <= 9
    theta0   = [0.175 0.698 1.222 1.745 2.269 2.793 3.491 4.189 4.712];
    phi0     = [1.484 1.222 1.396 1.047 1.484 1.222 0.698 1.396 0.960];
    Param.r0 = [11 6 8 20 15 30 25 20 15];
else
    theta0   = 2*pi*rand(1,Param.nD);
    phi0     = 2*pi*rand(1,Param.nD);
    Param.r0 = 15*rand(1, Param.nD);
end 

initial_pos = Param.r0.* [sin(theta0).*cos(phi0);
                sin(theta0).*sin(phi0);
                cos(theta0)];

% initialize variables for all drones:
t = 0:Param.dTi:Param.Tend;
nt = length(t);
nx = 18;
nu = 4;
for iD = 1:Param.nD

    % set initial conditions
    p0{iD} = initial_pos(:, iD);
    v0{iD} = [0;0;0];
    psi0{iD} = pi/20*((Param.nD-1)/2-iD+1);
    R0{iD} = Euler2R([0;0;psi0{iD}]);
    om0{iD} = [0;0;0];
    x{iD} = zeros(nx,Nsim+1);
    xiep{iD} = zeros(3,Nsim+1);
    T{iD} = zeros(1,Nsim);
    tau{iD} = zeros(3,Nsim);
    lbd{iD} = zeros(3,Nsim);
    x{iD}(:,1) = [p0{iD};v0{iD};reshape(R0{iD},[],1);om0{iD}];
    
    if Param.ref_mode == 1 % square wave reference
        t_step = 10;
        sw = mod(t,t_step)>=t_step/2; % boolean wave
        p_ref{iD} = p0{iD}*ones(1,Tend/Param.dTi+1) + p_ref_static*sw;
        v_ref{iD} = zeros(3,nt);
        a_ref{iD} = zeros(3,nt);
        j_ref{iD} = zeros(3,nt);
        psi_ref{iD} = psi0{iD}*ones(1,Tend/Param.dTi+1) + psi_ref_static*sw;
        dpsi_ref{iD} = zeros(1,Tend/Param.dTi+1);
        
    else % circle reference
        phase{iD} = (iD-1)*Param.dphase;
%         p_ref{iD} = [Param.Rad*cos(Param.omn*t+phase{iD});Param.Rad*sin(Param.omn*t+phase{iD});(1+dh*(iD-1))*ones(size(t))];
%         v_ref{iD} = [-Param.Rad*Param.omn*sin(omn*t+phase{iD});Param.Rad*Param.omn*cos(Param.omn*t+phase{iD});0*ones(size(t))];
        p_ref{iD} = [Param.Rad*cos(Param.omn*t+phase{iD});Param.Rad*sin(Param.omn*t+phase{iD});Param.vz_d*t+Param.p_ref_static(3)];
        v_ref{iD} = [-Param.Rad*Param.omn*sin(Param.omn*t+phase{iD});Param.Rad*Param.omn*cos(Param.omn*t+phase{iD});Param.vz_d*ones(size(t))];
        a_ref{iD} = [-Param.Rad*Param.omn^2*cos(Param.omn*t+phase{iD});-Param.Rad*Param.omn^2*sin(Param.omn*t+phase{iD});0*ones(size(t))];
        j_ref{iD} = [ Param.Rad*Param.omn^3*sin(Param.omn*t+phase{iD});-Param.Rad*Param.omn^3*cos(Param.omn*t+phase{iD});0*ones(size(t))];
        psi_ref{iD} = atan2(v_ref{iD}(2,:),v_ref{iD}(1,:));
        dpsi_ref{iD} = Param.omn*ones(size(psi_ref{iD}));
    end

    p_ref_all{iD} = [p_ref{iD};v_ref{iD};a_ref{iD};j_ref{iD}];
    psi_ref_all{iD} = [psi_ref{iD};dpsi_ref{iD}];
    
    % prepare data for simulink
    Param.("p0_" + num2str(iD)) = p0{iD};
    Param.("v0_" + num2str(iD)) = v0{iD};
    Param.("psi0_" + num2str(iD)) = psi0{iD};
    Param.("R0_" + num2str(iD)) = R0{iD};
    Param.("om0_" + num2str(iD)) = om0{iD};
end
