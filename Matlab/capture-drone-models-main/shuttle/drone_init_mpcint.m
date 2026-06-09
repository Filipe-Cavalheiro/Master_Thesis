% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)

% inicializations

clear all;
addpath('D:\Work\Research\Repositories\CasADi\casadi-3.6.3-windows64-matlab2018b')
import casadi.*

% Model and simulation parameters
Param.Tend = 60;
Param.dTi = 0.001;  % inner-loop and simulation sampling period
Param.dTo = 0.1;    % outer-loop sampling period
Nsim = round(Param.Tend/Param.dTi)+1;
Param.g = 9.81;     % earth gravity
Param.nD = 2; % number of drones

% reference parameters
Param.p_ref_static = [1;1;3];
Param.psi_ref_static = pi/3;

Param.vz_d = 0.1;
Param.dh = 0.05;      % safety height difference between drones
Param.Rad = 5;        % radius of circle
Param.omn = 2*pi/20;  % rotation frequency
Param.dphase = -pi/12;% ref circle angular difference between drones

Param.ref_mode = 2; % reference: 1 - square wave; 2 - circle

% M690B drone parameters and Mellinger controller gains
% (guessing parameters! needs identification)
Param.m = 5;        % drone mass (added board)
Param.I = diag([2e-2,2e-2,3e-2]);  % inertia tensor
Param.D = 0.00;     % frame drag coeficient
Param.kp = diag([15,15,15]);
Param.kv = diag([15,15,15]);
Param.ki = diag([0,0,0]);
Param.kR = diag([8,8,8]);
Param.kom= diag([0.5,0.5,0.5]);

% initialize variables for all drones:
t = 0:Param.dTi:Param.Tend;
nt = length(t);
nx = 18;
nu = 4;
for iD = 1:Param.nD

    % set initial conditions
    p0{iD} = [0;0.2*((Param.nD-1)/2-iD+1);0];
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

    %compute CPU times for each controller
    tCPU{iD} = zeros(1,Nsim);
    dtCPU{iD} = zeros(1,Nsim);
    
    % prepare data for simulink
    Param.("p0_" + num2str(iD)) = p0{iD};
    Param.("v0_" + num2str(iD)) = v0{iD};
    Param.("psi0_" + num2str(iD)) = psi0{iD};
    Param.("R0_" + num2str(iD)) = R0{iD};
    Param.("om0_" + num2str(iD)) = om0{iD};
end

% M690B drone MPC controller definition
% state
pos = MX.sym('pos',3);
vel = MX.sym('vel',3);
psi = MX.sym('psi',1);
epos_int = MX.sym('epos_int',3); % position error integral state
epsi_int = MX.sym('epsi_int',1); % yaw error integral state
xx = [pos;vel;psi;epos_int;epsi_int];
n_states = size(xx,1);
% control
acc = MX.sym('acc',3);
omegaz = MX.sym('omegaz',1);
epos = MX.sym('epos',3); % position error as external input
epsi = MX.sym('epsi',1); % yaw error as external input
uu = [acc;omegaz;epos;epsi];
n_controls = size(uu,1);
u_min = [-20; -20; -20; -pi; -1e3; -1e3; -1e3; -1e3]; u_max = [20; 20; 20; pi; 1e3; 1e3; 1e3; 1e3];
% Simplified Nonlinear ODE (Ordinary Differential Equation)
xdot = [vel; acc; omegaz; epos; epsi];
f_cont = Function('f_cont',{xx,uu},{xdot},{'x','u'},{'ode'}); % last two represent labels for better understanding
% controller parameters
Param.T = 10; % Time horizon
Param.N = 20; % Number of control intervals
Param.Q = blkdiag(1,1,1,2,2,2,0.1,1e-4,1e-4,1e-4,1e-4)*10;
Param.R = blkdiag(0.1,0.1,0.1,0.5,1e-4,1e-4,1e-4,1e-4);
Param.P = 10*Param.Q;
% Integrator to discretize the system
intg_options = struct;
intg_options.tf = Param.T/Param.N;
intg_options.simplify = true;
intg_options.number_of_finite_elements = 4;
% DAE (Differential-algebraic equations) problem
dae = struct;
dae.x = xx;          % States
dae.p = uu;          % Parameters are the things thar are fixed during the integration horizon --> In this case, controls
dae.ode = f_cont(xx,uu);   % System right hand side f(x,u) = ode
intg = integrator('intg','rk', dae, intg_options); % Label, especific integration method, problem definition, options
result = intg('x0',xx,'p',uu); % Evaluate with symbols
x_next = result.xf;
f_disc = Function('f_disc',{xx,uu},{x_next},{'x','u'},{'x_next'}); % {x,u} -> {x_next}

% prepare reference for MPC problem:
Param.nIO = Param.dTo/Param.dTi;

% Optimal control problem using multiple-shooting
MPC = casadi.Opti(); % to use ipopt
% MPC = casadi.Opti('conic'); % to use qpoasis
Param.xx = MPC.variable(n_states,Param.N+1); % Decision variables for state trajectory
Param.uu = MPC.variable(n_controls,Param.N);
Param.xx0 = MPC.parameter(n_states,1); % Because MPC, the first state must be equal to a measurement --> Parameter (not optimized over)
Param.xx_ref = MPC.parameter(n_states,Param.N+1); % reference state trajectory
% x_ini = MPC.parameter(n_states,Param.N+1); % initial guess for state sequence
% u_ini = MPC.parameter(n_controls,Param.N); % initial guess for input sequence
obj = 0;
tk = 0;
for k = 1:Param.N
    obj = obj + (Param.xx(:,k)-Param.xx_ref(:,k))'*Param.Q*(Param.xx(:,k)-Param.xx_ref(:,k)) + Param.uu(:,k)'*Param.R*Param.uu(:,k);
end
obj = obj + (Param.xx(:,Param.N+1)-Param.xx_ref(:,Param.N+1))'*Param.P*(Param.xx(:,Param.N+1)-Param.xx_ref(:,Param.N+1));
MPC.minimize(obj);
% Dynamic constraints of the multiple shooting
for k=1:Param.N
    MPC.subject_to(Param.xx(:,k+1) == f_disc(Param.xx(:,k),Param.uu(:,k))); % One step integration function predicts
end
% opti.subject_to(u_min <= u <= u_max)
MPC.subject_to(Param.xx(:,1) == Param.xx0)

% Solution Value
% use ipopt (mean dt_MPC = 13 ms, max dt_MPC = ~40 ms)
s_opts = struct;
s_opts.ipopt.print_level =0;
s_opts.print_time = 0;
MPC.solver('ipopt',s_opts);
% % use qpoases as problem is QP with linear equality constraints 
% % (mean dt_MPC = 40 ms, max dt_MPC = ~150 ms)
% s_opts = struct;
% s_opts.printLevel = 'none';
% s_opts.print_time = 0;
% MPC.solver('qpoases',s_opts);

Param.MPC = MPC;

% % Test if problem is well defined
% try
%     x0_test = [p0{iD}(:,1);v0{iD}(:,1);lbd{iD}(3,1)];
%     x_ref_test = [p_ref{1}(:,1:Param.nIO:Param.nIO*(Param.N+1));v_ref{1}(:,1:Param.nIO:Param.nIO*(Param.N+1));psi_ref{1}(:,1:Param.nIO:Param.nIO*(Param.N+1))];
%     x_ini_test = zeros(n_states,Param.N+1);
%     u_ini_test = zeros(n_controls,Param.N);
% 
%     MPC.set_value(Param.xx0,x0_test);
%     MPC.set_value(Param.xx_ref,x_ref_test);
%     MPC.set_initial(Param.xx,x_ini_test);
%     MPC.set_initial(Param.uu,u_ini_test);
%     sol = MPC.solve();
%     x_seq = sol.value(Param.xx);
%     u_seq = sol.value(Param.uu);
%     pos = x_seq(1:3,:);
%     pos_ref = x_ref_test(1:3,:);
%     vel = x_seq(4:6,:);
%     vel_ref = x_ref_test(4:6,:);
%     yaw = x_seq(7,:);
%     yaw_ref = x_ref_test(7,:);
%     t_test = 0:Param.dTo:Param.dTo*Param.N;
% 
%     figure(1);
%     plot3(pos(1,:),pos(2,:),pos(3,:),'b');
%     hold on;
%     plot3(pos_ref(1,:),pos_ref(2,:),pos_ref(3,:),'g--');
%     hold off;
%     xlabel('$$x$$ [m]');
%     ylabel('$$y$$ [m]');
%     ylabel('$$z$$ [m]');
%     grid on;
%     axis equal;
%     legend('$$p$$','$$p_{ref}$$');
% 
%     figure(2);
%     plot(t_test,pos);
%     hold on;
%     plot(t_test,pos_ref,'--');
%     hold off;
%     xlabel('$$t$$ [s]');
%     ylabel('$$p$$ [m]');
%     grid on;
%     legend('$$x$$','$$y$$','$$z$$','$$x_{ref}$$','$$y_{ref}$$','$$z_{ref}$$');
% 
%     figure(3);
%     plot(t_test,yaw*180/pi);
%     hold on;
%     plot(t_test,yaw_ref*180/pi,'--');
%     hold off;
%     xlabel('t [s]');
%     ylabel('$$\psi$$ [deg]');
%     grid on;
%     legend('$$\psi$$','$$\psi_{ref}$$');
% 
%     figure(4);
%     plot(t_test,vel);
%     hold on;
%     plot(t_test,vel_ref,'--');
%     hold off;
%     xlabel('$$t$$ [s]');
%     ylabel('$$v$$ [m]');
%     grid on;
%     legend('$$v_x$$','$$v_y$$','$$v_z$$','$$v_{x ref}$$','$$v_{y ref}$$','$$v_{z ref}$$');
% 
%     figure(5);
%     plot(t_test(1:end-1),u_seq);
%     xlabel('$$t$$ [s]');
%     ylabel('$$u$$');
%     grid on;
%     legend('$$a_x$$','$$a_y$$','$$a_z$$','$$\omega_z$$');
% 
% catch ME
%     error(ME.message);
% end

for iD = 1:Param.nD
    f_d_MPC{iD} = Param.m*Param.g*[0;0;1];
    x_MPC{iD} = zeros(n_states,Param.N+1);
    u_MPC{iD} = zeros(n_controls,Param.N);
end



