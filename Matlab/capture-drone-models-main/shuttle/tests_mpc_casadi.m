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
Param.nD = 1; % number of drones

% reference parameters
Param.p_ref_static = [1;1;3];
Param.psi_ref_static = pi/3;

Param.vz_d = 0.1;
Param.dh = 0.05;      % safety height difference between drones
Param.Rad = 5;        % radius of circle
Param.omn = 2*pi/20;  % rotation frequency
Param.dphase = -pi/12;% ref circle angular difference between drones

Param.ref_mode = 1; % reference: 1 - square wave; 2 - circle

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
    p0{iD} = [-1;-2+0.2*((Param.nD-1)/2-iD+1);-3];
    v0{iD} = [0;0;0];
    psi0{iD} = pi/20 + pi/20*((Param.nD-1)/2-iD+1);
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
        p_ref{iD} = p0{iD}*ones(1,Param.Tend/Param.dTi+1) + Param.p_ref_static*sw;
        v_ref{iD} = zeros(3,nt);
        a_ref{iD} = zeros(3,nt);
        j_ref{iD} = zeros(3,nt);
        psi_ref{iD} = psi0{iD}*ones(1,Param.Tend/Param.dTi+1) + Param.psi_ref_static*sw;
        dpsi_ref{iD} = zeros(1,Param.Tend/Param.dTi+1);
        
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

% M690B drone MPC controller definition
% state
p = MX.sym('p',3);
psi = MX.sym('psi');
x = [p;psi];
n_states = size(x,1);
% control
v = MX.sym('v',3);
omegaz = MX.sym('omegaz');
u = [v;omegaz];
n_controls = size(u,1);
u_min = [-20; -20;  -20; -2*pi]; u_max = [20; 20; 20; 2*pi];
% Simplified Nonlinear ODE (Ordinary Differential Equation)
xdot = [v; omegaz];
f_cont = Function('f_cont',{x,u},{xdot},{'x','u'},{'xdot'}); % last two represent labels for better understanding
% controller parameters
T = 2; % Time horizon
N = 20; % Number of control intervals
Q = blkdiag(1,1,1,0.1)*100;
R = blkdiag(0.1,0.1,0.1,0.5);
P = 10*Q;
% Integrator to discretize the system
intg_options = struct;
intg_options.tf = T/N;
intg_options.simplify = true;
intg_options.number_of_finite_elements = 4;
% DAE (Differential-algebraic equations) problem
dae = struct;
dae.x = x;          % States
dae.p = u;          % Parameters are the things thar are fixed during the integration horizon --> In this case, controls
dae.ode = f_cont(x,u);   % System right hand side f(x,u) = ode
intg = integrator('intg','rk', dae, intg_options); % Label, especific integration method, problem definition, options
result = intg('x0',x,'p',u); % Evaluate with symbols
x_next = result.xf;
f_disc = Function('f_disc',{x,u},{x_next},{'x','u'},{'x_next'}); % {x,u} -> {x_next}

% prepare reference for MPC problem:
nIO = Param.dTo/Param.dTi;

% Optimal control problem using multiple-shooting
opti = casadi.Opti();
x = opti.variable(n_states,N+1); % Decision variables for state trajectory
u = opti.variable(n_controls,N);
x0 = opti.parameter(n_states,1); % Because MPC, the first state must be equal to a measurement --> Parameter (not optimized over)
% x_ref = [0;0;0;0]; % reference state trajectory
x_ref = opti.parameter(n_states,N+1); % reference state trajectory
obj = 0;
tk = 0;
for k = 1:N
%     obj = obj + (x(:,k)-x_ref)'*Q*(x(:,k)-x_ref) + u(:,k)'*R*u(:,k);
    x_error = x(:,k)-x_ref(:,k);
    obj = obj + x_error'*Q*x_error + u(:,k)'*R*u(:,k);
end
% obj = obj + (x(:,N+1)-x_ref)'*P*(x(:,N+1)-x_ref);
x_error = x(:,N+1)-x_ref(:,N+1);
obj = obj + x_error'*P*x_error;
opti.minimize(obj);
% Dynamic constraints of the multiple shooting
for k=1:N
    opti.subject_to(x(:,k+1) == f_disc(x(:,k),u(:,k))); % One step integration function predicts
end
% opti.subject_to(u_min <= u <= u_max)
opti.subject_to(x(:,1) == x0)
% Solution Value
s_opts = struct;
s_opts.ipopt.max_iter = 2000;
s_opts.ipopt.print_level =0;
s_opts.print_time = 0;
s_opts.ipopt.acceptable_tol =1e-8;
s_opts.ipopt.acceptable_obj_change_tol = 1e-6;
opti.solver('ipopt',s_opts);
% MPC = opti.to_function('MPC',{x0},{u,x},{'x0'},{'u_opt','x_opt'});
MPC = opti.to_function('MPC',{x0,x_ref},{u,x},{'x0','x_ref'},{'u_opt','x_opt'});
% Test if problem is well defined
try
    x0_test = [p0{iD}(1:3,1);psi0{iD}(1,1)];
%     [u_opt,x_opt] = MPC(x0_test);
    x_ref_test = 2*[p_ref{1}(:,1:nIO:nIO*(N+1));psi_ref{1}(:,1:nIO:nIO*(N+1))];
    [u_opt,x_opt] = MPC(x0_test,x_ref_test);
%     x_test = full(x_opt(:,:));
%     x_ref = x_ref*ones(1,N+1);
    t_test = 0:Param.dTo:Param.dTo*N;

    figure(2);
    plot(t_test,full(x_opt(:,:)));
    hold on;
    plot(t_test,x_ref_test,'--');
    hold off;
    xlabel('t [s]');
    ylabel('x');
    grid on;
    legend('Visible','on');
%     legend('x','x_{ref}');

    figure(4);
    plot(t_test(1:end-1),full(u_opt));
    xlabel('t [s]');
    ylabel('u');
    grid on;
    legend('Visible','on');
%     legend('v_x','\omega_z');

catch ME
    error(ME.message);
end
