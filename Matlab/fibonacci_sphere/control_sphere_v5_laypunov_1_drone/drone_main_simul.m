% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iD = 1;

% System parameters (V. Numerical Simulations)
c = 0.3;                             % Nm (s/rad)
J = diag([0.0294 0.0305 0.0495]);    % kg m^2
tau_outerloop = zeros(3,1);

% Controller gains (29a, 29b)
K_r = diag([4.234 4.392 7.128]);
K_w = diag([7.056 7.320 11.88]) * 1e-1;

% Hat (skew) operator
S = @(x) [   0   -x(3)  x(2);
           x(3)    0   -x(1);
          -x(2)  x(1)    0 ];

% Initial conditions
% The initial conditions of the outer loop is the projection on the sphere
% of the initial conditions of the inner loop agent.
q = projectToSphere(initial_pos(:,1), Param.sphere_radius);

omega = [0; 0.3; 0];            % initial angular velocity

Q = eye(3);

% Desired equilibrium
q_d = [0; 1; 0];
q_d = q_d / norm(q_d);

omega_d = zeros(3,1);

% Storage
q_hist     = zeros(3, Nsim);
omega_hist = zeros(3, Nsim);
u_hist     = zeros(3, Nsim);
eq_hist    = zeros(3, Nsim);
Psi_hist   = zeros(1, Nsim);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("intereation:%d\n", k);

    % get state vector
    pos{iD} = x{iD}(1:3,k);
    v{iD} = x{iD}(4:6,k);
    R = reshape(x{iD}(7:15,k),3,3);
    om = x{iD}(16:18,k);

    % --- Attitude error function (Eq. 8)
    Psi = 2 - sqrt(2) * sqrt(1 + q' * q_d);

    % --- Configuration error vector (Eq. 12)
    e_q = (1 / (sqrt(2) * sqrt(1 + q' * q_d))) * Q' * S(q_d) * q;

    % --- Angular velocity error
    e_w = omega - omega_d;

    % --- Control law (Eq. 28, reduced to S^2)
    u = Q'* (-K_r * Q * e_q ...
        -K_w * e_w )...
        + S(omega) * J * omega ...
        + c * omega ...
        - tau_outerloop;

    % --- Rigid body dynamics (Eq. 5)
    omega_dot = J \ (u - c*omega - S(omega)*J*omega + tau_outerloop);
    omega = omega + Param.dTi * omega_dot;

    % --- Reduced attitude kinematics (Eq. 4)

    q_dot = S(omega) * q;
    q = q + Param.dTi * q_dot;

    % --- Enforce S^2 constraint
    q = (q / norm(q))*Param.sphere_radius;

    % --- Store
    q_hist(:,k)     = q;
    omega_hist(:,k) = omega;
    u_hist(:,k)     = u;
    eq_hist(:,k)    = e_q;
    Psi_hist(k)     = Psi;
    
    % get reference
    %p_d = p_ref{iD}(:,k);
    p_d =  q;
    %v_d = v_ref{iD}(:,k);
    v_d = zeros(3,1);
    %a_d = a_ref{iD}(:,k);
    a_d = zeros(3,1);
    %j_d = j_ref{iD}(:,k);
    j_d = zeros(3,1);
    %psi_d = psi_ref{iD}(:,k);
    psi_d = 0;
    %dpsi_d = dpsi_ref{iD}(:,k);
    dpsi_d = 0;

    % mellinger controller
    [T{iD}(:,k),tau{iD}(:,k),e_p] = drone_mellinger_ctrl(pos{iD},v{iD},R,om,Param,p_d,psi_d,xiep{iD}(:,k),v_d,dpsi_d,a_d,j_d);
    
    % integrate position error
    xiep{iD}(:,k+1) = xiep{iD}(:,k) + Param.dTi*e_p;
    
    % nonlinear drone model (continuous time)
    [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(v{iD},R,om,T{iD}(:,k),tau{iD}(:,k),Param);
    
    % discretization 
    pp = pos{iD} + Param.dTi*dot_p;
    vp = v{iD} + Param.dTi*dot_v;
    Rp = rot_integrate(R,om,Param.dTi);
    omp = om + Param.dTi*dot_om;
    x{iD}(:,k+1) = [pp;vp;reshape(Rp,[],1);omp];

    % auxiliary drone attitude computation from rotation matrix
    lbd{iD}(:,k) = R2Euler(R);

    if abs(1-norm(Rp'*Rp))>1e-4
        warning(['Problems with rotation matrix integration... stoping simulation: t = ' num2str(k*Param.dTi) ' s.']);
        break;
    end
end

% trim vectors to same length
clear pos;
iD = 1;
pos{iD} = x{iD}(1:3,:);

figure; hold on; grid on; axis equal
[x, y, z] = sphere;
x = x * Param.sphere_radius;
y = y * Param.sphere_radius;
z = z * Param.sphere_radius;

% Draw reference sphere
hSphere = surf(x, y, z);
hSphere.FaceAlpha = 0.3; 
hSphere.EdgeColor = 'none';
hSphere.FaceColor = [0.8627,0.8627,0.8627];

% Setup colors and plot handles for agents
colors = lines(10);
% Initialize line and marker objects
outer_loop_traj = plot3(NaN, NaN, NaN, 'Color', colors(1,:), 'LineWidth', 1);
inner_loop_traj = plot3(NaN, NaN, NaN, 'Color', colors(1,:), 'LineWidth', 1);
hAgent = plot3(NaN, NaN, NaN, 'x', 'Color', colors(1,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
href = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));

xlabel('x'); ylabel('y'); zlabel('z');
title('3D Trajectories of Agents – Evolution Over Time');
view(3);

plot3(q_hist(1,1), q_hist(2,1), q_hist(3,1), 'go','MarkerFaceColor','g');
plot3(q_hist(1,end), q_hist(2,end), q_hist(3,end), 'ro','MarkerFaceColor','r');

% Animation loop
nSteps = size(pos{1}, 2);  % number of time steps
for k = 1:nSteps-1

    % Update trajectory up to current time
    set(outer_loop_traj, 'XData', q_hist(1, 1:k), ...
                   'YData', q_hist(2, 1:k), ...
                   'ZData', q_hist(3, 1:k));

    % Update trajectory up to current time
    set(inner_loop_traj, 'XData', pos{iD}(1, 1:k), ...
                   'YData', pos{iD}(2, 1:k), ...
                   'ZData', pos{iD}(3, 1:k));
    
    % Update agent position marker
    set(hAgent, 'XData', pos{iD}(1, k), ...
                    'YData', pos{iD}(2, k), ...
                    'ZData', pos{iD}(3, k));
    
    % Update ref position marker
    set(href, 'XData', q_hist(1, k), ...
                    'YData', q_hist(2, k), ...
                    'ZData', q_hist(3, k));

    drawnow;
    pause(0.01); % adjust speed of animation
end
