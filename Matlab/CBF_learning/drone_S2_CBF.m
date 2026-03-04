clc; clear; close all;

% === Simulation parameters ===
Tend = 10;
dt   = 0.05;
Nsim = round(Tend/dt) + 1;
t    = linspace(0, Tend, Nsim);

% === Parameters CBF ===
% Parameters for the circle 
x = 90; % Longitude in degrees
y = 180;  % Latitude in degrees
r_safe = 0.2;

% Convert degrees to Radians for computation 
x_rad = deg2rad(x); 
y_rad = deg2rad(y);

% Calculate the circle's center in Cartesian coordinates 
center_x = cos(y_rad) * cos(x_rad); 
center_y = cos(y_rad) * sin(x_rad); 
center_z = sin(y_rad); 

obstacle  = [center_x;center_y;center_z];
k0 = 10;        % CBF gain

% System parameters (V. Numerical Simulations)
c = 0.3;                             % Nm (s/rad)
J = diag([0.0294 0.0305 0.0495]);    % kg m^2
tau = zeros(3,1);

% Controller gains (29a, 29b)
K_r = diag([4.234 4.392 7.128]);
K_w = diag([7.056 7.320 11.88]) * 1e-1;

% Hat (skew) operator
S = @(x) [   0   -x(3)  x(2);
           x(3)    0   -x(1);
          -x(2)  x(1)    0 ];

% Initial conditions
q = [0; 0; 1];                  % initial pointing direction
q = q / norm(q);                % enforce S^2

omega = [0; 0.3; 0];            % initial angular velocity

Q = eye(3);

% Desired equilibrium
q_d = [0; -0.0175; -0.9998];
q_d = q_d / norm(q_d);

omega_d = zeros(3,1);

% Storage
q_hist     = zeros(3, Nsim);
omega_hist = zeros(3, Nsim);
u_hist     = zeros(3, Nsim);
eq_hist    = zeros(3, Nsim);
Psi_hist   = zeros(1, Nsim);

% Main simulation loop
for k = 1:Nsim
    
    % if k == round(Tend/2/dt) % new desired goal for t >= 5 sec
    %     % New desired equilibrium points
    %     q_d = [1; 0; 0];
    %     q_d = q_d / norm(q_d);
    %     omega_d = zeros(3,1); 
    % end

    % === Nominal Lyapunov control ========================================
    % Attitude error function (Eq. 8)
    Psi = 2 - sqrt(2) * sqrt(1 + q' * q_d);

    % Configuration error vector (Eq. 12)
    e_q = (1 / (sqrt(2) * sqrt(1 + q' * q_d))) * Q' * S(q_d) * q;

    % Angular velocity error
    e_w = omega - omega_d;

    % Control law (Eq. 28, reduced to S^2)
    u_nom = Q'* (-K_r * Q * e_q ...
        -K_w * e_w )...
        + S(omega) * J * omega ...
        + c * omega ...
        - tau;
    
    % === Control barrier function=========================================
    V = 2;
    % Configuration error vector of safety barrier function
    if V == 1
        theta = acos( dot(q,obstacle) / (norm(q)*norm(obstacle)));
        h = sin(theta/2) - (pi/8);    
        grad_h = cos(theta/2)/2;

        % For Au <= b form
        A = -grad_h';
        b = k0*h;
    elseif V == 2
        safety_q = (1 / (sqrt(2) * sqrt(1 + q' * obstacle))) * S(obstacle) * q;
        h = safety_q' * safety_q - r_safe^2;
        Jacob = [[(2^(1/2)*obstacle(1)*obstacle(3)*q(2))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(1)*obstacle(2)*q(3))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)), (2^(1/2)*obstacle(2)*obstacle(3)*q(2))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(2)^2*q(3))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(3))/(2*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(1/2)), (2^(1/2)*obstacle(2))/(2*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(1/2)) + (2^(1/2)*obstacle(3)^2*q(2))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(2)*obstacle(3)*q(3))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2))]
    [(2^(1/2)*obstacle(3))/(2*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(1/2)) + (2^(1/2)*obstacle(1)^2*q(3))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(1)*obstacle(3)*q(1))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)),                                                     (2^(1/2)*obstacle(1)*obstacle(2)*q(3))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(2)*obstacle(3)*q(1))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)), (2^(1/2)*obstacle(1)*obstacle(3)*q(3))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(3)^2*q(1))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(1))/(2*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(1/2))]
    [(2^(1/2)*obstacle(1)*obstacle(2)*q(1))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(1)^2*q(2))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(2))/(2*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(1/2)), (2^(1/2)*obstacle(1))/(2*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(1/2)) + (2^(1/2)*obstacle(2)^2*q(1))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(1)*obstacle(2)*q(2))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)),                                                     (2^(1/2)*obstacle(2)*obstacle(3)*q(1))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2)) - (2^(1/2)*obstacle(1)*obstacle(3)*q(2))/(4*(obstacle(1)*q(1) + obstacle(2)*q(2) + obstacle(3)*q(3) + 1)^(3/2))]];
        grad_h = 2*safety_q;

        % For Au <= b form
        A = -grad_h';
        b = k0*h;
    else
        theta_safe = pi/8;
        h = q' * obstacle - cos(theta_safe);

        % For Au <= b form
        A = -obstacle';
        b = k0 * h;
    end
    
    % Check constraint
    if A*u_nom <= b
        u = u_nom;
    else
        % Project onto the half-space(constraint boundary)
        u = u_nom - ((A*u_nom - b)/(A*A')) * A';
    end

    % --- Rigid body dynamics (Eq. 5)
    omega_dot = J \ (u - c*omega - S(omega)*J*omega + tau);
    omega = omega + dt * omega_dot;

    % --- Reduced attitude kinematics (Eq. 4)
    q_dot = S(omega) * q;
    q = q + dt * q_dot;

    % --- Enforce S^2 constraint
    q = q / norm(q);

    % --- Store
    q_hist(:,k)     = q;
    omega_hist(:,k) = omega;
    u_hist(:,k)     = u;
    eq_hist(:,k)    = e_q;
    Psi_hist(k)     = Psi;
end

% === Plots ===
% Attitude trajectory on S^2
figure; hold on; grid on; axis equal;

xlabel('x'); ylabel('y'); zlabel('z');
title('Pointing Direction Trajectory on S^2');
%legend('Unit Sphere','Trajectory','Start','End');
view(3);

[xs,ys,zs] = sphere(50);
surf(xs,ys,zs, ...
    'FaceAlpha',0.25,'EdgeColor','none','FaceColor',[0.85 0.85 0.85]);

plot3(q_hist(1,:), q_hist(2,:), q_hist(3,:), 'r','LineWidth',2);
plot3(q_hist(1,1), q_hist(2,1), q_hist(3,1), 'go','MarkerFaceColor','g');
plot3(q_hist(1,end), q_hist(2,end), q_hist(3,end), 'ro','MarkerFaceColor','r');

% Define a circle in the tangent plane at obstacle
theta = linspace(0, 2*pi, 100);

% Find two orthogonal vectors tangent to the sphere at obstacle
v1 = null(obstacle');   % 3x2 orthonormal basis for tangent plane
v1 = v1(:,1:2);   % take both vectors

% Circle in tangent plane
circle_local = r_safe * (v1(:,1)*cos(theta) + v1(:,2)*sin(theta));  % 3x100

% Project circle points onto the sphere
circle_points = zeros(3,length(theta));
for i = 1:length(theta)
    circle_points(:,i) = (obstacle + circle_local(:,i));
    circle_points(:,i) = circle_points(:,i) / norm(circle_points(:,i)); % normalize to S^2
end

% Plot the circle
plot3(circle_points(1,:), circle_points(2,:), circle_points(3,:), 'r-', 'LineWidth',2);