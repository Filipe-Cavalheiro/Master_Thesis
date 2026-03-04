clc; clear; close all;

%sim parameters
dt = 0.02;
T = 20;

% Parameters
goal = [10;10;10];
obs  = [4.99;5;5];
r_obs = 2;
safety_margin = 0.5;
r_safe = r_obs + safety_margin;
    
% Control Barrier Function        
k0 = 3;
k1 = 3;

% Initial state [px; py; pz; vx; vy; vz]
x = [0;0;0;0;0;0];

traj = x;

for t = 0:dt:T
    
    p = x(1:3);
    v = x(4:6);
    
    % Nominal controller (PD to goal)
    kp = 3;
    kd = 2;
    
    u_nom = kp*(goal - p) - kd*v;
    
    h      = r_safe^2 - (p - obs)'*(p - obs);
    h_dot  = -2*(p - obs)'*v;
    
    Lfh = -2*(p - obs)'*v;

    L2fh  = -2*(v'*v);
    LgLfh = -2*(p - obs)';
    
    % For Au <= b form
    A = LgLfh;
    b = -L2fh - k1*h_dot - k0*h;

    % Check constraint
    if A*u_nom <= b
        u = u_nom;
    else
        % Project onto the half-space(constraint boundary)
        u = u_nom - ((A*u_nom - b)/(A*A')) * A';
    end
    
    % Apply dynamics
    
    p_dot = v;
    v_dot = u;
    
    x = x + dt*[p_dot; v_dot];
    
    traj = [traj x];
    
    % Stop if close to goal
    if norm(p - goal) < 0.1
        break;
    end
end

% 3D Plot
figure; hold on; grid on; axis equal;

% Trajectory
plot3(traj(1,:), traj(2,:), traj(3,:), ...
      'b','LineWidth',2);

% Goal
plot3(goal(1), goal(2), goal(3), ...
      'gx','MarkerSize',12,'LineWidth',2);

% Obstacle sphere
[Xs,Ys,Zs] = sphere(40);

surf(obs(1) + r_obs*Xs, ...
     obs(2) + r_obs*Ys, ...
     obs(3) + r_obs*Zs, ...
     'FaceColor','r', ...
     'FaceAlpha',0.4, ...
     'EdgeColor','none');

% Safety boundary
surf(obs(1) + r_safe*Xs, ...
     obs(2) + r_safe*Ys, ...
     obs(3) + r_safe*Zs, ...
     'FaceColor','r', ...
     'FaceAlpha',0.1, ...
     'EdgeColor','none');

xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Cable-Driven Vehicle with CBF Obstacle Avoidance');
legend('Trajectory','Goal','Obstacle','Safety boundary');

view(3);
