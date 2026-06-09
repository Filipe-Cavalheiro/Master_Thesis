clc; clear; close all;

%% Simulation parameters
dt = 0.02;
T  = 20;

%% Goal
goal = [10;10;10];

%% Obstacles
obs = [ 4.99  5.00  5.00;
        6.00  3.00  4.00;
        3.00  8.00  4.00 ]';

r_obs = [2; 1.5; 1.8];

safety_margin = 0.5;
r_safe = r_obs + safety_margin;

nObs = size(obs,2);

%% Control Barrier Function gains
k0 = 3;
k1 = 3;

%% Initial state
x = [0;0;0;0;0;0];

traj = x;

%% Simulation loop
for t = 0:dt:T

    p = x(1:3);
    v = x(4:6);

    %% Nominal controller
    kp = 3;
    kd = 2;

    u_nom = kp*(goal - p) - kd*v;

    %% Build constraints for all obstacles
    A_all = [];
    b_all = [];

    for i = 1:nObs

        obs_i = obs(:,i);

        h     = r_safe(i)^2 - (p-obs_i)'*(p-obs_i);
        h_dot = -2*(p-obs_i)'*v;

        L2fh  = -2*(v'*v);
        LgLfh = -2*(p-obs_i)';

        A_i = LgLfh;
        b_i = -L2fh - k1*h_dot - k0*h;

        A_all = [A_all; A_i];
        b_all = [b_all; b_i];
    end

    %% Check if nominal control satisfies all constraints
    if all(A_all*u_nom <= b_all)

        u = u_nom;

    else

        % Sequential projection onto violated constraints
        u = u_nom;

        for i = 1:nObs

            A = A_all(i,:);
            b = b_all(i);

            if A*u > b
                u = u - ((A*u - b)/(A*A'))*A';
            end
        end
    end

    %% Dynamics
    p_dot = v;
    v_dot = u;

    x = x + dt*[p_dot; v_dot];

    traj = [traj x];

    %% Stop near goal
    if norm(p-goal) < 0.1
        break;
    end
end

%% Plot
figure;
hold on;
grid on;
axis equal;

%% Trajectory
plot3(traj(1,:),traj(2,:),traj(3,:), ...
      'b','LineWidth',2);

%% Goal
plot3(goal(1),goal(2),goal(3), ...
      'gx','MarkerSize',12,'LineWidth',2);

%% Draw obstacles
[Xs,Ys,Zs] = sphere(40);

colors = {'r','g','m'};

for i = 1:nObs

    surf(obs(1,i)+r_obs(i)*Xs,...
         obs(2,i)+r_obs(i)*Ys,...
         obs(3,i)+r_obs(i)*Zs,...
         'FaceColor',colors{i},...
         'FaceAlpha',0.4,...
         'EdgeColor','none');

    surf(obs(1,i)+r_safe(i)*Xs,...
         obs(2,i)+r_safe(i)*Ys,...
         obs(3,i)+r_safe(i)*Zs,...
         'FaceColor',colors{i},...
         'FaceAlpha',0.1,...
         'EdgeColor','none');
end

xlabel('X');
ylabel('Y');
zlabel('Z');

title('3D Vehicle with Multiple CBF Obstacles');

legend('Trajectory','Goal');

view(3);