clc; clear; close all;

% Simulation parameters
Tend = 8;
dt   = 0.001;
Nsim = round(Tend/dt) + 1;
t    = linspace(0, Tend, Nsim);

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

    if k == Tend/2/dt % new desired goal for t >= 5 sec
        % New desired equilibrium points
        q_d = [1; 0; 0];
        q_d = q_d / norm(q_d);
        omega_d = zeros(3,1); 
    end

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
        - tau;

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

% --- Attitude trajectory on S^2
figure; hold on; grid on; axis equal;
[xs,ys,zs] = sphere(50);
surf(xs,ys,zs, ...
    'FaceAlpha',0.25,'EdgeColor','none','FaceColor',[0.85 0.85 0.85]);

plot3(q_hist(1,:), q_hist(2,:), q_hist(3,:), 'r','LineWidth',2);
plot3(q_hist(1,1), q_hist(2,1), q_hist(3,1), 'go','MarkerFaceColor','g');
plot3(q_hist(1,end), q_hist(2,end), q_hist(3,end), 'ro','MarkerFaceColor','r');

xlabel('x'); ylabel('y'); zlabel('z');
title('Pointing Direction Trajectory on S^2');
legend('Unit Sphere','Trajectory','Start','End');
view(3);

% --- Error function
figure; grid on;
plot(t, Psi_hist, 'k','LineWidth',1.5);
xlabel('Time [s]');
ylabel('\Psi(q,q_d)');
title('Attitude Error Function');

% --- Configuration error components
figure;
subplot(3,1,1); plot(t, eq_hist(1,:)); grid on; ylabel('e_{q1}');
subplot(3,1,2); plot(t, eq_hist(2,:)); grid on; ylabel('e_{q2}');
subplot(3,1,3); plot(t, eq_hist(3,:)); grid on; ylabel('e_{q3}');
xlabel('Time [s]');
sgtitle('Configuration Error Vector');

% --- Angular velocity
figure;
subplot(3,1,1); plot(t, omega_hist(1,:)); grid on; ylabel('\omega_1');
subplot(3,1,2); plot(t, omega_hist(2,:)); grid on; ylabel('\omega_2');
subplot(3,1,3); plot(t, omega_hist(3,:)); grid on; ylabel('\omega_3');
xlabel('Time [s]');
sgtitle('Angular Velocity');

% === Time-Accurate Animation of Pointing Direction on S^2 ===

animation = figure; hold on; grid on; axis equal;
[xs,ys,zs] = sphere(50);
surf(xs,ys,zs, ...
    'FaceAlpha',0.25,'EdgeColor','none','FaceColor',[0.85 0.85 0.85]);

xlabel('x'); ylabel('y'); zlabel('z');
title('Pointing Direction Animation on S^2');
view(45,30);

% --- Desired equilibrium points
q_d1 = [0; -0.0175; -0.9998]; q_d1 = q_d1/norm(q_d1);
q_d2 = [1; 0; 0];

plot3(q_d1(1), q_d1(2), q_d1(3), 'kp', ...
      'MarkerSize',12,'MarkerFaceColor','k');
text(q_d1(1), q_d1(2), q_d1(3), ' q_{d1}', 'FontSize',12);

plot3(q_d2(1), q_d2(2), q_d2(3), 'kp', ...
      'MarkerSize',12,'MarkerFaceColor','k');
text(q_d2(1), q_d2(2), q_d2(3), ' q_{d2}', 'FontSize',12);

% --- Trajectory & current point
traj = plot3(q_hist(1,1), q_hist(2,1), q_hist(3,1), ...
             'r','LineWidth',2);
current = plot3(q_hist(1,1), q_hist(2,1), q_hist(3,1), ...
                'ro','MarkerFaceColor','r','MarkerSize',6);

% --- Time annotation
timeText = text(-1.7,0,1.2,'t = 0.00 s', ...
    'FontSize',12,'FontWeight','bold');

% === Video settings (TIME-ACCURATE) ===
save_to_mp4 = 1;
frameStep   = 10;                % must match animation loop
sim_dt      = dt * frameStep;    % simulated time per frame
videoFPS    = 1 / sim_dt;        % makes video exactly 10 sec

if save_to_mp4
    mp4File = 'lyapunov_stability_in_s2.mp4';
    v = VideoWriter(mp4File,'MPEG-4');
    v.FrameRate = videoFPS;
    open(v);
end

% === Animation loop ===
for k = 1:frameStep:Nsim

    % Update trajectory
    set(traj, 'XData', q_hist(1,1:k), ...
              'YData', q_hist(2,1:k), ...
              'ZData', q_hist(3,1:k));

    % Update current point
    set(current, 'XData', q_hist(1,k), ...
                 'YData', q_hist(2,k), ...
                 'ZData', q_hist(3,k));

    % Update time text
    set(timeText,'String',sprintf('t = %.2f s',t(k)));

    drawnow limitrate nocallbacks

    if save_to_mp4
        frame = getframe(animation);
        writeVideo(v,frame);
    end
end

if save_to_mp4
    close(v);
end
