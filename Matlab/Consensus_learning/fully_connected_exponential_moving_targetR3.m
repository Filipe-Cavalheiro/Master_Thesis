clc; clear; close all;

Tend = 10;
dt   = 0.01;
Nsim = round(Tend/dt) + 1;
t    = linspace(0, Tend, Nsim);

n_drones = 3;

% consensus Laplacian
L = n_drones*eye(n_drones) - ones(n_drones);

% parameters
alpha = 0.2;          % exponential moving average weight
noise_std = 0.2;      % measurement noise

% initial estimates
target_position = randn(3,n_drones);

% EMA memory
ema_estimate = target_position;

% storage
target = zeros(3,n_drones,Nsim);
target_true = zeros(3,Nsim);

% circular motion parameters
R = 5;
omega = 0.5;

for k = 1:Nsim
    
    % ---- true moving target (circle in XY plane) ----
    true_pos = [R*cos(omega*t(k));
                R*sin(omega*t(k));
                1];
            
    target_true(:,k) = true_pos;

    % ---- noisy measurements ----
    measurements = true_pos + noise_std*randn(3,n_drones);

    % ---- exponential moving average filter ----
    ema_estimate = (1-alpha)*ema_estimate + alpha*measurements;

    % ---- consensus update ----
    target_position = ema_estimate - dt*ema_estimate*L;

    % store
    target(:,:,k) = target_position;
end

figure;
hold on;
grid on;
axis equal; 
view(3);

% plot true trajectory
plot3(target_true(1,:),target_true(2,:),target_true(3,:),'k','LineWidth',3)

for i = 1:n_drones
    x = squeeze(target(1,i,:));
    y = squeeze(target(2,i,:));
    z = squeeze(target(3,i,:));

    plot3(x,y,z,'LineWidth',2)
end

for i = 1:n_drones
    x = squeeze(target(1,i,:));
    y = squeeze(target(2,i,:));
    z = squeeze(target(3,i,:));
 
    scatter3(x(1),y(1),z(1),80,'filled')
    scatter3(x(end),y(end),z(end),80,'d','filled')
end

xlabel('X')
ylabel('Y')
zlabel('Z')

title('Consensus Tracking of Moving Target (EMA + Noisy Measurements)')
legend('True target','Drone 1 est','Drone 2 est','Drone 3 est')