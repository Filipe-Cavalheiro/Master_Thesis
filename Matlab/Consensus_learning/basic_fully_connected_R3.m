clc; clear; close all;

Tend = 10;
dt   = 0.005;
Nsim = round(Tend/dt) + 1;
t    = linspace(0, Tend, Nsim);

n_drones = 3;
target_position = randn(18,n_drones); % R3 positions
L = n_drones*eye(n_drones) - ones(n_drones);

disp("Correct consensus value:")
disp(mean(target_position))

% store trajectories
target = zeros(18,n_drones,Nsim);

for k = 1:Nsim
    target(:,:,k) = target_position;
    target_position = target_position - dt*target_position*L;
end

disp("Final consensus positions:")
disp(target_position)

figure
hold on
grid on
view(3)

for i = 1:n_drones
    x = squeeze(target(1,i,:));
    y = squeeze(target(2,i,:));
    z = squeeze(target(3,i,:));
    
    plot3(x,y,z,'LineWidth',2)
    scatter3(x(1),y(1),z(1),80,'filled') % start point
    scatter3(x(end),y(end),z(end),80,'d','filled') % end point
end

xlabel('X')
ylabel('Y')
zlabel('Z')
title('3D Drone Consensus Trajectories')