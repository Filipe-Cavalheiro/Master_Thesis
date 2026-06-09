clc; clear all; close all;

Tend = 10;
dt   = 0.005;
Nsim = round(Tend/dt) + 1;
t    = linspace(0, Tend, Nsim);

n_drones = 10;

target_position = rand(n_drones,1); % R1 positions

L = n_drones*eye(n_drones) - ones(n_drones);

disp("Correct consensus value:")
disp(mean(target_position))

for k = 1:Nsim
    target_position = target_position - dt*L*target_position;
end

disp("target positions from consensus:")
disp(target_position)