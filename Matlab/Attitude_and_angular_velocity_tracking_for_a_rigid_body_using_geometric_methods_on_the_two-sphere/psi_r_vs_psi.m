clear vars; close all; clc;

theta = linspace(-pi+1e-6, pi-1e-6, 1000);

% Equation (7)
Psi_r = 1 - cos(theta);
e_r = sin(theta);

figure;
plot(theta, Psi_r, 'LineWidth', 1.5); hold on;
plot(theta, e_r, 'LineWidth', 1.5);
grid on;
xlabel('$\theta\;[\mathrm{rad}]$', 'Interpreter', 'latex');
legend({'$\Psi_r$', '$e_{r,3}$'}, 'Interpreter', 'latex');

% Equation (8)
Psi = 2 - 2/(sqrt(2))*sqrt(1 + cos(theta));

% e_q is undefined for theta = +/- pi %
e_q = sin(theta)./(sqrt(2)*sqrt(1+cos(theta)));

figure;
plot(theta, Psi, 'LineWidth', 1.5); hold on;
plot(theta, e_q, 'LineWidth', 1.5);
grid on;
xlabel('$\theta\;[\mathrm{rad}]$', 'Interpreter', 'latex');
legend({'$\Psi$', '$e_{q,3}$'}, 'Interpreter', 'latex');
