clear vars; close all; clc;

theta = linspace(-pi+1e-6, pi-1e-6, 1000);

% Equation option 3
e_q = 1+sin(theta)/2;

figure;
plot(theta, e_q, 'LineWidth', 1.5);
grid on;
xlabel('$\theta\;[\mathrm{rad}]$', 'Interpreter', 'latex');
legend({'$e_q3$'}, 'Interpreter', 'latex');

% Equation option 4
e_q = exp(-abs(sin(theta)./(sqrt(2)*sqrt(1+cos(theta)))).^2);
figure;
plot(theta, e_q, 'LineWidth', 1.5);
grid on;
xlabel('$\theta\;[\mathrm{rad}]$', 'Interpreter', 'latex');

