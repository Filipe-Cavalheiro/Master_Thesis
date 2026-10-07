clear vars; close all; clc;

set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);   % affects xlabel/ylabel/title text
set(groot, 'defaultLegendFontSize', 22);

theta = linspace(-pi+1e-6, pi-1e-6, 1000);

% Equation (7)
Psi_r = 1 - cos(theta);
e_r = sin(theta);

figure;
plot(theta, Psi_r, 'LineWidth', 1.5); hold on;
plot(theta, e_r, 'LineWidth', 1.5);
grid on;
xlabel('$\theta\;[\mathrm{rad}]$', 'Interpreter', 'latex');
ylabel('$\Psi_r,\; e_r\;[-]$', 'Interpreter', 'latex');
lgd = legend({'$\Psi_r$', '$e_{r}$'}, 'Interpreter', 'latex', 'FontSize', 36);
lgd.FontWeight = 'bold';   % optional, makes it pop more
lgd.Location = 'best';     % avoids overlapping the trajectory

print2pdf("old_error", true);

% Equation (8)
Psi = 2 - 2/(sqrt(2))*sqrt(1 + cos(theta));

% e_q is undefined for theta = +/- pi %
e_q = sin(theta)./(sqrt(2)*sqrt(1+cos(theta)));

figure;
plot(theta, Psi, 'LineWidth', 1.5); hold on;
plot(theta, e_q, 'LineWidth', 1.5);
grid on;
xlabel('$\theta\;[\mathrm{rad}]$', 'Interpreter', 'latex');
ylabel('$\Psi_r,\; e_q\;[-]$', 'Interpreter', 'latex');
lgd = legend({'$\Psi$', '$e_{q}$'}, 'Interpreter', 'latex', 'FontSize', 36);
lgd.FontWeight = 'bold';   % optional, makes it pop more
lgd.Location = 'best';     % avoids overlapping the trajectory

print2pdf("new_error", true);