function [v_circle, v_cone, v_sphere] = pre_compute_meshes(P,Param)
%PRE_COMPUTE_MESHES Summary of this function goes here
%   Detailed explanation goes here
% --- pre_compute circle  ---
N = floor((P-1)/2);             % Since P = 2N + 1 works best for odd P
golden_angle = pi * (3 - sqrt(5));
x0 = zeros(P,1);
y0 = zeros(P,1);
z0 = zeros(P,1);

for k = 1:P
    r = Param.sphere_radius * sqrt(k / N);        % radius scaling
    theta = k * golden_angle;   % golden angle increment
    
    y0(k) = r * cos(theta);
    z0(k) = r * sin(theta);
end
v_circle = [x0(:), y0(:), z0(:)];

% --- pre_compute cone  ---
v_cone = [ ...
    Param.sphere_radius * (1 - (sqrt(v_circle(:,2).^2 + v_circle(:,3).^2)/Param.sphere_radius).^2), ...
    v_circle(:,2), ...
    v_circle(:,3)];

% --- pre_compute sphere  ---
phi_golden = (1 + sqrt(5)) / 2; % Golden ratio

% Latitude and phi
i = (0:P-1)';
phi = 2*pi*i/phi_golden;
theta = acos(1 - 2*(i+0.5)/P);

x_s = Param.sphere_radius * cos(theta);
y_s = Param.sphere_radius * sin(theta).*sin(phi);
z_s = Param.sphere_radius * sin(theta).*cos(phi);
v_sphere = [x_s(:), y_s(:), z_s(:)];
end

