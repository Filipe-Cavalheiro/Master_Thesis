% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fibonacci sphere values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%calculate Fibonacci points for the unit sphere (the radius is applied
%after
fib_ref_points= get_fibonacci_points(Param.nD, Param.sphere_radius);
fib_ref_points = fib_ref_points';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign each agent to nearest ref point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute geodesic distances between all pairs
pos_on_sphere = projectToSphere(initial_pos, Param.sphere_radius);
pos_on_sphere = pos_on_sphere(:, 1:Param.nD);
geo_distances_all = zeros(Param.nD, Param.nD);
for i = 1:Param.nD
    for j = 1:Param.nD
        a = fib_ref_points(:,i);
        b = pos_on_sphere(:,j);
        dot_product = dot(a, b);
        geo_distances_all(i, j) = Param.sphere_radius * acos(dot_product / (Param.sphere_radius^2));
    end
end

%the theoretical maximum two points will be apart is pi so 10xpi is just to
%force full one-to-one assignment
costofnonassignment = 10*pi;
[assignments, unassignedrows, unassignedcolumns] = ...
    assignmunkres(geo_distances_all, costofnonassignment);

if ~isempty(unassignedrows) || ~isempty(unassignedcolumns)
    disp("ERROR: impossible to assing 1-to-1");
    exit;
end

organized_initial_pos = zeros(3,Param.nD);
organized_Rad = zeros(1,Param.nD);
for i = 1:Param.nD
    organized_initial_pos(:,i) = initial_pos(:,assignments(i,1));
    organized_Rad(i) = Param.r0(assignments(i,1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%target motion reference
target_fun = @(t) [ ...
    (t<=1.5).*3*t + (t>1.5 & t<=3).*3*t + (t>3).*9 ; ...
    (t<=1.5).*0    + (t>1.5 & t<=3).*(-3*(t-1.5)) + (t>3).*(-4.5) ; ...
    zeros(size(t)) ];

for iD = 1:Param.nD
    % initial values
    Rad{iD} = organized_Rad(iD);
    x{iD}(1:3,1) = organized_initial_pos(:,iD);
    q{iD} = organized_initial_pos(:,iD);
    omega{iD} = [0; 0; 0];
    
    % desired values
    q_d{iD} = fib_ref_points(:,iD)/norm(fib_ref_points(:,iD))*Rad{iD};
    
    omega_d{iD} = [0; 0; 0];
end

% Storage
for iD = 1:Param.nD
    q_hist{iD} = zeros(3, Nsim);
    Rad_hist{iD} = zeros(1, Nsim);
    q_d_hist{iD} = zeros(3, Nsim);
end
c_sphere = zeros(3,Nsim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("Computing simulation interation: %04d\n", k);
    
    t = (k-1)*Param.dTi;
    c_sphere(:,k) = target_fun(t);

    for iD = 1:Param.nD
        % get state vector
        pos{iD} = x{iD}(1:3,k);
        v{iD} = x{iD}(4:6,k);
        R = reshape(x{iD}(7:15,k),3,3);
        om = x{iD}(16:18,k);

        % Position reference generation
        q{iD} = linear_motion_ref(Rad{iD}, pos{iD},omega{iD},q_d{iD},omega_d{iD},c_sphere(:,k), Param);

        % Reduce q_d radius
        [Rad{iD}, q_d{iD}] = update_radius(Rad{iD}, q{iD}, q_d{iD}, c_sphere(:,k), Param);

        % --- Store
        Rad_hist{iD}(k) = Rad{iD};
        q_hist{iD}(:,k) = q{iD};
        q_d_hist{iD}(:,k) = q_d{iD};

        % get reference
        p_d =  q{iD};
        v_d = zeros(3,1);
        a_d = zeros(3,1);
        j_d = zeros(3,1);
        psi_d = 0;
        dpsi_d = 0;
    
        % mellinger controller
        [T{iD}(:,k),tau{iD}(:,k),e_p] = drone_mellinger_ctrl(pos{iD},...
            v{iD},R,om,Param,p_d,psi_d,xiep{iD}(:,k),v_d,dpsi_d,a_d,j_d);
        
        % integrate position error
        xiep{iD}(:,k+1) = xiep{iD}(:,k) + Param.dTi*e_p;
        
        % nonlinear drone model (continuous time)
        [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(v{iD},R,om,T{iD}(:,k),tau{iD}(:,k),Param);
        
        % % discretization
        pp = pos{iD} + Param.dTi*dot_p;
        vp = v{iD} + Param.dTi*dot_v;
        Rp = rot_integrate(R,om,Param.dTi);
        omp = om + Param.dTi*dot_om;
        x{iD}(:,k+1) = [pp;vp;reshape(Rp,[],1);omp];
    
        % auxiliary drone attitude computation from rotation matrix
        lbd{iD}(:,k) = R2Euler(R);
    
        if abs(1-norm(Rp'*Rp))>1e-4
            warning(['Problems with rotation matrix integration... stoping simulation: t = ' num2str(k*Param.dTi) ' s.']);
            break;
        end
    end
end
%%
figure; hold on; grid on; axis equal; view(3);
[x_sphere, y_sphere, z_sphere] = sphere;
x_sphere = x_sphere * Param.sphere_radius;
y_sphere = y_sphere * Param.sphere_radius;
z_sphere = z_sphere * Param.sphere_radius;

% Draw reference sphere
hSphere = surf(x_sphere, y_sphere, z_sphere);
hSphere.FaceAlpha = 0.3;
hSphere.EdgeColor = 'none';
hSphere.FaceColor = [0.8627,0.8627,0.8627];

% Setup colors and plot handles for agents
colors = lines(Param.nD);

%%Create a label
for iD = 1:Param.nD
    scatter3(x{iD}(1,1), x{iD}(2,1), x{iD}(3,1), ...
        50, colors(iD,:), 'x', 'LineWidth', 2, ...
        'HandleVisibility','off');
end

hInit = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'x', 'LineWidth', 2);

hProj = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'd', 'LineWidth', 2);

legend([hInit, hProj], ...
    {'Initial Positions', 'Final Objective'});
lgd = legend;
lgd.Units = 'normalized';     % relative to figure
lgd.Position = [0.55 0.75 0.25 0.1];  % [x y width height]
lgd.AutoUpdate = 'off';
%%End of create a label

for iD = 1:Param.nD
    % Initialize line and marker objects
    outer_loop_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
    inner_loop_traj{iD} = plot3(NaN, NaN, NaN,'-', 'Color', colors(iD,:), 'LineWidth', 1);
    hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    href_ref{iD} = plot3(NaN, NaN, NaN, '+', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    fib_ref_points_animation{iD} = scatter3(NaN, NaN, NaN, 50, colors(iD,:), 'd', 'LineWidth', 2);
end

c_sphere_animation = plot3(NaN, NaN, NaN, 'k', 'LineWidth', 1);

xlabel('x'); ylabel('y'); zlabel('z');
%title('3D Trajectories of Agents – Evolution Over Time');


% Animation loop
for k = 1:10:Nsim
    fprintf("Showing simulation interation: %04d\n", k);

    % Index range: last X steps (or fewer at the beginning)
    % k_start = max(1, k-X);
    k_start = max(1, k-Nsim);
    idx = k_start:k;

    t = (k-1)*Param.dTi;

    for iD = 1:Param.nD
        % Update trajectory up to current time
        set(outer_loop_traj{iD}, 'XData', q_hist{iD}(1, idx), ...
                       'YData', q_hist{iD}(2, idx), ...
                       'ZData', q_hist{iD}(3, idx));
    
        % Update trajectory up to current time
        set(inner_loop_traj{iD}, 'XData', x{iD}(1,idx), ...
                       'YData', x{iD}(2,idx), ...
                       'ZData', x{iD}(3,idx));
        
        % Update agent position marker
        set(hAgent{iD}, 'XData', x{iD}(1, k), ...
                        'YData', x{iD}(2, k), ...
                        'ZData', x{iD}(3, k));
        
        % Update ref position marker
        set(href{iD}, 'XData', q_hist{iD}(1, k), ...
                        'YData', q_hist{iD}(2, k), ...
                        'ZData', q_hist{iD}(3, k));

        set(hSphere, ...
            'XData', c_sphere(1,k)+x_sphere, ...
            'YData', c_sphere(2,k)+y_sphere, ...
            'ZData', c_sphere(3,k)+z_sphere);

        set(c_sphere_animation, ...
            'XData', c_sphere(1,idx), ...
            'YData', c_sphere(2,idx), ...
            'ZData', c_sphere(3,idx));
        
        set(fib_ref_points_animation{iD}, ...
            'XData', c_sphere(1,k) + fib_ref_points(1,iD) ,...
            'YData', c_sphere(2,k) + fib_ref_points(2,iD), ...
            'ZData', c_sphere(3,k) + fib_ref_points(3,iD));
        

        set(href_ref{iD}, ...
            'XData', c_sphere(1,k) + q_d_hist{iD}(1, k), ...
            'YData', c_sphere(2,k) + q_d_hist{iD}(2, k), ...
            'ZData', c_sphere(3,k) + q_d_hist{iD}(3, k));
    end
    drawnow;
    pause(0.01); % adjust speed of animation
end

print2pdf("final_result_t_15.pdf",true);