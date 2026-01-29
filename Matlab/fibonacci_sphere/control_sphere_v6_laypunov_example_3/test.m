% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
clearvars iD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fibonacci sphere values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fib_ref_points= get_fibonacci_points(Param.nD, Param.sphere_radius);
fib_ref_points = fib_ref_points';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iD = 1:Param.nD
    % initial values
    % Rad{iD} = Param.sphere_radius; % old algorithm
    Rad{iD} = norm(initial_pos(:,iD)); % new algorithm
    q{iD} = projectToSphere(initial_pos(:,iD)', Rad{iD})';
    omega{iD} = [0; 0; 0];            
    
    % desired values
    q_d{iD} = (fib_ref_points(:,iD)/Param.sphere_radius)*Rad{iD};

    omega_d{iD} = [0; 0; 0];
end

% Storage
for iD = 1:Param.nD
    q_hist{iD} = zeros(3, Nsim);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("Computing simulation interation: %04d\n", k);

    for iD = 1:Param.nD
        % get state vector
        pos{iD} = x{iD}(1:3,k);
        v{iD} = x{iD}(4:6,k);
        R = reshape(x{iD}(7:15,k),3,3);
        om = x{iD}(16:18,k);
    
        % position reference generation
        [Rad{iD},q{iD},q_d{iD}] = linear_motion_ref(Rad{iD},pos{iD},omega{iD},q_d{iD},omega_d{iD},Param);

        % --- Store
        q_hist{iD}(:,k) = q{iD};

        % get reference
        p_d =  q_hist{iD}(:,k);
        %v_d = v_ref{iD}(:,k);
        v_d = zeros(3,1);
        %a_d = a_ref{iD}(:,k);
        a_d = zeros(3,1);
        %j_d = j_ref{iD}(:,k);
        j_d = zeros(3,1);
        %psi_d = psi_ref{iD}(:,k);
        psi_d = 0;
        %dpsi_d = dpsi_ref{iD}(:,k);
        dpsi_d = 0;
    
        % mellinger controller
        [T{iD}(:,k),tau{iD}(:,k),e_p] = drone_mellinger_ctrl(pos{iD},v{iD},R,om,Param,p_d,psi_d,xiep{iD}(:,k),v_d,dpsi_d,a_d,j_d);
        
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

% trim vectors to same length
clear pos;
for iD = 1:Param.nD
    pos{iD} = x{iD}(1:3,:);
end

figure; hold on; grid on; axis equal
[x, y, z] = sphere;
x = x * Param.sphere_radius;
y = y * Param.sphere_radius;
z = z * Param.sphere_radius;

% Draw reference sphere
hSphere = surf(x, y, z);
hSphere.FaceAlpha = 0.3; 
hSphere.EdgeColor = 'none';
hSphere.FaceColor = [0.8627,0.8627,0.8627];

% Setup colors and plot handles for agents
colors = lines(Param.nD);

%%Create a label
for iD = 1:Param.nD
    scatter3( ...
        pos{iD}(1,1), pos{iD}(2,1), pos{iD}(3,1), ...
        50, colors(iD,:), 'x', 'LineWidth', 2, ...
        'HandleVisibility','off');

        scatter3(fib_ref_points(1,iD),fib_ref_points(2,iD),...
        fib_ref_points(3,iD), 50, colors(iD,:), 'd', 'LineWidth', 2, ...
        'HandleVisibility','off');
end

hInit = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'x', 'LineWidth', 2);

hProj = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'd', 'LineWidth', 2);

legend([hInit, hProj], ...
    {'Initial Positions', 'Final Objective'});
lgd = legend;
lgd.AutoUpdate = 'off';
%%End of create a label

for iD = 1:Param.nD
    % Initialize line and marker objects
    outer_loop_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
    inner_loop_traj{iD} = plot3(NaN, NaN, NaN,'-', 'Color', colors(iD,:), 'LineWidth', 1);
    hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
end

xlabel('x'); ylabel('y'); zlabel('z');
title('3D Trajectories of Agents – Evolution Over Time');
view(3);

% Animation loop
for k = 1:Nsim
    fprintf("Showing simulation interation: %04d\n", k);

    % Index range: last 100 steps (or fewer at the beginning)
    k_start = max(1, k-99);
    k_start = max(1, k-Nsim);
    idx = k_start:k;

    for iD = 1:Param.nD
        % Update trajectory up to current time
        set(outer_loop_traj{iD}, 'XData', q_hist{iD}(1, idx), ...
                       'YData', q_hist{iD}(2, idx), ...
                       'ZData', q_hist{iD}(3, idx));
    
        % Update trajectory up to current time
        set(inner_loop_traj{iD}, 'XData', pos{iD}(1, idx), ...
                       'YData', pos{iD}(2, idx), ...
                       'ZData', pos{iD}(3, idx));
        
        % Update agent position marker
        set(hAgent{iD}, 'XData', pos{iD}(1, k), ...
                        'YData', pos{iD}(2, k), ...
                        'ZData', pos{iD}(3, k));
        
        % Update ref position marker
        set(href{iD}, 'XData', q_hist{iD}(1, k), ...
                        'YData', q_hist{iD}(2, k), ...
                        'ZData', q_hist{iD}(3, k));
    end
    drawnow;
    pause(0.01); % adjust speed of animation
end
