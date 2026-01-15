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
if Param.nD >= 3
    fib_ref_points= get_fibonacci_points(Param.nD, Param.sphere_radius);
    fib_ref_points = fib_ref_points';
    
    % Compute geodesic distances between all pairs
    ref_geo_distances = geo_distances(fib_ref_points', Param.sphere_radius);
    
    dist_vals = zeros(Param.nD, Param.nD-1);
    
    for i = 1:Param.nD
        distances = ref_geo_distances(i, :);
        [sorted_dists, sorted_idx] = sort(distances);
        dist_vals(i,:) = sorted_dists(1:(Param.nD-1));
    end
    
    overall_avg_distance = mean(mean(dist_vals, 2)); % Mean across each row
    fprintf('overall_avg_distance: %.4f \n', overall_avg_distance);
    min_safe_dist = overall_avg_distance;
    
    clearvars dist_vals sorted_dists sorted_idx distances;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System parameters (V. Numerical Simulations)
c = 0.3;                             % Nm (s/rad)
J = diag([0.0294 0.0305 0.0495]);    % kg m^2
tau_outerloop = zeros(3,1);

% Controller gains (29a, 29b)
K_r = diag([4.234 4.392 7.128]);
K_w = diag([7.056 7.320 11.88]) * 1e-1;

% Hat (skew) operator
S = @(x) [   0   -x(3)  x(2);
           x(3)    0   -x(1);
          -x(2)  x(1)    0 ];

Q = eye(3);

for iD = 1:Param.nD
    % initial values
    q{iD} = projectToSphere(initial_pos(:,iD)', Param.sphere_radius)';
    omega{iD} = [0; 0.3; 0];            
    
    % desired values
    if Param.nD >= 3

        q_d{iD} = (fib_ref_points(:,iD) / norm(fib_ref_points(:,iD)))*Param.sphere_radius;
    else
        q_d{iD} = [0; 1; 0];
    end
    omega_d{iD} = [0; 0; 1];
end

% Storage
for iD = 1:Param.nD
    q_hist{iD} = zeros(3, Nsim);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main time loop for simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nsim
    fprintf("intereation:%d\n", k);

    for iD = 1:Param.nD
        % get state vector
        pos{iD} = x{iD}(1:3,k);
        v{iD} = x{iD}(4:6,k);
        R = reshape(x{iD}(7:15,k),3,3);
        om = x{iD}(16:18,k);
    
        % --- Configuration error vector (Eq. 12)
        q_hat   = q{iD}   / Param.sphere_radius;
        qd_hat  = q_d{iD} / Param.sphere_radius;
        e_q{iD} = (1 / (sqrt(2) * sqrt(1 + (q_hat' * qd_hat)))) * Q' * S(q_d{iD}) * q{iD};
    
        % --- Angular velocity error
        e_w{iD} = omega{iD} - omega_d{iD};
    
        % --- Control law (Eq. 28, reduced to S^2)
        u{iD} = Q'* (-K_r * Q * e_q{iD} ...
            -K_w * e_w{iD} )...
            + S(omega{iD}) * J * omega{iD} ...
            + c * omega{iD} ...
            - tau_outerloop;
    
        % --- Rigid body dynamics (Eq. 5)
        omega_dot{iD} = J \ (u{iD} - c*omega{iD} - S(omega{iD})*J*omega{iD} + tau_outerloop);
        omega{iD} = omega{iD} + Param.dTi * omega_dot{iD};
    
        % --- Reduced attitude kinematics (Eq. 4)
    
        q_dot = S(omega{iD}) * q{iD};
        q{iD} = q{iD} + Param.dTi * q_dot;
    
        % --- Enforce S^2 constraint
        q{iD} = (q{iD} / norm(q{iD}))*Param.sphere_radius;
        
        % --- Store
        q_hist{iD}(:,k)   = q{iD};
        
        q_d_dot = S(omega_d{iD}) * q_d{iD};
        q_d{iD} = q_d{iD} + Param.dTi * q_d_dot;
        
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
for iD = 1:Param.nD
    % Initialize line and marker objects
    outer_loop_traj{iD} = plot3(NaN, NaN, NaN, 'Color', colors(iD,:), 'LineWidth', 1);
    inner_loop_traj{iD} = plot3(NaN, NaN, NaN, 'Color', colors(iD,:), 'LineWidth', 1);
    hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
end
xlabel('x'); ylabel('y'); zlabel('z');
title('3D Trajectories of Agents – Evolution Over Time');
view(3);

% Animation loop
for k = 1:Nsim
    for iD = 1:Param.nD
        % Update trajectory up to current time
        set(outer_loop_traj{iD}, 'XData', q_hist{iD}(1, 1:k), ...
                       'YData', q_hist{iD}(2, 1:k), ...
                       'ZData', q_hist{iD}(3, 1:k));
    
        % Update trajectory up to current time
        set(inner_loop_traj{iD}, 'XData', pos{iD}(1, 1:k), ...
                       'YData', pos{iD}(2, 1:k), ...
                       'ZData', pos{iD}(3, 1:k));
        
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
