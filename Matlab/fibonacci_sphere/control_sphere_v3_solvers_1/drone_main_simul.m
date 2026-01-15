% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% Filipe Cavalheiro (fs.cavalheiro@campus.fct.unl.pt)

% Summary: simulate simple dynamic system model of a drone
clear; clc; close all;
drone_init;
initial_pos = zeros(3, Param.nD);
for iD = 1:Param.nD
    initial_pos(:, iD) = x{iD}(1:3,1);
end
trajectory = control_sphere_v3(Param.nD, 5, initial_pos);

for iD = 1:Param.nD
    drone_traj = cell2mat(arrayfun(@(s) s.points(iD,:), trajectory, 'UniformOutput', false)');
    % Compute distance of each segment
    segment_lengths = vecnorm(diff(drone_traj), 2, 2);
    total_length = sum(segment_lengths);
    
    % Compute how many points per segment (proportional to length)
    points_per_segment = round(segment_lengths / total_length * (Nsim - 1));
    
    expanded_traj{iD} = drone_traj(1, :); % start with first point
    
    for i = 1:size(trajectory, 2)-1
        p1 = drone_traj(i, :);
        p2 = drone_traj(i+1, :);
        Ni = points_per_segment(i);
    
        % Generate Ni points between p1 and p2 (excluding p1)
        if Ni > 0
            t = linspace(0, 1, Ni+1);
            t = t(2:end); % skip the first to avoid duplication
            new_points = p1 + (p2 - p1) .* t';
            expanded_traj{iD} = [expanded_traj{iD}; new_points];
        end
    end
    
    % Adjust if total count doesn't match exactly (small rounding errors)
    if size(expanded_traj{iD},1) < Nsim
        expanded_traj{iD} = [expanded_traj{iD}; drone_traj(end,:)];
    elseif size(expanded_traj{iD},1) > Nsim
        expanded_traj{iD} = expanded_traj{iD}(1:Nsim, :);
    end
end 
% main time loop for simulation
for k = 1:Nsim
    fprintf("intereation:%d\n", k);
    for iD = 1:Param.nD
                
        % get state vector
        pos{iD} = x{iD}(1:3,k);
        v{iD} = x{iD}(4:6,k);
        R = reshape(x{iD}(7:15,k),3,3);
        om = x{iD}(16:18,k);
        
        % get reference
        %p_d = p_ref{iD}(:,k);
        p_d = expanded_traj{iD}(k, :)';
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
        
        % discretization 
        pp = pos{iD} + Param.dTi*dot_p;
        vp = v{iD} + Param.dTi*dot_v;
        Rp = rot_integrate(R,om,Param.dTi);
        omp = om + Param.dTi*dot_om;
        x{iD}(:,k+1) = [pp;vp;reshape(Rp,[],1);omp];

        % auxiliary drone attitude computation from rotation matrix
        lbd{iD}(:,k) = R2Euler(R);

        if abs(1-norm(Rp'*Rp))>1e-4
            warning(['Problems with rotation matrix integration... stoping simulation: t = ' num2str(k*dTi) ' s.']);
            break;
        end
    
    end

    if abs(1-norm(Rp'*Rp))>1e-4
        break;
    end
end

% trim vectors to same length
clear pos;
for iD = 1:Param.nD
    x{iD}(:,k+1) = [];
    pos{iD} = x{iD}(1:3,:);
end

%drone_show_data;

% --- Plot ---
% 3D trajectories
figure; hold on; grid on; axis equal
[x, y, z] = sphere;
x = x * Param.Rad;
y = y * Param.Rad;
z = z * Param.Rad;

% Draw reference sphere
hSphere = surf(x, y, z);
hSphere.FaceAlpha = 0.3; 
hSphere.EdgeColor = 'none';
hSphere.FaceColor = [0.8627,0.8627,0.8627];

set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure

colors = lines(Param.nD);
for iD = 1:Param.nD
    plot3(squeeze(pos{iD}(1, :)), squeeze(pos{iD}(2, :)), squeeze(pos{iD}(3, :)), ...
        'Color', colors(iD,:), 'LineWidth',0.5);
    plot3(pos{iD}(1, 1), pos{iD}(2, 1), pos{iD}(3, 1), 'o', 'Color', colors(iD,:), 'MarkerSize',6); % start
    plot3(pos{iD}(1, end), pos{iD}(2, end), pos{iD}(3, end), 'x', 'Color', colors(iD,:), 'MarkerSize',6); % end
end
view(3);
title('3D Trajectories of 9 Agents');
hold off;