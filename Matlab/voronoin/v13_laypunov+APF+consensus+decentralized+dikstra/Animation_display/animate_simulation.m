function animate_simulation(x, target_position, numAgents_history, mesh_vertices, p_d_hist, Param)
    if Param.animation.save_to_mp4 == true
        mp4File = fullfile(pwd,'control_sphere_v6_laypunov_example_3.mp4');
        v = VideoWriter(mp4File,'MPEG-4');
        v.FrameRate = videoFPS;
        open(v);
    end
    
    figure('units','normalized','outerposition',[0 0 1 1]);
    hold on; grid on; 
    axis equal; 
    view(3);
    
    % === Create a label ===
    hInit = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'x', 'LineWidth', 2);

    hdrone = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'x');

    hp_d = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'd', 'LineWidth', 2);

    lgd = legend([hInit, hdrone, hp_d], ...
        {'Initial Positions', 'drone', ...
         '$p_i^\star$'}, ...
        'Interpreter', 'latex');

    lgd.FontSize = 10; 
    lgd.Interpreter = 'latex';
    lgd.Position = [0.5825    0.7677    0.0752    0.1113];  % [x y width height]
    lgd.AutoUpdate = 'off';
    % === End of create a label ===

    % --- camera setup ---
    % campos([55.2158  -78.4364   53.1560]);
    % camtarget([2.6397    2.5720    1.7751]);
    % camup([0     0     1]);
    % camva(8.3316);    

    maxAgents = numAgents_history(1);
    colors = lines(maxAgents);
    
    initial_pos = cell(1,maxAgents);
    outer_traj = cell(1,maxAgents);
    inner_traj = cell(1,maxAgents);
    hAgent = cell(1,maxAgents);
    href_ref_final = cell(1,maxAgents);
     % Graphics handles
    for iD = 1:maxAgents
        initial_pos{iD} = scatter3(x{iD}(1,1), x{iD}(2,1), x{iD}(3,1), ...
            50, colors(iD,:), 'x', 'LineWidth', 2, ...
            'HandleVisibility','off');
        outer_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
        inner_traj{iD} = plot3(NaN, NaN, NaN,'-', 'Color', colors(iD,:), 'LineWidth', 1);
        hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href_ref_final{iD} = scatter3(NaN, NaN, NaN, 50, colors(iD,:), 'd', 'LineWidth', 2);
    end
    
   
    hCenter = plot3(NaN,NaN,NaN,'k','LineWidth',1.5);
        
    xlabel('x'); ylabel('y'); zlabel('z');
    %title('3D Agent Trajectories');
    
    % reference  target trajectory
    plot3(Param.ref_target_traj(1,:), Param.ref_target_traj(2,:), Param.ref_target_traj(3,:));
    
    data.x = x;
    data.target_position = target_position;
    data.numAgents_history = numAgents_history;
    data.removed_initial_pos = false;
    data.mesh_vertices = mesh_vertices;
    data.p_d_hist = p_d_hist;
    
    gfx.initial_pos = initial_pos;
    gfx.outer_traj = outer_traj;
    gfx.inner_traj = inner_traj;
    gfx.hAgent = hAgent;
    gfx.href_ref_final = href_ref_final;
    gfx.hCenter = hCenter;
    gfx.hdrone = hdrone;
    gfx.hInit = hInit;
    gfx.hp_d = hp_d;
    
    if Param.animation.animate == true
        for k = 1:Param.animation.step:Param.Nsim
            animation_cycle(k, data, gfx, Param);
            pause(0.2);
        end
    end

    if Param.animation.create_pdf == true
        animation_cycle(50, data, gfx, Param);
        print2pdf("decentralized_half_plane_dropped_moving_t_0.5.pdf", 1);
        
        animation_cycle(300, data, gfx, Param);
        print2pdf("decentralized_half_plane_dropped_moving_t_3.pdf", 1);

        animation_cycle(600, data, gfx, Param);
        print2pdf("decentralized_half_plane_dropped_moving_t_6.pdf", 1);

        animation_cycle(3000, data, gfx, Param);
        print2pdf("decentralized_half_plane_dropped_moving_t_30.pdf", 1);

        %=== Create a label ===
        hdrone = scatter3(NaN, NaN, NaN, ...
            50, 'k', 'x');

        hq = scatter3(NaN, NaN, NaN, ...
            50, 'k', 'o', 'filled');

        hq_d = scatter3(NaN, NaN, NaN, ...
            50, 'k', '+');

        lgd = legend([hdrone, hq, hq_d], ...
            {'drone', ...
             '$\tilde{q}_i$', '$q_i^\star$'});

        lgd.FontSize = 10; 
        lgd.Interpreter = 'latex';
        lgd.Position = [0.6208    0.6642    0.0752    0.1055];  % [x y width height]
        lgd.AutoUpdate = 'off';
        % === End of create a label ===

        r = Param.sphere_radius+1; % ensure cubic box
        sphere_center = data.target_position(:,3000);

        xlim(sphere_center(1) + [-r r]);
        ylim(sphere_center(2) + [-r r]);
        zlim(sphere_center(3) + [-r r]);

        view(3);
        camtarget(sphere_center);
        print2pdf("decentralized_half_plane_dropped_simulation_normal.pdf", 1);

        lgd.Position = [0.7090    0.8005    0.0752    0.1055];  % [x y width height]
        view(90,0);
        camtarget(sphere_center);
        print2pdf("decentralized_half_plane_dropped_simulation_side.pdf", 1);

        view(0,90);
        camtarget(sphere_center);
        print2pdf("decentralized_half_plane_dropped_simulation_top.pdf", 1);
    end 
end

