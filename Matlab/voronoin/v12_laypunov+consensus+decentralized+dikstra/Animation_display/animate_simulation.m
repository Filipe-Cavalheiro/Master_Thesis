function animate_simulation(x, q_hist, q_d_hist, target_position, cells_history, numAgents_history, mesh_vertices, mesh_faces, p_d_hist, Param)
    if Param.animation.save_to_mp4 == true
        mp4File = fullfile(pwd,'control_sphere_v6_laypunov_example_3.mp4');
        v = VideoWriter(mp4File,'MPEG-4');
        v.FrameRate = videoFPS;
        open(v);
    end
    
    animation = figure('units','normalized','outerposition',[0 0 1 1]);
    hold on; grid on; 
    axis equal; 
    view(3);
    
    %Colors
    maxAgents = numAgents_history(1);
    colors = lines(maxAgents);
    
    for iD = 1:maxAgents
        initial_pos{iD} = scatter3(x{iD}(1,1), x{iD}(2,1), x{iD}(3,1), ...
            50, colors(iD,:), 'x', 'LineWidth', 2, ...
            'HandleVisibility','off');
    end

    %=== Create a label ===
    hInit = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'x', 'LineWidth', 2);

    hdrone = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'x');

    hq = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'o', 'filled');

    hq_d = scatter3(NaN, NaN, NaN, ...
        50, 'k', '+');

    hp_d = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'd', 'LineWidth', 2);

    lgd = legend([hInit, hdrone, hq, hq_d, hp_d], ...
        {'Initial Positions', 'drone', ...
         '$\tilde{q}_i$', '$q_i^\star$', '$p_i^\star$'}, ...
        'Interpreter', 'latex');

    lgd.FontSize = 10; 
    lgd.Interpreter = 'latex';
    lgd.Position = [0.5825    0.7677    0.0752    0.1113];  % [x y width height]
    lgd.AutoUpdate = 'off';
    % === End of create a label ===

    % campos([55.2158  -78.4364   53.1560]);
    % camtarget([2.6397    2.5720    1.7751]);
    % camup([0     0     1]);
    % camva(8.3316);

    % Sphere geometry
    % gfx.mesh_surf = trisurf(mesh_faces, ...
    % mesh_vertices(:,1)+target_position(1,1), ...
    % mesh_vertices(:,2)+target_position(2,1), ...
    % mesh_vertices(:,3)+target_position(3,1), ...
    % zeros(size(mesh_vertices,1),1));
    % gfx.mesh_surf.FaceAlpha = 0.3;
    % gfx.mesh_surf.EdgeColor = 'none';
    % gfx.mesh_surf.FaceColor = [0.86 0.86 0.86];

    maxAgents = numAgents_history(1);
    
    % Colors
    colors = lines(maxAgents);
    
    % Graphics handles
    for iD = 1:maxAgents
        outer_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
        inner_traj{iD} = plot3(NaN, NaN, NaN,'-', 'Color', colors(iD,:), 'LineWidth', 1);
        hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href_ref{iD} = plot3(NaN, NaN, NaN, '+', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href_ref_final{iD} = scatter3(NaN, NaN, NaN, 50, colors(iD,:), 'd', 'LineWidth', 2);
    end
    
    hCenter = plot3(NaN,NaN,NaN,'k','LineWidth',1.5);

    % edges
    hEdges = gobjects(maxAgents,1);
    
    for i = 1:maxAgents
        hEdges(i) = plot3(NaN,NaN,NaN,'Color',colors(i,:),'LineWidth',1.5);
    end
        
    xlabel('x'); ylabel('y'); zlabel('z');
    %title('3D Agent Trajectories');
    
    data.x = x;
    data.q_hist = q_hist;
    data.q_d_hist = q_d_hist;
    data.target_position = target_position;
    data.cells_history = cells_history;
    data.numAgents_history = numAgents_history;
    data.removed_initial_pos = false;
    data.mesh_vertices = mesh_vertices;
    data.p_d_hist = p_d_hist;
    
    gfx.initial_pos = initial_pos;
    gfx.outer_traj = outer_traj;
    gfx.inner_traj = inner_traj;
    gfx.hAgent = hAgent;
    gfx.href = href;
    gfx.href_ref = href_ref;
    gfx.href_ref_final = href_ref_final;
    gfx.hCenter = hCenter;
    gfx.hEdges = hEdges;
    gfx.hdrone = hdrone;
    gfx.hInit = hInit;
    gfx.hq = hq;
    gfx.hq_d = hq_d;
    gfx.hp_d = hp_d;
    
    if Param.animation.animate == true
        for k = 1:Param.animation.step:Param.Nsim
            animation_cycle(k, data, gfx, Param);
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

