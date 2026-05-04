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
    
    hp_T = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'x', 'LineWidth', 2);

    lgd = legend([hInit, hdrone, hp_d, hp_T], ...
        {'Initial Positions', 'drone', ...
         '$p_i^\star$', '$p_T$'}, ...
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

    colors = lines(Param.nD);
    
    initial_pos = cell(1,Param.nD);
    outer_traj = cell(1,Param.nD);
    inner_traj = cell(1,Param.nD);
    hAgent = cell(1,Param.nD);
    href_ref_final = cell(1,Param.nD);
     % Graphics handles
    for iD = 1:Param.nD
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
    
    % reference  target trajectory
    plot3(Param.ref_target_traj(1,:), Param.ref_target_traj(2,:), Param.ref_target_traj(3,:));
    
    data.x = x;
    data.target_position = target_position;
    data.removed_initial_pos = false;
    data.mesh_vertices = mesh_vertices;
    data.p_d_hist = p_d_hist;
    data.numAgents_history = numAgents_history;
    
    gfx.initial_pos = initial_pos;
    gfx.outer_traj = outer_traj;
    gfx.inner_traj = inner_traj;
    gfx.hAgent = hAgent;
    gfx.href_ref_final = href_ref_final;
    gfx.hCenter = hCenter;
    gfx.hdrone = hdrone;
    gfx.hInit = hInit;
    gfx.hp_d = hp_d;
    gfx.hp_T = hp_T;
    
    if Param.animation.animate == true
        for k = 1:Param.animation.step:Param.Nsim
            animation_cycle(k, data, gfx, Param);
            pause(0.2);
        end
    end

    if Param.animation.create_pdf == true
        %initialize patch
        gfx.patch = patch( ...
            'Vertices', zeros(0,3), ...
            'Faces', zeros(0,3), ...
            'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'b', ...
            'EdgeAlpha', 0.03, ...
            'FaceAlpha', 0.01);
        [v_circle, v_cone, v_sphere] = pre_compute_meshes(1001,Param);
        surf_direction = [1, 0, 0];
        faces = NaN;

        ax = gca;
        origCameraPos = get(ax, 'CameraPosition');
        origCameraTarget = get(ax, 'CameraTarget');
        origCameraUp = get(ax, 'CameraUpVector');
        origCameraViewAngle = get(ax, 'CameraViewAngle');

        steps = [50, 300, 600, 3000];
        for k = steps
            % normal view print
            set(ax, ...
                'CameraPosition', origCameraPos, ...
                'CameraTarget', origCameraTarget, ...
                'CameraUpVector', origCameraUp, ...
                'CameraViewAngle', origCameraViewAngle);
            animation_cycle(k, data, gfx, Param);
            set(gfx.patch, 'Vertices', zeros(0,3), 'Faces', zeros(0,3));
            print2pdf(sprintf("decentralized_dijktra_t_%g.pdf", k/100), 1);
            
            % close up view print
            [vertices, faces, ~] = pos_based_mesh_transform( ...
                target_position(:,k), ...
                [0,0,0], ...
                surf_direction, ...
                v_circle, v_cone, v_sphere, faces, Param);
            set(gfx.patch, 'Vertices', vertices, 'Faces', faces);
            set_dynamic_camera(ax, target_position(:,k), [-1 -1 0.4], 60, 8);
            print2pdf(sprintf("decentralized_dijktra_t_%g_close.pdf", k/100), 1);
        end
    end 
end

