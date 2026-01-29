function animate_simulation(x, q_hist, q_d_hist, fib_ref_points, c_sphere, Param, Nsim)
    figure('units','normalized','outerposition',[0 0 1 1]);
    hold on; grid on; axis equal; view(3);
    % Sphere geometry
    [xs, ys, zs] = sphere;
    xs = xs * Param.sphere_radius;
    ys = ys * Param.sphere_radius;
    zs = zs * Param.sphere_radius;

    hSphere = surf(xs, ys, zs);
    hSphere.FaceAlpha = 0.3;
    hSphere.EdgeColor = 'none';
    hSphere.FaceColor = [0.86 0.86 0.86];

    % Arena
    plotcube([6.5 6.7 3],[-3.25 -3.35 0],0,[1 1 1],'--');

    % Colors
    colors = lines(Param.nD);

    % Graphics handles
    for iD = 1:Param.nD
        outer_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
        inner_traj{iD} = plot3(NaN, NaN, NaN,'-', 'Color', colors(iD,:), 'LineWidth', 1);
        hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href_ref{iD} = plot3(NaN, NaN, NaN, '+', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        fib_ref_points_animation{iD} = scatter3(NaN, NaN, NaN, 50, colors(iD,:), 'd', 'LineWidth', 2);
    end

    hCenter = plot3(NaN,NaN,NaN,'k','LineWidth',1.5);

    xlabel('x'); ylabel('y'); zlabel('z');
    title('3D Agent Trajectories');

    % Animation loop
    trail_length = 200;

    for k = 1:10:Nsim
        fprintf("Showing simulation interation: %04d\n", k);
        idx = max(1,k-trail_length):k;

        for iD = 1:Param.nD
            set(outer_traj{iD}, ...
                'XData', q_hist{iD}(1,idx), ...
                'YData', q_hist{iD}(2,idx), ...
                'ZData', q_hist{iD}(3,idx));

            set(inner_traj{iD}, ...
                'XData', x{iD}(1,idx), ...
                'YData', x{iD}(2,idx), ...
                'ZData', x{iD}(3,idx));

            set(hAgent{iD}, ...
                'XData', x{iD}(1,k), ...
                'YData', x{iD}(2,k), ...
                'ZData', x{iD}(3,k));

            set(href{iD}, 'XData', q_hist{iD}(1, k), ...
                        'YData', q_hist{iD}(2, k), ...
                        'ZData', q_hist{iD}(3, k));

            % set(fib_ref_points_animation{iD}, ...
            %     'XData', fib_ref_points(1,iD) ,...
            %     'YData', fib_ref_points(2,iD), ...
            %     'ZData', fib_ref_points(3,iD));
    
            set(href_ref{iD}, ...
                'XData', q_d_hist{iD}(1, k), ...
                'YData', q_d_hist{iD}(2, k), ...
                'ZData', q_d_hist{iD}(3, k));
        end

        set(hSphere, ...
            'XData', c_sphere(1,k)+xs, ...
            'YData', c_sphere(2,k)+ys, ...
            'ZData', c_sphere(3,k)+zs);

        set(hCenter, ...
            'XData', c_sphere(1,idx), ...
            'YData', c_sphere(2,idx), ...
            'ZData', c_sphere(3,idx));     

        drawnow;
        pause(0.1);
    end
end
