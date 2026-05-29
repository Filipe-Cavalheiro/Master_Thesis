function consensus_plot(data, Param)
figure;
hold on;
grid on;
axis equal;
view(3);

target_position = data.target_position;
perceived_target_hist = data.perceived_target_hist;

% Plot true trajectory
plot3(target_position(1,:), target_position(2,:), target_position(3,:), ...
      'k', 'LineWidth', 3);

[Ks, Is] = size(perceived_target_hist);

for i = 1:Is
    x = nan(1, Ks);
    y = nan(1, Ks);
    z = nan(1, Ks);

    for k = 1:Ks
        if ~isempty(perceived_target_hist{k,i})
            est = perceived_target_hist{k,i}; % expected 18x1
            x(k) = est(1);
            y(k) = est(2);
            z(k) = est(3);
        end
    end

    % Plot trajectory (NaNs automatically skipped)
    plot3(x, y, z, 'LineWidth', 2);

    % Find valid indices (where drone exists)
    valid_idx = find(~isnan(x));

    if ~isempty(valid_idx)
        % Start point
        scatter3(x(valid_idx(1)), y(valid_idx(1)), z(valid_idx(1)), ...
                 80, 'filled');

        % End point
        scatter3(x(valid_idx(end)), y(valid_idx(end)), z(valid_idx(end)), ...
                 80, 'd', 'filled');
    end
end

xlabel('X');
ylabel('Y');
zlabel('Z');

title('Consensus Tracking of Moving Target (EMA + Noisy Measurements)');

% dynamic legend
legend_entries = [{'True target'}, ...                  
    arrayfun(@(i) sprintf('Drone %d est', i), 1:Is, 'UniformOutput', false)];
legend(legend_entries{:});

print2pdf(sprintf("consensus_%d_agents", Param.nD), 1);
end

