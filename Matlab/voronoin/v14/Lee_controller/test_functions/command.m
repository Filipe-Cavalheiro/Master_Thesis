function desired = command(chaser_pos, target, Param)
% chaser_pos = [1;1;1];

target_pos = target.pos;
target_vel = target.vel;

% If target is on x <= 0, project velocity onto plane x = 0
% (remove x-component of velocity)
% if target_pos(1) <= 0
%     target_vel = [0; target_vel(2); target_vel(3)];
% end

% Velocity command
desired.v = target_vel;

% Position
desired.x = target_pos + desired.v * Param.dTi;

% Zero acceleration and higher derivatives
desired.x_2dot = [0; 0; 0];
desired.x_3dot = [0; 0; 0];
desired.x_4dot = [0; 0; 0];

desired.b1 = [1; 0; 0];
desired.b1_dot = [0; 0; 0];
desired.b1_2dot = [0; 0; 0];

end