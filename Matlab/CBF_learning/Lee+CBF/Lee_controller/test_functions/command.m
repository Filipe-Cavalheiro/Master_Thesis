function desired = command(t)
%vel = 0.0667;
vel = 0.5;
% Position
desired.pos = [2.2*sin(vel*t);...
               2.2*cos(vel*t);...
               2.2*sin(vel*t)];

% Velocity command
desired.vel = [2.2*vel*cos(vel*t);
               -2.2*vel*sin(vel*t);
               2.2*vel*cos(vel*t)];

% Zero acceleration and higher derivatives
desired.x_2dot = [0; 0; 0];
desired.x_3dot = [0; 0; 0];
desired.x_4dot = [0; 0; 0];

desired.b1 = [1; 0; 0];
desired.b1_dot = [0; 0; 0];
desired.b1_2dot = [0; 0; 0];

end