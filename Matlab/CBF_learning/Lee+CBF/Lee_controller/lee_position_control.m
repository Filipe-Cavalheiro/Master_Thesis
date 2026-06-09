function [f_nom, pos_cmd, ei_dot, error] = ...
    lee_position_control(X, desired, Sim, Param)

[x, v, R, W, ei, ~] = split_to_states(X);

sigma = Param.sigma;
c1 = Param.c1;
m = Param.drone_mass;
g = Param.gravity;
e3 = [0;0;1];

k = Param.Lee;

% Position errors
error.x = x - desired.pos(:, Sim.iD);
error.v = v - desired.vel;

% Translational controller
A = -k.x * error.x ...
    -k.v * error.v ...
    -m*g*e3 ...
    +m*desired.x_2dot ...
    -k.i*sat(sigma,ei);

ei_dot = error.v + c1*error.x;

b3 = R*e3;
f_nom = -dot(A,b3);

% Derivatives
ea = g*e3 ...
    -f_nom/m*b3 ...
    -desired.x_2dot ...
    +Param.x_delta/m;

A_dot = -k.x*error.v ...
        -k.v*ea ...
        +m*desired.x_3dot ...
        -k.i*satdot(sigma,ei,ei_dot);

ei_ddot = ea + c1*error.v;

b3_dot = R*hat(W)*e3;

f_dot = -dot(A_dot,b3) - dot(A,b3_dot);

eb = -f_dot/m*b3 ...
     -f_nom/m*b3_dot ...
     -desired.x_3dot;

A_ddot = -k.x*ea ...
         -k.v*eb ...
         +m*desired.x_4dot ...
         -k.i*satdot(sigma,ei,ei_ddot);

% Desired attitude generation
[b3c,b3c_dot,b3c_ddot] = ...
    deriv_unit_vector(-A,-A_dot,-A_ddot);

A2 = -hat(desired.b1)*b3c;

A2_dot = ...
    -hat(desired.b1_dot)*b3c ...
    -hat(desired.b1)*b3c_dot;

A2_ddot = ...
    -hat(desired.b1_2dot)*b3c ...
    -2*hat(desired.b1_dot)*b3c_dot ...
    -hat(desired.b1)*b3c_ddot;

[b2c,b2c_dot,b2c_ddot] = ...
    deriv_unit_vector(A2,A2_dot,A2_ddot);

b1c = hat(b2c)*b3c;

b1c_dot = hat(b2c_dot)*b3c ...
        + hat(b2c)*b3c_dot;

b1c_ddot = hat(b2c_ddot)*b3c ...
         + 2*hat(b2c_dot)*b3c_dot ...
         + hat(b2c)*b3c_ddot;

Rc = [b1c b2c b3c];
Rc_dot = [b1c_dot b2c_dot b3c_dot];
Rc_ddot = [b1c_ddot b2c_ddot b3c_ddot];

Wc = vee(Rc'*Rc_dot);
Wc_dot = vee(Rc'*Rc_ddot - hat(Wc)^2);

W3 = dot(R*e3,Rc*Wc);

W3_dot = dot(R*e3,Rc*Wc_dot) ...
       + dot(R*hat(W)*e3,Rc*Wc);

% Outputs for attitude controller
pos_cmd.b3c = b3c;
pos_cmd.b3c_dot = b3c_dot;
pos_cmd.b3c_ddot = b3c_ddot;
pos_cmd.b1c = b1c;

pos_cmd.Rc = Rc;
pos_cmd.Wc = Wc;
pos_cmd.Wc_dot = Wc_dot;

pos_cmd.W3 = W3;
pos_cmd.W3_dot = W3_dot;

end