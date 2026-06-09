function Xdot = eom(t, X, desired, Sim, Param)

e3 = [0, 0, 1]';
m = Param.drone_mass;
J = Param.J;

[~, v, R, W, ~, ~] = split_to_states(X);

[f, M, ei_dot, eI_dot, ~, ~] = lee_position_control(X, desired, Sim, Param);

xdot = v;
vdot = Param.gravity * e3 ...
    - f / m * R * e3 + Param.x_delta / m;
Wdot = J \ (-hat(W) * J * W + M + Param.R_delta);
Rdot = R * hat(W);

Xdot=[xdot; vdot; Wdot; reshape(Rdot,9,1); ei_dot; eI_dot];
end