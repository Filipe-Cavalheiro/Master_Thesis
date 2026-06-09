function Xdot = eom(t, X, desired, Sim, Param)

    e3 = [0, 0, 1]';
    m = Param.drone_mass;
    J = Param.J;
    
    [~, v, R, W, ~] = split_to_states(X);
    
    [f_nom, M_nom, ei_dot, eI_dot, ~] = lee_control(X, desired, Sim, Param);
    
    u_nom = - f_nom / m * R * e3 + Param.x_delta / m;
    
    xdot = v;
    vdot = Param.gravity * e3 + u_nom;
    Wdot = J \ (-hat(W) * J * W + M_nom + Param.R_delta);
    Rdot = R * hat(W);
    
    Xdot=[xdot; vdot; Wdot; reshape(Rdot,9,1); ei_dot; eI_dot];
end