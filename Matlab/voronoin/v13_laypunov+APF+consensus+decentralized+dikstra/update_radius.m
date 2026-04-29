function Rad = update_radius(Rad, q, q_d, chaser, target, Rad_d, Param)
    %   UPDATE_RADIUS controller responsible for adjusting the radius 
    %   from a current level to a desired one.
    %   Rad - radius of sphere
    %   q - drone desired position
    %   q_d - desired drone position on sphere
    %   chaser - chaser drone
    %   target - target drone
    %   Rad_d - desired radius
    %   Param - Parameters
    
    % Shift to sphere frame (back to origin)
    q_s  = q   - target(1:3);
    q_d_s  = q_d   - target(1:3);

    % Unit-sphere projection
    q_hat  = q_s/ norm(q_s);
    qd_hat = q_d_s / norm(q_d_s);
    
    % Configuration error
    e_q = (1 / (sqrt(2) * sqrt(1 + q_hat' * qd_hat))) * ...
          skew(qd_hat) * q_hat;

    % Position and tangential velocity
    r = target(1:3) - chaser(1:3);
    v = target(4:6) - chaser(4:6);

    r_hat = r / norm(r);
    v_perpendicular = v - dot(v, r_hat) * r_hat;
    
    Rad_dot = Param.k_rad * ...
        exp(-(norm(e_q) + norm(v_perpendicular))) * ...
        (Rad_d - Rad);

    Rad = Rad + Param.dTi * Rad_dot;
end

