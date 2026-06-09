function Rad = update_radius(Rad, e_q, q, q_d, chaser, target, Param)
    %UPDATE_RADIUS Summary of this function goes here
    %   Rad - radius of sphere
    %   e_q - Error value of q position
    %   chaser - chaser drone
    %   target - target drone
    %   Param - Parameters
    % ---------- Radius dynamics (purely scalar) ----------
    V = 4;

    if V == 1
         % V1 only position
         q_s  = q - target(1:3);

        Rad_dot = Param.k_rad * 0.5 * ...
            (1 + q_s' * q_d / (norm(q_s) * norm(q_d))) * ...
            (Param.sphere_radius - Rad);
    
    
    elseif V == 2
        % V2 position (from error function)
        Rad_dot = Param.k_rad * ...
            exp(-(norm(e_q))) * ...
            (Param.sphere_radius - Rad);

    elseif V == 3
        % V3 position and velocity
        Rad_dot = Param.k_rad * ...
            exp(-(norm(e_q) + norm(target(4:6) - chaser(4:6)))) * ...
            (Param.sphere_radius - Rad);
    else
        % ---------- Shift to sphere frame (back to origin)----------
        q_s  = q   - target(1:3);
        q_d_s  = q_d   - target(1:3);
    
        % ---------- Unit-sphere projection ----------
        q_hat  = q_s/ norm(q_s);
        qd_hat = q_d_s / norm(q_d_s);
        
        % ---------- Configuration error ----------
        e_q = (1 / (sqrt(2) * sqrt(1 + q_hat' * qd_hat))) * ...
              skew(qd_hat) * q_hat;


        % V4 position and tangential velocity
        r = target(1:3) - chaser(1:3);
        v = target(4:6) - chaser(4:6);
    
        r_hat = r / norm(r);
        v_perpendicular = v - dot(v, r_hat) * r_hat;
        
        Rad_dot = Param.k_rad * ...
            exp(-(norm(e_q) + norm(v_perpendicular))) * ...
            (Param.sphere_radius - Rad);
    end

    Rad = Rad + Param.dTi * Rad_dot;
end

