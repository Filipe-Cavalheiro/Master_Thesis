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
        % V4 position and tangential velocity (only)
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

