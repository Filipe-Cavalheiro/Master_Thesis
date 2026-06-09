function [q, e_q] = CLF_and_CBF( ...
            Rad, q, omega, q_d, omega_d, c_sphere, current_pos, iD, min_safe_dist, Param)

    % System parameters (V. Numerical Simulations)
    % original values:
    % c = 0.3;                             % Nm (s/rad)
    % Param.Iq = diag([0.0294 0.0305 0.0495]);    % kg m^2
    % tau = zeros(3,1);
    % 
    % % Controller gains (29a, 29b)
    % K_r = diag([4.234 4.392 7.128]);
    % K_w = diag([7.056 7.320 11.88]) * 1e-1;

    
    for i = 1:Param.nD
        points_on_sphere(:,i) = (current_pos(:,i)-c_sphere)/norm(current_pos(:,i)-c_sphere);
    end

    Q = eye(3);
    % ---------- Shift to sphere frame (back to origin)----------
    q_s  = q   - c_sphere;
    q_d_s  = q_d   - c_sphere;

    % ---------- Unit-sphere projection ----------
    q_hat  = q_s/ norm(q_s);
    qd_hat = q_d_s / norm(q_d_s);
    
    % ---------- Configuration error ----------
    e_q = (1 / (sqrt(2) * sqrt(1 + q_hat' * qd_hat))) * ...
          Q' * skew(qd_hat) * q_hat;

    % ---------- Angular velocity error ----------
    e_w = omega - omega_d;

    % ---------- Control law ----------
    u_nom = Q' * (-Param.K_r * Q * e_q - Param.K_w * e_w) ...
        + skew(omega) * Param.Iq * omega ...
        + Param.c * omega ...
        - Param.tau;

    u = 0;
    for i = 1:Param.nD
        if q_hat == points_on_sphere(:,i)
            continue;
        end
        h = cos(min_safe_dist) - q_hat'*points_on_sphere(:,i);
        h_dot = -(skew(omega)*q_hat)'*points_on_sphere(:,i);

        Lfh = -points_on_sphere(:,i)'*(skew(omega)*q_hat);
        Lgh = 0;

        % L^2fg
        L2gh = 0;
        % L^2fh
        L2fh = (-skew(omega)*points_on_sphere(:,i))'*(skew(omega)*q_hat)+(skew(points_on_sphere(:,i))*q_hat)'*(inv(Param.Iq)*(-Param.c*omega - skew(omega)*Param.Iq*omega));
        %LgLfh
        LgLfh = -(skew(points_on_sphere(:,i))*q)' * inv(Param.Iq);
        % Control effectiveness term

        A = -LgLfh;
        b = L2fh + Param.k1*h_dot + Param.k0*h;
        % Check constraint
        if A*u_nom > b
            u = u;
        else
            % Project onto the half-space(constraint boundary)
            u = u + ((A*u_nom - b)/(A*A')) * A';
        end
    end
    u = u_nom + u;

    % ---------- Angular dynamics ----------
    omega_dot = Param.Iq \ (u - Param.c*omega - skew(omega)*Param.Iq*omega + Param.tau);
    omega = omega + Param.dTi * omega_dot;

    % ---------- Reduced attitude kinematics ----------
    q_dot = skew(omega) * q_s;
    q_s = q_s + Param.dTi * q_dot;

    % ---------- Enforce S² constraint (local) ----------
    q_s = Rad * q_s  / norm(q_s);

    % ---------- Shift back to world frame ----------
    q = q_s  + c_sphere;
end
