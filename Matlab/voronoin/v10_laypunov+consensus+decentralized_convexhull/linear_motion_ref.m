function [q, e_q] = linear_motion_ref( ...
            Rad, q, omega, q_d, omega_d, c_sphere, Param)

    S = @(x) [   0   -x(3)  x(2);
               x(3)    0   -x(1);
              -x(2)  x(1)    0 ];

    % ---------- Shift to sphere frame (back to origin)----------
    q_s  = q   - c_sphere;
    q_d_s  = q_d   - c_sphere;

    % ---------- Unit-sphere projection ----------
    q_hat  = q_s/ norm(q_s);
    qd_hat = q_d_s / norm(q_d_s);
    
    % ---------- Configuration error ----------
    e_q = S(qd_hat) * q_hat / (sqrt(2*(1 + q_hat' * qd_hat)));

    q_dot = - Param.k_geo*S(e_q)*q_hat;

    q_s = q_s + Param.dTi * q_dot;

    % ---------- Enforce S² constraint (local) ----------
    q_s = Rad * q_s  / norm(q_s);

    % ---------- Shift back to world frame ----------
    q = q_s  + c_sphere;
end
