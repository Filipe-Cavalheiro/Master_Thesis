function [target] = target_motion(target, p_d, v_d, Param)
    %TARGET_MOTION Propagates target drone one timestep
    %
    % Inputs:
    %   target   - struct with fields x (18x1) and xiep (3x1)
    %   p_d      - desired position (3x1)
    %   Param    - parameter struct
    %
    % Outputs:
    %   target   - updated target struct
    %   c_sphere - target position (3x1)

    % Extract state
    pos = target.pos;
    v   = target.vel;
    R   = reshape(target.attitude,3,3);
    om  = target.ang_vel;

    % Desired quantities (simple hover / trajectory tracking)
    a_d    = zeros(3,1);
    j_d    = zeros(3,1);
    psi_d  = 0;
    dpsi_d = 0;

    % Controller (reuse drone controller)
    [T, tau] = drone_mellinger_ctrl( ...
        pos, v, R, om, Param, ...
        p_d, psi_d, ...
        v_d, dpsi_d, a_d, j_d);

    % Drone dynamics
    [dot_p, dot_v, dot_R, dot_om] = ...
        drone_3dfull_dyn(v, R, om, T, tau, Param);

    % Discretization (same as your agents)
    target.pos = pos + Param.dTi*dot_p;
    target.vel   = v   + Param.dTi*dot_v;
    target.attitude   = rot_integrate(R, om, Param.dTi);
    target.ang_vel  = om  + Param.dTi*dot_om;
end
