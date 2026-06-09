function [f, M, ei_dot, eI_dot, pos_cmd]= lee_control( ...
    X, desired, Sim, Param)

[f_nom, pos_cmd, ei_dot, pos_error] = ...
    lee_position_control(X, desired, Sim, Param);

[f, pos_cmd] = R3_spherical_CBF(f_nom, X, Sim, Param);

[M, eI_dot, att_error] = ...
    attitude_control_decoupled_yaw(X, ...  % states
    pos_cmd.b3c, pos_cmd.b3c_dot, pos_cmd.b3c_ddot, pos_cmd.b1c, pos_cmd.W3, pos_cmd.W3_dot, ...  % desired values
    Param... % gains and parameters
 );
end