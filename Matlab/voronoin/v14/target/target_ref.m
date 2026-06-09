function [p_d, v_d] = target_ref(current_time, ref_target_traj, Param)
    %TARGET_REF Computes the next reference point along a trajectory
    % based on the current target position.

    % Output reference
    p_d = ref_target_traj.pos(current_time+Param.dTi);
    v_d = ref_target_traj.vel(current_time+Param.dTi);
end