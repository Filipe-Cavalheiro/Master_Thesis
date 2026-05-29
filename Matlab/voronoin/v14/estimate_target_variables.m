function [estimated_target_position, ema_estimated] = estimate_target_variables(target, iD, ema_estimated, active_ids_idx, connection_Laplacian, Param)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 % get target estimated position
    noisey_target = target.pos + Param.noise_std*randn(3,1);
    % exponential moving average filter
    ema_estimated(:,active_ids_idx) = (1-Param.ema_alpha)*ema_estimated(:,active_ids_idx) + Param.ema_alpha*noisey_target;
    % consensus update
    estimated_target_variables(:,iD) = ema_estimated(:,active_ids_idx)...
        - Param.dTi*ema_estimated*connection_Laplacian(:,active_ids_idx);
    estimated_target_position = estimated_target_variables(1:3,iD);        
end

