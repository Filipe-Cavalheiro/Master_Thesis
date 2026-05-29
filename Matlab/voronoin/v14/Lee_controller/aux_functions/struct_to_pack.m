function x = struct_to_pack(k, chaser)
%STRUCT_TO_PACK Summary of this function goes here
%   Detailed explanation goes here
         x = [
            chaser.pos(:,k);
            chaser.vel(:,k);
            chaser.ang_vel(:,k);
            chaser.attitude(:,k)
            chaser.ei(:,k);
            chaser.eI(:,k);
        ];
end

