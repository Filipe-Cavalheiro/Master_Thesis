function chaser = pack_to_struct(k, chaser, X)
    %PACK_STRUCT Summary of this function goes here
    %   Detailed explanation goes here
    chaser.pos(:,k) = X(1:3); % initialize the agents on the initially desired point
    chaser.vel(:,k) = X(4:6);
    chaser.attitude(:,k) = reshape(X(10:18),[],1);
    chaser.ang_vel(:,k) = X(7:9);
    chaser.ei(:,k) = X(19:21);
    chaser.eI(:,k) = X(22:24);
end

