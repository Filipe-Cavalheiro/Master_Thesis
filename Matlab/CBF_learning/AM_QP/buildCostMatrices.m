function [Q,q] = buildCostMatrices(...
    P,Pddot,...
    goal,...
    wg,ws)

Qgoal = P'*P;

Qsmooth = Pddot'*Pddot;

Q = wg*Qgoal + ws*Qsmooth;

q = -2*wg*P'*goal;
end