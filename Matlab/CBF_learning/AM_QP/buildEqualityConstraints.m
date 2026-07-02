function [Aeq,beq] = buildEqualityConstraints(...
    P,Pdot,Pddot,...
    x0,v0,a0)

Aeq = [
    P(1,:);
    Pdot(1,:);
    Pddot(1,:);
];

beq = [
    x0;
    v0;
    a0
];
end