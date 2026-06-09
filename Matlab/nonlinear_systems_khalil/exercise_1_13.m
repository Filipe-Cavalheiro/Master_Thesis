syms X Y
F1 = (X-Y)*(X^2+Y^2-1);
F2 = (X+Y)*(X^2+Y^2-1);

A = [F1;F2];

% Solve A == 0
eqns = A(:) == 0;   % turn matrix into vector of equations
sol = solve(eqns, [X Y]);

A_diff = [diff(F1,X) diff(F1,Y);
     diff(F2,X) diff(F2,Y)];

for k = 1:length(sol.X)
    Ak = subs(A_diff, [X Y], [sol.X(k), sol.Y(k)]);
    eigVals = eig(Ak);
    disp(eigVals)
end