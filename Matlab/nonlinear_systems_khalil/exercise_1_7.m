P = 0.815;
u = 1.22;
n1 = 2;
n2 = 2.7;
n3 = 1.7;
tau = 6.6;
M = 0.0147;
D_M = 4;

sys x1 x2 x3
0 = (P-D*x2-n1*x3*sin(x1))/M;
0 = (-n2*x3+n3*cos(x1)+u)/tau;
x2 = 0;
