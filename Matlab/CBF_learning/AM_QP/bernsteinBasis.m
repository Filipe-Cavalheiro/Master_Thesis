% bernstein/bernsteinBasis.m
function [P,Pdot,Pddot] = bernsteinBasis(n,t)
% n - polenomial degree
% t - 

N = length(t);
P = zeros(N,n+1);

for i = 0:n
    P(:,i+1) = nchoosek(n,v) .* ...
        t^v.*(1-t).^(n-v) ;
end

Pdot  = gradient(P,t(2)-t(1));
Pddot = gradient(Pdot,t(2)-t(1));

end