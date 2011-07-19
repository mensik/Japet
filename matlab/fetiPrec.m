pdLoad;
n = size(A,1);
m = size(B,1);
K = [A B';B zeros(m)];
u = K \ [b; zeros(m,1)];

xp = u(1:n);
lmbp = u(n+1:end);