R = null(full(A));
G = B*R;
P = eye(size(B,1)) - G*inv(G'*G)*G';

F = P' * B * pinv(full(A)) * B';
d = P' * B * pinv(full(A)) * b;
lo = G * R' * b;
