sd = 2;

A = cell(sd,1);
Areg = cell(sd, 1);

cd ../data

for i = 1:sd
   A{i} = PetscBinaryRead(['A'  num2str(i - 1) '.m']);  
   Areg{i} = PetscBinaryRead(['Ar'  num2str(i - 1) '.m']);  
end

K = blkdiag(A{:});
Kreg = blkdiag(Areg{:});
clear A

[b, B, R,u] = PetscBinaryRead('outP.m'); 
lmb = PetscBinaryRead('outD.m');
Fc = PetscBinaryRead('F.m');
t = PetscBinaryRead('fTest.m');
cd ../feti

[pSize, dSize] = size(B');
KKT = [K B'; B zeros(dSize)];
x = KKT \ [b; zeros(dSize,1)];

uPrec = x(1:pSize);
lmbPrec = x(pSize + 1 : pSize + dSize);
clear x

G = B * R;
P = eye(dSize) - G * inv(G' * G) * G';

reg = zeros(pSize, sd*3);

pivotNodes = [4 117 65 55 60 125 238 186 176 181];

for i = 1:10
    node = pivotNodes(i);
    reg(node*2:node*2 + 1, :) = R(node*2:node*2 + 1, :); 
end

%Kreg = K + reg*reg';

R = chol(Kreg);
F = B * (R'\(R\B'));

