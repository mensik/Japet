A = cell(16,1);
for i = 1:16
   A{i} = PetscBinaryRead(['A'  num2str(i - 1) '.m']);  
end

B = PetscBinaryRead('out.m');

BClust = cell(4,1);
RClustG = cell(4,1);
RClust = cell(4,1);
KClust = cell(4,1);


gClust = cell(4,1);
ClustOut = cell(4,1);

map = [1 2 5 6; 3 4 7 8; 9 10 13 14; 11 12 15 16];

K = blkdiag(A{1},A{2},A{3},A{4},A{5},A{6},A{7},A{8},A{9},A{10},A{11},A{12},A{13},A{14},A{15},A{16});

for i = 1:4
    [BClust{i}, RClust{i}, RClustG{i}, gClust{i}(1,:), gClust{i}(2,:), gClust{i}(3,:),ClustOut{i}(1,:),ClustOut{i}(2,:), ClustOut{i}(3,:)] = PetscBinaryRead(['out'  num2str(i - 1) '.m']);  
    KClust{i} = blkdiag(A{map(i,1)},A{map(i,2)},A{map(i,3)},A{map(i,4)});
end

G = BClust{1}*RClust{1};




