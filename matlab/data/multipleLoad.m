A = cell(4,1);
Ar = cell(4,1);
for i = 1:4
   A{i} = PetscBinaryRead(['A'  num2str(i - 1) '.m']); 
   Ar{i} = PetscBinaryRead(['Ar'  num2str(i - 1) '.m']);  
end

B = PetscBinaryRead('out.m');



