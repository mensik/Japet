[A,b,B,u,lmb] = PetscBinaryRead('out.m');
[x,e,dirch,dual,primal] = loadMesh('mesh.m');
!x(:,3) = u;
drawMesh(x,e,dirch,dual,primal);