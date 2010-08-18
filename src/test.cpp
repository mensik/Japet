static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "fem.h"
#include "structures.h"
#include "petscmat.h"

PetscScalar funConst(Point n) {
	return 1;
}

int main(int argc, char *argv[]) {
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.05;
	PetscScalar (*fList[])(Point) = {funConst};

	PetscInitialize(&argc,&argv,(char *)0,help);
	PetscInt size,rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);
	
	Mesh *mesh = new Mesh();
	mesh->generateRectangularMesh(m, n, k, l, h);
	mesh->partition(size);
	
	PetscPrintf(PETSC_COMM_WORLD, "Mesh was partitioned\n");
	DistributedMesh dm;
	mesh->tear(&dm);
	delete mesh;
	Mat A;
	Vec b;
	PetscPrintf(PETSC_COMM_WORLD, "Mesh was distributed - generation A...\n");

	FEMAssemble2DLaplace(PETSC_COMM_WORLD, &dm, A, b, fList[0], fList[0]);

	//mesh.save("domain.jpm", true);
	
	//Mesh mesh2;
	//mesh2.load("domain.jpm", false);

	PetscViewer v;
	PetscPrintf(PETSC_COMM_WORLD, "Saving...\n");
	
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
	//mesh.dumpForMatlab(v);
	
	MatView(A, v);
	
	PetscViewerDestroy(v);

	PetscFinalize();
  return 0;
}
