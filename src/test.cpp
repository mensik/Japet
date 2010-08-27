static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "fem.h"
#include "structures.h"
#include "petscmat.h"

PetscReal funConst(Point n) {
	return 1;
}

int main(int argc, char *argv[]) {
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.025;
	PetscReal (*fList[])(Point) = {funConst};

	PetscInitialize(&argc,&argv,(char *)0,help);
	PetscInt size,rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);
	
	PetscPrintf(PETSC_COMM_WORLD, "STARTING\n");
	Mesh *mesh = new Mesh();
	mesh->generateRectangularMesh(m, n, k, l, h);
	mesh->partition(size);


	PetscPrintf(PETSC_COMM_WORLD, "Mesh was partitioned\n");

	mesh->tear();

	PetscPrintf(PETSC_COMM_WORLD, "Mesh was distributed - generation A...\n");

	Mat A;
	Vec b;
	FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh, A, b, fList[0], fList[0]);

	//mesh.save("domain.jpm", true);
	
	//Mesh mesh2;
	//mesh2.load("domain.jpm", false);

	PetscViewer v;
	PetscPrintf(PETSC_COMM_WORLD, "Saving...\n");
	
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
	mesh->dumpForMatlab(v);
	PetscViewerDestroy(v);
	delete mesh;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/out.m", FILE_MODE_WRITE, &v);
	MatView(A, v);
	VecView(b, v);
	PetscViewerDestroy(v);

	PetscFinalize();
  return 0;
}
