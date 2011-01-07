static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "fem.h"
#include "structures.h"
#include "solver.h"
#include "petscmat.h"

PetscReal funConst(Point n) {
	return 1;
}

int main(int argc, char *argv[]) {

	PetscInitialize(&argc, &argv, (char *) 0, help);
	PetscInt size, rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	PetscPrintf(PETSC_COMM_WORLD, "STARTING\n");


	Mesh *mesh = new Mesh();
	mesh->generateRectangularMesh(0, 1, 0, 1, 0.01);
	mesh->partition(size);
	mesh->tear();
	mesh->analyzeDomainConection();

	PetscViewer v;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
	mesh->dumpForMatlab(v);
	PetscViewerDestroy(v);

	mesh->save("testMesh.msh",false);

	delete mesh;
	PetscFinalize();
	return 0;
}
