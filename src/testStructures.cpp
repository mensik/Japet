static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "fem.h"
#include "feti.h"
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

	PetscPrintf(PETSC_COMM_WORLD, "Mesh preparation ...");

	Mesh *mesh = new Mesh();

	bool bound[] = {true, true, true, true};

	mesh->generateTearedRectMesh(0, 200, 0, 200, 10, 3, 3, bound);

	PetscPrintf(PETSC_COMM_WORLD, " done\n\n");

	PetscPrintf(PETSC_COMM_WORLD, "Node count: %d \n", mesh->vetrices.size() * size);

	PetscViewer v;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
	mesh->dumpForMatlab(v);
	PetscViewerDestroy(v);

	delete mesh;
	PetscFinalize();
	return 0;
}
