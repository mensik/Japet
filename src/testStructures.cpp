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
	mesh->generateRectangularMesh(0, 1, 0, 1, 0.05);

	PetscPrintf(PETSC_COMM_WORLD, "done\n\n Tearing ....");

	mesh->partition(size);
	mesh->tear();
	SubdomainCluster cluster;
	PetscPrintf(PETSC_COMM_WORLD, "done\n\n");
	mesh->createCluster(&cluster);

	Mat Bl, Bg;
	Vec lmbG, lmbL;
	//GenerateJumpOperator(mesh, Bg, lmbG);

	GenerateClusterJumpOperator(mesh, &cluster, Bg, lmbG, Bl, lmbL);

	Generate2DLaplaceClusterNullSpace(mesh, &cluster);
	PetscViewer v;

	PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matlab/R.m", &v);
	//MatView(Rg, v);
	PetscViewerDestroy(v);

	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
	mesh->dumpForMatlab(v);
	PetscViewerDestroy(v);

	//mesh->save("testMesh.msh", false);

	delete mesh;
	PetscFinalize();
	return 0;
}
