static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "japetLa.h"
#include "fem.h"
#include "feti.h"

using namespace std;

int main(int argc, char *argv[]) {
	PetscMPIInt rank, size;
	PetscInitialize(&argc, &argv, 0, help);
	PetscViewer v;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	Mesh *mesh = new Mesh();
	mesh->loadHDF5("benchmarks/rect_small.med");

	mesh->partition(size);
	mesh->tear();

	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
	mesh->dumpForMatlab(v);
	PetscViewerDestroy(v);

	Mat A;
	Vec b;



	FEMAssemble2DElasticity(PETSC_COMM_WORLD, mesh, A, b);

	Mat B;
	Vec lmb;

	GenerateTotalJumpOperator(mesh, 2, B, lmb);

	NullSpaceInfo nullSpace;

	Generate2DElasticityNullSpace(mesh, &nullSpace, PETSC_COMM_WORLD);

	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/elast.m", FILE_MODE_WRITE, &v);
	MatView(A, v);
	VecView(b, v);
	MatView(B, v);
	PetscViewerDestroy(v);

	PetscFinalize();

	return 0;
}
