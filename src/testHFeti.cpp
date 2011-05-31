static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "fem.h"
#include "solver.h"
#include "feti.h"

PetscReal funConst(Point n) {
	return 1;
}

PetscReal funSin(Point n) {
	return sin(n.x + n.y);
}

PetscReal funStep(Point n) {

	PetscReal c = n.x > 2 ? -1 : 1;
	c *= n.y > 2 ? -1 : 1;
	return c;
}

bool cf(PetscInt itNumber, PetscReal rNorm, Vec *r) {
	PetscPrintf(PETSC_COMM_SELF, "%d - %e\n", itNumber, rNorm);
	return itNumber > 5;
}

int main(int argc, char *argv[]) {
	PetscReal (*fList[])(Point) = {funConst, funSin, funStep};
	PetscErrorCode ierr;
	PetscMPIInt rank, size;
	PetscReal m = 0.0, n = 1.0, k = 0.0, l = 1.0, h = 0.1;
	PetscInitialize(&argc, &argv, 0, help);
	PetscInt f = 2;
	PetscTruth flg;
	char fileName[PETSC_MAX_PATH_LEN] = "matlab/out.m";

	PetscOptionsGetReal(PETSC_NULL, "-test_m", &m, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-test_n", &n, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-test_k", &k, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-test_l", &l, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-test_h", &h, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-test_f", &f, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-test_out_file", fileName, PETSC_MAX_PATH_LEN
			- 1, &flg);
	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		PetscViewer v;
		Mesh *mesh = new Mesh();

		PetscPrintf(PETSC_COMM_WORLD, "Generating mesh ... ");
		//mesh->generateRectangularMesh(m, n, k, l, h);
		mesh->loadHDF5("benchmarks/mesh.med");
		PetscPrintf(PETSC_COMM_WORLD, "done.\n\nTearing mesh ...");
		mesh->partition(size);

		mesh->tear();
		PetscPrintf(PETSC_COMM_WORLD, "done.\n\n");

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		mesh->dumpForMatlab(v);
		PetscViewerDestroy(v);

		mesh->saveHDF5("outMesh.med");

		Mat B;
		Vec lmb;
		GenerateJumpOperator(mesh, B, lmb);

		SubdomainCluster cluster;
		mesh->createCluster(&cluster);

		Mat Bl, Bg;
		Vec lmbG, lmbL;
		//GenerateJumpOperator(mesh, Bg, lmbG);
		GenerateClusterJumpOperator(mesh, &cluster, Bg, lmbG, Bl, lmbL);
		Generate2DLaplaceClusterNullSpace(mesh, &cluster);

		NullSpaceInfo nullSpace;

		Generate2DLaplaceNullSpace(mesh, nullSpace.isDomainSingular, nullSpace.isSubDomainSingular, &(nullSpace.R));

		Mat A;
		Vec b, x;
		FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh, A, b, funConst, funConst);

		HFeti
				*hFeti =
						new HFeti(A, b, Bg, Bl, lmbG, lmbL, &cluster, mesh->vetrices.size(), PETSC_COMM_WORLD);
		Feti1
				*feti =
						new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		delete mesh;

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/system.m", FILE_MODE_WRITE, &v);
		feti->dumpSystem(v);
		hFeti->dumpSolution(v);
		PetscViewerDestroy(v);

		hFeti->setIsVerbose(true);
		PetscPrintf(PETSC_COMM_WORLD, "Starting!!! \n\n");
		hFeti->solve();
		VecDuplicate(b, &x);
		hFeti->copySolution(x);
		saveScalarResultHDF5("outMesh.med", "hFetiResult", x);

		PetscPrintf(PETSC_COMM_WORLD, "Starting FETI1!!! \n\n");

		feti->setIsVerbose(true);
		feti->solve();

		Vec xFeti;
		VecDuplicate(x, &xFeti);
		feti->copySolution(xFeti);
		saveScalarResultHDF5("outMesh.med", "fetiResult", xFeti);

		VecAXPY(xFeti, -1, x);
		saveScalarResultHDF5("outMesh.med", "difference", xFeti);

		//hFeti->saveIterationInfo("hFeti.log");
		//delete hFeti;
	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}