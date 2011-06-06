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

	PetscReal E = 2.1e5, mu = 0.3, h = 2.0, H = 100;
	PetscInt m = 3, n = 3;

	PetscInitialize(&argc, &argv, 0, help);
	PetscInt f = 2;
	PetscTruth flg;
	char fileName[PETSC_MAX_PATH_LEN] = "matlab/out.m";

	PetscOptionsGetInt(PETSC_NULL, "-japet_m", &m, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_n", &n, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_h", &h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_HH", &H, PETSC_NULL);
	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		PetscViewer v;
		Mesh *mesh = new Mesh();

		PetscPrintf(PETSC_COMM_WORLD, "Generating mesh ... ");
		//mesh->generateRectangularMesh(m, n, k, l, h);

		bool bound[] = { false, false, false, true };
		mesh->generateTearedRectMesh(0, m * H, 0, n * H, h, m, n, bound);

		Mat B;
		Vec lmb;
		GenerateJumpOperator(mesh, B, lmb);

		SubdomainCluster cluster;
		mesh->generateRectMeshCluster(&cluster, m , n, 2, 2);

		Mat Bl, Bg;
		Vec lmbG, lmbL;
		//GenerateJumpOperator(mesh, Bg, lmbG);
		GenerateClusterJumpOperator(mesh, &cluster, Bg, lmbG, Bl, lmbL);
		Generate2DLaplaceClusterNullSpace(mesh, &cluster);

		Generate2DLaplaceTotalNullSpace(mesh,&nullSpace, PETSC_COMM_WORLD);

		Mat A;
		Vec b, x;
		FEMAssembleTotal2DLaplace(PETSC_COMM_WORLD, mesh, A, b, funConst, funConst);

		HFeti
				*hFeti =
						new HFeti(A, b, Bg, Bl, lmbG, lmbL, &cluster, mesh->vetrices.size(), PETSC_COMM_WORLD);
//		Feti1
//				*feti =
//						new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		delete mesh;

//		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/system.m", FILE_MODE_WRITE, &v);
//		feti->dumpSystem(v);
//		hFeti->dumpSolution(v);
//		PetscViewerDestroy(v);

		hFeti->setIsVerbose(true);
		PetscPrintf(PETSC_COMM_WORLD, "Starting!!! \n\n");
		hFeti->solve();
//		VecDuplicate(b, &x);
//		hFeti->copySolution(x);
/*
		PetscPrintf(PETSC_COMM_WORLD, "Starting FETI1!!! \n\n");

		feti->setIsVerbose(true);
		feti->solve();

		Vec xFeti;
		VecDuplicate(x, &xFeti);
		feti->copySolution(xFeti);

		VecAXPY(xFeti, -1, x);
*/
		//hFeti->saveIterationInfo("hFeti.log");
		//delete hFeti;
	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
