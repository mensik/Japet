static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
//#include "fem.h"
//#include "structures.h"
//#include "solver.h"
#include "petscmat.h"
//#include "japetUtils.h"

//PetscReal funConst(Point n) {
//	return 1;
//}

int main(int argc, char *argv[]) {

	PetscInitialize(&argc, &argv, (char *) 0, help);

	//PDCommManager pdManager(MPI_COMM_WORLD, ALL_ONE_SAMEROOT);

	//if (pdManager.isDual()) PetscPrintf(pdManager.getDual(), "TEST");

	/*
	 PetscInt size, rank;
	 PetscTruth flg;
	 MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	 MPI_Comm_size(PETSC_COMM_WORLD, &size);

	 PetscPrintf(PETSC_COMM_WORLD, "STARTING\n");

	 PetscReal prec = 1e-4;
	 char fileNameA[PETSC_MAX_PATH_LEN] = "AK.m";
	 char fileNameB[PETSC_MAX_PATH_LEN] = "bK.m";

	 PetscOptionsGetReal(PETSC_NULL, "-jpt_prec", &prec, PETSC_NULL);
	 PetscOptionsGetString(PETSC_NULL, "-jpt_file_A", fileNameA, PETSC_MAX_PATH_LEN
	 - 1, &flg);
	 PetscOptionsGetString(PETSC_NULL, "-jpt_file_b", fileNameB, PETSC_MAX_PATH_LEN
	 - 1, &flg);

	 Mat A;
	 Vec b;
	 PetscViewer v;
	 PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileNameA, FILE_MODE_READ, &v);
	 MatLoad(v, MATSEQAIJ, &A);
	 PetscViewerDestroy(v);
	 PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileNameB, FILE_MODE_READ, &v);
	 VecLoad(v, VECSEQ, &b);
	 PetscViewerDestroy(v);

	 PetscPrintf(PETSC_COMM_WORLD, "Computing...\n");

	 Vec x, bo, x2;
	 VecDuplicate(b, &x);
	 VecDuplicate(b, &x2);
	 VecDuplicate(b, &bo);
	 VecCopy(b, bo);

	 Solver *solver = new ReCGSolver(A, b, x);
	 solver->setPrecision(prec);
	 solver->setIsVerbose(true);
	 solver->solve();
	 solver->reset(bo, x2);
	 solver->solve();
	 solver->saveIterationInfo("lanczos.dat");

	 PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/x.m", FILE_MODE_WRITE, &v);

	 VecView(x, v);

	 PetscViewerDestroy(v);

	 // VecDuplicate(b, &x);
	 // VecDuplicate(b, &x2);
	 // Solver *solverCG = new CGSolver(A, b, x2);
	 //	 solverCG->setPrecision(prec);
	 //	 solverCG->setIsVerbose(true);
	 //	 solverCG->solve();
	 //	 solverCG->saveIterationInfo("cg.dat");

	 //Solver *solver2 = new ASinStep(A, b, x2);
	 //solver2->setPrecision(prec);
	 //solver2->setIsVerbose(true);
	 //solver2->solve();
	 //solver2->saveIterationInfo("asin.dat");


	 delete solver;
	 VecDestroy(x);
	 VecDestroy(x2);
	 VecDestroy(b);
	 VecDestroy(bo);
	 MatDestroy(A);

	 */

	PetscFinalize();
	return 0;

}
