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
	PetscTruth 			flg;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	PetscPrintf(PETSC_COMM_WORLD, "STARTING\n");

	PetscReal prec = 1e-2;
	char fileNameA[PETSC_MAX_PATH_LEN]="AK.m";
	char fileNameB[PETSC_MAX_PATH_LEN]="bK.m";


	PetscOptionsGetReal(PETSC_NULL, "-jpt_prec", &prec, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-jpt_file_A", fileNameA, PETSC_MAX_PATH_LEN-1, &flg);
	PetscOptionsGetString(PETSC_NULL, "-jpt_file_b", fileNameB, PETSC_MAX_PATH_LEN-1, &flg);

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

	Vec x, x2;
	VecDuplicate(b, &x);
	VecDuplicate(b, &x2);
  Solver *solver = new CGSolver(A, b, x);
	solver->setPrecision(prec);
	solver->setIsVerbose(true);
	solver->solve();
	solver->saveIterationInfo("cg.dat");

	Solver *solver2 = new ASinStep(A, b, x2);
	solver2->setPrecision(prec);
	solver2->setIsVerbose(true);
	solver2->solve();
	solver2->saveIterationInfo("asin.dat");


	PetscFinalize();
	return 0;
}
