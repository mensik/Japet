static char help[] = "My fils"
		"rst own testing utility for PETSc\n\n";

#include <iostream>
#include <sstream>
#include <string>
#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "petsclog.h"
#include "fem.h"
#include "solver.h"
#include "feti.h"
#include "smalbe.h"

using namespace std;

PetscReal funConst(Point n) {
	return 1;
}

PetscReal funSin(Point n) {
	return sin((n.x + n.y) * 2*3.1415);
}

PetscReal funConstNeg(Point n) {
	return -1;
}

PetscReal funL(Point n) {
	return (n.x > 0.25) && (n.x < 0.75) && (n.y > 0.25) && (n.y < 0.75)?-0.1:-0.01; 
}

PetscReal funTable(Point n) {
	return (n.x > 0.0) && (n.x < 0.5) && (n.y > 0.25) && (n.y < 0.75)?-0.02:-0.3;
}

bool cf(PetscInt itNumber, PetscReal rNorm, Vec *r) {
	PetscPrintf(PETSC_COMM_SELF, "%d - %e\n", itNumber, rNorm);
	return itNumber >	5;
}

int main(int argc, char *argv[]) {
	PetscReal (*fList[])(Point) = {funConst, funSin, funConstNeg};
	PetscErrorCode 	ierr;
	PetscMPIInt			rank, size;
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.01;
	PetscReal				mi=1e-2,ro=1,beta=1.1,M=3;
  PetscInitialize(&argc,&argv,0,help);
	PetscInt				f = 2;
	PetscTruth 			flg;
	char logFile[PETSC_MAX_PATH_LEN]="smalbe";
	
	PetscOptionsGetReal(PETSC_NULL, "-jpt_m", &m, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-jpt_n", &n, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-jpt_k", &k, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-jpt_l", &l, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-jpt_h", &h, PETSC_NULL);

	PetscOptionsGetInt(PETSC_NULL, "-jpt_f", &f, PETSC_NULL);

	PetscOptionsGetReal(PETSC_NULL, "-jpt_mi", &mi, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-jpt_ro", &ro, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-jpt_beta", &beta, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-jpt_sM", &M, PETSC_NULL);

	PetscOptionsGetString(PETSC_NULL, "-jpt_logfile", logFile, PETSC_MAX_PATH_LEN-1, &flg);

	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		//PetscViewer v;
		PetscLogStage meshStage, femStage, smalbeStage;
		PetscLogStageRegister("Meshing",&meshStage);
		PetscLogStageRegister("FEM",&femStage);
		PetscLogStageRegister("Smalbe",&smalbeStage);
	
		PetscLogStagePush(meshStage);
		Mesh *mesh = new Mesh();
		mesh->generateRectangularMesh(m, n, k, l, h);
		mesh->partition(size);
		mesh->tear();
		PetscLogStagePop();

		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
		//mesh->dumpForMatlab(v);
		//PetscViewerDestroy(v);
	
		Mat A;
		Vec b;
		Mat B;
		Vec c;
		Vec L;
		Vec lmb;
		PetscLogStagePush(femStage);
		FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh, A, b, fList[2], fList[0]);
		mesh->evalInNodes(funTable, &L);
		GenerateJumpOperator(mesh, B, lmb);
		VecDuplicate(lmb, &c);
		VecSet(c, 0);
		PetscLogStagePop();
		delete mesh;

		PetscInt mA,nA,mB,nB;
		MatGetSize(A, &mA, &nA);
		MatGetSize(B, &mB, &nB);
		PetscPrintf(PETSC_COMM_WORLD, "DOF: %d, Dual: %d \n\n",mA, mB);
		PetscLogStagePush(smalbeStage);
		Smalbe smalbe(A,b,B,c,L,mi, ro, beta, M);

		std::stringstream oss;
		oss << logFile << '-' << mA;
		smalbe.setLogFilename(oss.str());

		smalbe.solve();
		PetscLogStagePop();


		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName, FILE_MODE_WRITE, &v);
		//smalbe.dumpSolution(v);
		//PetscViewerDestroy(v);

	
		MatDestroy(A);
		VecDestroy(b);
		MatDestroy(B);
		VecDestroy(c);
		VecDestroy(L);
		VecDestroy(lmb);
	}


	
	
	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
