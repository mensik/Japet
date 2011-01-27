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
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.02;
  PetscInitialize(&argc,&argv,0,help);
	PetscInt				f = 2;
	PetscTruth 			flg;
	char fileName[PETSC_MAX_PATH_LEN]="matlab/out.m";			
	
	PetscOptionsGetReal("-test_m", &m, PETSC_NULL);
	PetscOptionsGetReal("-test_n", &n, PETSC_NULL);
	PetscOptionsGetReal("-test_k", &k, PETSC_NULL);
	PetscOptionsGetReal("-test_l", &l, PETSC_NULL);
	PetscOptionsGetReal("-test_h", &h, PETSC_NULL);
	PetscOptionsGetInt("-test_f", &f, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-test_out_file", fileName, PETSC_MAX_PATH_LEN-1, &flg);
	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		PetscViewer v;
		Mesh *mesh = new Mesh();
		mesh->generateRectangularMesh(m, n, k, l, h);

		//mesh->loadHDF5("mesh.med");

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
		mesh->dumpForMatlab(v);
		PetscViewerDestroy(v);
	
		Mat A;
		Vec b;
		Vec l;
		Vec x;

		FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh, A, b, fList[2], fList[0]);
		VecDuplicate(b, &l);
		VecDuplicate(b, &x);
		mesh->evalInNodes(funTable, &l);
		delete mesh;
		//VecSet(l, -0.04);
		VecSet(x, 0);


		PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName, FILE_MODE_WRITE, &v);

		PetscReal ANorm;
		MatNorm(A, NORM_1, &ANorm);

		PetscPrintf(PETSC_COMM_WORLD, "START\n");
		MPRGP mprgp(A,b,l,x,1,2/ANorm);
		CGSolver cg(A,b,x);
		cg.solve();
		cg.saveIterationInfo("mprgp.dat");
		PetscPrintf(PETSC_COMM_WORLD, "END\n");
		MatView(A, v);
		VecView(b, v);
		VecView(x, v);
		VecView(l, v);
		PetscViewerDestroy(v);
	
		MatDestroy(A);
		VecDestroy(b);
		VecDestroy(l);
		VecDestroy(x);

	}


	
	
	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
