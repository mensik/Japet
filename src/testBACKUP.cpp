static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "structures.h"
#include "fem.h"
#include "solver.h"
#include "feti.h"


PetscScalar funConst(Point n) {
	return 1;
}

PetscScalar funSin(Point n) {
	return sin(n.x + n.y);
}

void applyMat(Vec x) {
	VecSet(x, 13);
}

int main(int argc, char *argv[]) {
	PetscScalar (*fList[])(Point) = {funConst, funSin};
	PetscInt			f=0,K=0,d=1;
	PetscErrorCode 	ierr;
	PetscMPIInt			rank;
	Mesh			*mesh;
	PetscReal				m=0.0,n=2.0,k=0.0,l=2.0,h=0.5;
	//KSP							ksp;
	Mat							A;
	Vec							b;
	Mat							B;
	Vec							lmb;
	//Vec							x;
	//Iniciace
  PetscInitialize(&argc,&argv,(char *)0,help);
		
		

	{
	/*
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

	VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DECIDE, &b);
	VecSet(b, -1);
	applyMat(b);
	VecView(b, PETSC_VIEWER_STDOUT_WORLD);
	
	VecDestroy(b);
	*/
	/*
	VecCreateGhost(PETSC_COMM_WORLD, 1, PETSC_DECIDE,0, PETSC_NULL, &b);

	Vec bLoc;

	VecGhostGetLocalForm(b, &bLoc);
	VecSetValue(bLoc, 0, rank, INSERT_VALUES);

	VecView(b, PETSC_VIEWER_STDOUT_WORLD);

	PetscInt size;
	VecGetSize(b, &size);
	
	Vec t;
	VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, size, &t);
	VecSet(t, 10);

	VecCopy(t, b);

	PetscPrintf(PETSC_COMM_SELF, "%d:", rank);
	VecView(bLoc, PETSC_VIEWER_STDOUT_SELF);
*/
		
/*
	PetscOptionsGetInt("-test_f", &f, PETSC_NULL);
	PetscOptionsGetInt("-test_K", &K, PETSC_NULL);
	PetscOptionsGetInt("-test_d", &d, PETSC_NULL);
	PetscOptionsGetReal("-test_h", &h, PETSC_NULL);


	BoundSide dirchlet[] = {LEFT,TOP,RIGHT,BOTTOM};
	
	generateRectangularTearedMesh(m,n,k,l,h,4,4,d,dirchlet, &mesh);
	GenerateJumpOperator(mesh, B,lmb);
	FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh,A,b,fList[f],fList[K]);

	mesh = new RectMesh(m,n,k,l,h);
	FEMAssemble2D(PETSC_COMM_WORLD, mesh,A,b,fList[f],fList[K]);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "RectMesh constructed  : %d DOF\n",mesh->xPoints * mesh->yPoints);CHKERRQ(ierr);
	
	//Nulovani
	
	ierr = FEMSetDirchletBound(A,b,mesh->xPoints, mesh->iB);CHKERRQ(ierr);
	ierr = FEMSetDirchletBound(A,b,mesh->xPoints, mesh->iT);CHKERRQ(ierr);
	ierr = FEMSetDirchletBound(A,b,mesh->yPoints, mesh->iL);CHKERRQ(ierr);
	ierr = FEMSetDirchletBound(A,b,mesh->yPoints, mesh->iR);CHKERRQ(ierr);

	ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
	ierr = VecSet(x,0);CHKERRQ(ierr);
	//ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
	//PC pc;
	//ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
	//ierr = PCSetType(pc, PCMG);CHKERRQ(ierr);
	//ierr = PCMGSetLevels(pc, 1, PETSC_NULL);CHKERRQ(ierr);
	//ierr = KSPSetOperators(ksp, A, A, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	//ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
	//ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);

		
	//PetscInt iterationsCount;
	//PetscReal rNorm;
	
	CGSolver solver(A,b,x);
	solver.solve();
	x = solver.getX();
	//ierr = KSPGetIterationNumber(ksp, &iterationsCount);CHKERRQ(ierr);
	//ierr = KSPGetResidualNorm(ksp, &rNorm);CHKERRQ(ierr);
	//ierr = PetscPrintf(PETSC_COMM_WORLD, "Num. of iteratins : %d\n", iterationsCount);CHKERRQ(ierr);
	//ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm     : %e\n", rNorm);CHKERRQ(ierr);
	//ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


	MatDestroy(A);
	VecDestroy(b);
	VecDestroy(x);
	//KSPDestroy(ksp);

	
	PetscViewer v;	
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/out.m" , FILE_MODE_WRITE, &v);
	
	mesh->dumpForMatlab(v);

	
	PetscObjectSetName((PetscObject)A,"A_jap");
	MatView(A,v);
	PetscObjectSetName((PetscObject)b,"b_jap");
	VecView(b,v);
	PetscObjectSetName((PetscObject)B,"B_jap");
	MatView(B,v);
	PetscObjectSetName((PetscObject)lmb,"lmb_jap");
	VecView(lmb,v);

	//PetscObjectSetName((PetscObject)x,"u_jap");
	//VecView(x,v);
	PetscViewerDestroy(v);
	
		
		PetscPrintf(PETSC_COMM_WORLD, "** Dual Indexes **\n");
		
		for (std::set<PetscInt>::iterator i = mesh->indDual.begin();
				 i != mesh->indDual.end(); i++) {
			PetscPrintf(PETSC_COMM_WORLD, "%d ", *i);
		}
		PetscPrintf(PETSC_COMM_WORLD, "\n\n** Dual pairings **\n");
		
		for (int i = 0; i < mesh->n_pairings; i++) {
			PetscPrintf(PETSC_COMM_WORLD, "%d - %d   ", mesh->pointPairings[i*2], mesh->pointPairings[i*2 + 1]);
		}


		PetscPrintf(PETSC_COMM_WORLD, "\n\n** Dirchlet Indexes **\n");
		for (std::set<PetscInt>::iterator i = mesh->indDirchlet.begin();
				i != mesh->indDirchlet.end(); i++) {
			PetscPrintf(PETSC_COMM_WORLD, "%d ", *i);
		}
		PetscPrintf(PETSC_COMM_WORLD, "\n\n");
		
		if (rank == 2) {
			for (std::map<PetscInt, Element>::iterator i = mesh->element.begin();
					i != mesh->element.end(); i++) {
				PetscPrintf(PETSC_COMM_SELF, "%d: ", i->first);
				for (std::set<PetscInt>::iterator j = i->second.vetrices.begin();
						j != i->second.vetrices.end(); j++) {
					PetscPrintf(PETSC_COMM_SELF, "%d ",*j);
				}
				PetscPrintf(PETSC_COMM_SELF, "\n");
			}
		}
*/
		
		//delete mesh;
		//MatDestroy(A);
		//MatDestroy(B);
		//VecDestroy(lmb);
	}

	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
