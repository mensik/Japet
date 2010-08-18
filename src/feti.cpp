#include "feti.h"

Feti1::Feti1(DistributedMesh *mesh,PetscScalar (*f)(Point), PetscScalar (*K)(Point)) {
	
	FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh,A,b,f,K);
	GenerateJumpOperator(mesh,B,lmb);
	Generate2DLaplaceNullSpace(mesh, isSingular, isLocalSingular, &R);

	//Extrakce lokalni casti matice tuhosti A
	extractLocalAPart(A, &Aloc);
	//Sestaveni Nuloveho prostoru lokalni casti matice tuhosti A
	if (isLocalSingular) {
		MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, PETSC_NULL, &locNS);
	}

	if (isSingular){
		MatMatMult(B,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT, &G);
		
		PetscInt rank;
		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		Mat GTG;
		MatGetSize(G, &gM, &gN);
		if (!rank) { //Bloody messy hellish way to compute GTG - need to compute localy
			IS ISrows,IScols;
			ISCreateStride(PETSC_COMM_SELF,gM, 0, 1, &ISrows);
			ISCreateStride(PETSC_COMM_SELF,gN, 0, 1, &IScols);
			Mat *gl;
			MatGetSubMatrices(G, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);

			Mat GLOC, GTGloc;
			GLOC = *gl;

			MatMatMultTranspose(GLOC,GLOC,  MAT_INITIAL_MATRIX,PETSC_DEFAULT,  &GTGloc);
			MatCreateMPIDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, gN,gN, PETSC_NULL, &GTG);
			
			PetscScalar data[gN * gN];
			PetscInt idx[gN];
			for (int i = 0; i < gN; i++) {
				idx[i] = i;
			}
			MatGetValues(GTGloc, gN, idx, gN, idx, data);
			MatSetValues(GTG, gN, idx, gN, idx, data, INSERT_VALUES);
			MatDestroy(GLOC);
			MatDestroy(GTGloc);
			ISDestroy(ISrows);
			ISDestroy(IScols);
			//TODO Zeptej se Davida Horaka!!!! CO TO SAKRA JAKO JE?
		} else {
			IS ISrows,IScols;
			ISCreateStride(PETSC_COMM_SELF,0, 0, 1, &ISrows);
			ISCreateStride(PETSC_COMM_SELF,0, 0, 1, &IScols);
			Mat *gl;
			MatGetSubMatrices(G, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);
			MatCreateMPIDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, gN,gN, PETSC_NULL, &GTG);
			ISDestroy(ISrows);
			ISDestroy(IScols);
		}

		MatAssemblyBegin(GTG, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(GTG, MAT_FINAL_ASSEMBLY);
		
		KSPCreate(PETSC_COMM_WORLD, &kspG);
		KSPSetOperators(kspG, GTG, GTG, SAME_PRECONDITIONER);
		MatDestroy(GTG);

		VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, gN, &tgA);
		VecDuplicate(tgA, &tgB);
	}
	//Priprava ghostovaneho vektoru TEMP a kopie vektoru b do lokalnich casti bloc	
	VecCreateGhost(PETSC_COMM_WORLD, mesh->nVetrices, PETSC_DECIDE, 0, PETSC_NULL, &temp);
	VecCopy(b,temp);
	VecGhostGetLocalForm(temp, &tempLoc);
	VecDuplicate(tempLoc, &tempLocB);

	VecCreateSeq(PETSC_COMM_SELF, mesh->nVetrices, &bloc);
	VecCopy(tempLoc, bloc);

	//Ghostovany vektor reseni
	VecCreateGhost(PETSC_COMM_WORLD, mesh->nVetrices, PETSC_DECIDE, 0, PETSC_NULL,  &u);
	VecSet(u,0);
	
	VecGhostGetLocalForm(u, &uloc);

	KSPCreate(PETSC_COMM_SELF, &kspA);
	KSPSetOperators(kspA, Aloc, Aloc, SAME_PRECONDITIONER);
}

Feti1::~Feti1() {
	MatDestroy(A);
	VecDestroy(b);
	MatDestroy(B);
	VecDestroy(lmb);
	VecDestroy(u);
		
	KSPDestroy(kspA);
	
	MatDestroy(Aloc);
	VecDestroy(uloc);
	VecDestroy(bloc);
	
	if (isSingular) {
		MatDestroy(R);
		MatDestroy(G);
		VecDestroy(tgA);
		VecDestroy(tgB);

		KSPDestroy(kspG);
	}
	if (isLocalSingular) {
		MatNullSpaceDestroy(locNS);
	}
	VecDestroy(temp);
	VecDestroy(tempLoc);
	VecDestroy(tempLocB);

}

void Feti1::solve() {
	
	PetscInt locSizeA, locSizeM;
	MatGetSize(B, &locSizeM, &locSizeA);
	
	if (isLocalSingular) KSPSetNullSpace(kspA, locNS);	
	
	if (isLocalSingular) MatNullSpaceRemove(locNS, tempLoc,PETSC_NULL);
	
	KSPSolve(kspA, tempLoc, tempLoc);
	
	Vec d;
	VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, locSizeM, &d);
	MatMult(B, temp, d);

	if (isSingular) projectGOrth(d);
	
	//Priprava vektoru lambda
	VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, locSizeM, &lmb);
	if (isSingular) { //Je li singularni, je treba pripavit vhodne vstupni lambda
		Vec tlmb,ttlmb;
		VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, gN, &tlmb);
		MatMultTranspose(R, b, tlmb);
		VecDuplicate(tlmb, &ttlmb);
		KSPSolve(kspG, tlmb, ttlmb);
		MatMult(G, ttlmb, lmb);

		VecDestroy(tlmb);
		VecDestroy(ttlmb);
	}
	CGSolver solver(this, d, lmb);
	solver.setSolverCtr(this);
	//Solve!!!
	solver.solve();
	
	solver.getX(lmb);

	VecScale(lmb, -1);
	MatMultTransposeAdd(B, lmb, b, temp);

	if (isLocalSingular) MatNullSpaceRemove(locNS, tempLoc, PETSC_NULL);
	KSPSolve(kspA, tempLoc, uloc);
	if (isSingular) {
		Vec tLmb, bAlp,alpha;
		VecDuplicate(lmb, &tLmb);
		MatMult(B, u, tLmb);
		VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, gN, &bAlp);
		VecDuplicate(bAlp, &alpha);
		MatMultTranspose(G, tLmb, bAlp); 	
		KSPSolve(kspG, bAlp, alpha);

		VecScale(alpha, -1);
		MatMultAdd(R, alpha, u, u);
	
		VecDestroy(tLmb);
		VecDestroy(bAlp);
		VecDestroy(alpha);
	}
	VecDestroy(d);
}

void Feti1::projectGOrth(Vec in) {
	MatMultTranspose(G, in, tgA);
	KSPSolve(kspG, tgA, tgB);
	
	VecScale(in, -1);
	MatMultAdd(G, tgB, in, in);
	VecScale(in, -1);
}

bool Feti1::isConverged(PetscInt itNumber, PetscScalar norm, Vec *vec) {
	PetscPrintf(PETSC_COMM_WORLD, "It.%d: residual norm:%f\n", itNumber, norm);
	return norm < 1e-3;
}

void Feti1::applyMult(Vec in, Vec out) {
	MatMultTranspose(B, in, temp);
	//VecDuplicate(tempLoc, &tmpLoc);
	
	if (isLocalSingular) MatNullSpaceRemove(locNS, tempLoc, PETSC_NULL);
	KSPSolve(kspA, tempLoc, tempLocB);
	VecCopy(tempLocB, tempLoc);
	
	MatMult(B,temp, out);

	if (isSingular) projectGOrth(out);
}

void Feti1::dumpSolution(PetscViewer v) {
	VecView(u,v);	
	VecView(lmb,v);
}

void Feti1::dumpSystem(PetscViewer v) {
	MatView(A,v);
	VecView(b,v);
	MatView(B,v);
}

void GenerateJumpOperator(DistributedMesh *mesh,Mat &B, Vec &lmb) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	///TODO Make jump operator only local? There so need for paralelization here (so far)
	MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE,mesh->nVetrices,mesh->nPairs,PETSC_DECIDE,2,PETSC_NULL, 2, PETSC_NULL, &B);
	
	if (!rank) {
		for (int i = 0; i < mesh->nPairs; i++) {
			MatSetValue(B,i, mesh->pointPairing[i*2],1,INSERT_VALUES);
			MatSetValue(B,i, mesh->pointPairing[i*2 + 1],-1,INSERT_VALUES);
		}
	}

	MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);

	VecCreate(PETSC_COMM_WORLD,&lmb);
	VecSetSizes(lmb, PETSC_DECIDE, mesh->nPairs);	
	VecSetFromOptions(lmb);
	VecSet(lmb,0);
}

void Generate2DLaplaceNullSpace(DistributedMesh *mesh,bool &isSingular, bool &isLocalSingular, Mat *R) {
	PetscInt rank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);

	//Zjisti, zda ma subdomena na tomto procesoru dirchletovu hranici (zda je regularni)
	PetscInt hasDirchBound = 0;
	isLocalSingular = true;
	for (int i = 0; i < mesh->nVetrices; i++) {
		if (mesh->indDirchlet.count(i + mesh->startIndex) > 0) {
			hasDirchBound = 1;
			isLocalSingular = false;
			break;
		}
	}
	
	PetscInt nullSpaceDim;
	MPI_Allreduce(&hasDirchBound, &nullSpaceDim, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD); //Sum number of regular subdomains
	nullSpaceDim = size - nullSpaceDim; //Dimnesion of null space is number of subdomains without dirch. border
	PetscInt nsDomInd[nullSpaceDim];

	if (nullSpaceDim > 0) {	
		if (!rank) { //Master gathers array of singular domain indexes and sends it to all proceses
			MPI_Status stats;
			PetscInt counter = 0;
			if (hasDirchBound == 0) {
				nsDomInd[counter++] = rank;
			}

			for (; counter < nullSpaceDim; counter++) 
				MPI_Recv(nsDomInd+counter, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, PETSC_COMM_WORLD, &stats);
		} else {
			if (hasDirchBound == 0)
				MPI_Send(&rank, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD);
		}
		MPI_Bcast(nsDomInd, nullSpaceDim, MPI_INT, 0, PETSC_COMM_WORLD);
		
		//Creating of matrix R - null space basis
		MatCreateMPIDense(PETSC_COMM_WORLD, mesh->nVetrices, PETSC_DECIDE, PETSC_DECIDE, nullSpaceDim, PETSC_NULL, R);
		for (int i = 0; i < nullSpaceDim; i++) {
			if (nsDomInd[i] == rank) {
				for (int j = 0; j < mesh->nVetrices; j++) {
					MatSetValue(*R, j + mesh->startIndex,i, 1, INSERT_VALUES);  
				}
			}
		}
		
		MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);
		isSingular = true;
		PetscPrintf(PETSC_COMM_WORLD, "Null space dimension: %d \n", nullSpaceDim);
		}
	else {
		isSingular = false;
	}
}

