#include "feti.h"
AFeti::AFeti(Vec b, Mat B, Vec lmb, NullSpaceInfo *nullSpace, MPI_Comm c,
		CoarseProblemMethod cpM) {

	comm = c;
	this->b = b;
	this->B = B;
	this->lmb = lmb;
	this->cpMethod = cpM;

	MPI_Comm_rank(comm, &myRank);

	PetscLogStageRegister("Coarse stage", &coarseStage);

	isLocalSingular = nullSpace->isSubDomainSingular;
	isSingular = nullSpace->isDomainSingular;
	R = nullSpace->R;

	outerSolver = NULL;

	isVerbose = false;

	//If the matrix A is singular, the matrix G and G'G has to be prepared.
	if (isSingular) initCoarse();

	PetscInt lNodeCount;
	VecGetLocalSize(b, &lNodeCount);

	//Priprava ghostovaneho vektoru TEMP a kopie vektoru b do lokalnich casti bloc
	VecCreateGhost(comm, lNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &temp);
	VecCopy(b, temp);
	VecGhostGetLocalForm(temp, &tempLoc);

	//Ghostovany vektor reseni
	VecCreateGhost(comm, lNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &u);
	VecSet(u, 0);
}

void AFeti::initCoarse() {

	PetscInt rank;
	Mat GTG;

	switch (cpMethod) {
	case ParaCG:
		MatMatMult(B, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &G);
		MPI_Comm_rank(comm, &rank);
		MatGetSize(G, &gM, &gN);
		if (!rank) { //Bloody messy hellish way to compute GTG - need to compute localy

			IS ISrows, IScols;
			ISCreateStride(PETSC_COMM_SELF, gM, 0, 1, &ISrows);
			ISCreateStride(PETSC_COMM_SELF, gN, 0, 1, &IScols);
			Mat *gl;

			MatGetSubMatrices(G, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);

			Mat GLOC, GTGloc;
			GLOC = *gl;

			MatMatMultTranspose(GLOC, GLOC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GTGloc);
			MatCreateMPIDense(comm, PETSC_DECIDE, PETSC_DECIDE, gN, gN, PETSC_NULL, &GTG);

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

		} else {
			IS ISrows, IScols;
			ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &ISrows);
			ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &IScols);
			Mat *gl;
			MatGetSubMatrices(G, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);
			MatCreateMPIDense(comm, PETSC_DECIDE, PETSC_DECIDE, gN, gN, PETSC_NULL, &GTG);
			ISDestroy(ISrows);
			ISDestroy(IScols);
		}

		MatAssemblyBegin(GTG, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(GTG, MAT_FINAL_ASSEMBLY);

		KSPCreate(comm, &kspG);
		KSPSetOperators(kspG, GTG, GTG, SAME_PRECONDITIONER);
		KSPSetTolerances(kspG, 1e-16, 1e-16, 1e7, 15000);

		MatDestroy(GTG);

		VecCreateMPI(comm, PETSC_DECIDE, gN, &tgA);
		VecDuplicate(tgA, &tgB);

		break;
	case MasterWork:
		MatMatMult(B, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &G);
		MPI_Comm_rank(comm, &rank);
		MatGetSize(G, &gM, &gN);
		if (!rank) {

			IS ISrows, IScols;
			ISCreateStride(PETSC_COMM_SELF, gM, 0, 1, &ISrows);
			ISCreateStride(PETSC_COMM_SELF, gN, 0, 1, &IScols);
			Mat *gl;

			MatGetSubMatrices(G, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);

			Mat GLOC, GTGloc;
			GLOC = *gl;

			MatMatMultTranspose(GLOC, GLOC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GTGloc);

			PC pcGTG;
			PCCreate(PETSC_COMM_SELF, &pcGTG);
			PCSetOperators(pcGTG, GTGloc, GTGloc, SAME_PRECONDITIONER);
			PCSetType(pcGTG, "lu");
			PCSetUp(pcGTG);

			KSPCreate(PETSC_COMM_SELF, &kspG);
			KSPSetOperators(kspG, GTGloc, GTGloc, SAME_PRECONDITIONER);
			KSPSetPC(kspG, pcGTG);

			MatDestroy(GLOC);
			MatDestroy(GTGloc);
			ISDestroy(ISrows);
			ISDestroy(IScols);

		} else {
			IS ISrows, IScols;
			ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &ISrows);
			ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &IScols);
			Mat *gl;
			MatGetSubMatrices(G, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);
			ISDestroy(ISrows);
			ISDestroy(IScols);

		}

		VecCreateMPI(comm, PETSC_DECIDE, gN, &tgA);
		VecDuplicate(tgA, &tgB);

		VecScatterCreateToZero(tgA, &tgScat, &tgLocIn);
		VecDuplicate(tgLocIn, &tgLocOut);

		break;
	default:
		break;
	}
}

void AFeti::applyInvGTG(Vec in, Vec out) {
	switch (cpMethod) {
	case ParaCG:
		KSPSetTolerances(kspG, 1e-10, 1e-10, 1e7, 5000);
		KSPSolve(kspG, in, out);
		break;
	case MasterWork:
		VecScatterBegin(tgScat, in, tgLocIn, INSERT_VALUES, SCATTER_FORWARD);
		VecScatterEnd(tgScat, in, tgLocIn, INSERT_VALUES, SCATTER_FORWARD);

		if (!myRank) {
			KSPSetTolerances(kspG, 1e-10, 1e-10, 1e7, 2);
			KSPSolve(kspG, tgLocIn, tgLocOut);
		}

		VecScatterBegin(tgScat, tgLocOut, out, INSERT_VALUES, SCATTER_REVERSE);
		VecScatterEnd(tgScat, tgLocOut, out, INSERT_VALUES, SCATTER_REVERSE);
		break;
	}
}

void AFeti::projectGOrth(Vec in) {

	PetscLogStagePush(coarseStage);
	MatMultTranspose(G, in, tgA);
	applyInvGTG(tgA, tgB);
	VecScale(in, -1);
	MatMultAdd(G, tgB, in, in);
	VecScale(in, -1);
	PetscLogStagePop();
}

AFeti::~AFeti() {

	VecDestroy(lmb);
	VecDestroy(u);

	if (isSingular) {
		MatDestroy(R);
		MatDestroy(G);
		//VecDestroy(tgA);
		//VecDestroy(tgB);

		PetscPrintf(comm, "Destroy! \n");
		//KSPDestroy(kspG);
	}
	VecDestroy(temp);
	VecDestroy(tempLoc);

	if (outerSolver != NULL) delete outerSolver;

}

void AFeti::dumpSolution(PetscViewer v) {
	VecView(u, v);
	VecView(lmb, v);
}

void AFeti::dumpSystem(PetscViewer v) {
	//View A
	VecView(b, v);
	MatView(B, v);
}

void AFeti::applyMult(Vec in, Vec out, IterationManager *info) {

	outIterations++;

	MatMultTranspose(B, in, temp);
	applyInvA(temp, info);
	MatMult(B, temp, out);

}

Solver* AFeti::instanceOuterSolver(Vec d, Vec l) {
	return new CGSolver(this, d, l, this);
}

void AFeti::solve() {

	inIterations = 0;
	outIterations = 0;

	VecCopy(b, temp);

	//Preparation of the right-hand side vector d=PBA^+b
	//The matrix P is projector on the space orthogonal to range(G)
	PetscInt locSizeA, locSizeM;
	MatGetSize(B, &locSizeM, &locSizeA);
	applyInvA(temp, NULL);
	Vec d;
	VecCreateMPI(comm, PETSC_DECIDE, locSizeM, &d);
	MatMult(B, temp, d);
	if (isSingular) projectGOrth(d); //Projection

	//Feasible lambda_0 preparation
	VecCreateMPI(comm, PETSC_DECIDE, locSizeM, &lmb);
	if (isSingular) { //Je li singularni, je treba pripavit vhodne vstupni lambda
		Vec tlmb, ttlmb;
		VecCreateMPI(comm, PETSC_DECIDE, gN, &tlmb);
		MatMultTranspose(R, b, tlmb);

		VecDuplicate(tlmb, &ttlmb);

		applyInvGTG(tlmb, ttlmb);

		MatMult(G, ttlmb, lmb);
		/*
		 Vec dTemp;
		 VecDuplicate(lmb, &dTemp);
		 applyMult(lmb, dTemp, NULL);

		 VecAXPY(d, 1, dTemp);
		 //projectGOrth(d);

		 VecDestroy(dTemp);
		 */
		VecDestroy(tlmb);
		VecDestroy(ttlmb);
	}

	outerSolver = instanceOuterSolver(d, lmb);
	outerSolver->setSolverCtr(this);
	outerSolver->setIsVerbose(isVerbose);

	//Solve!!!
	outerSolver->solve();
	outerSolver->getX(lmb);

	//Solution reconstruction
	VecScale(lmb, -1);
	MatMultTransposeAdd(B, lmb, b, u);

	applyInvA(u, NULL);

	if (isSingular) {
		Vec tLmb, bAlp, alpha;
		VecDuplicate(lmb, &tLmb);
		MatMult(B, u, tLmb);
		VecCreateMPI(comm, PETSC_DECIDE, gN, &bAlp);
		VecDuplicate(bAlp, &alpha);
		MatMultTranspose(G, tLmb, bAlp);

		applyInvGTG(bAlp, alpha);

		VecScale(alpha, -1);
		MatMultAdd(R, alpha, u, u);

		VecDestroy(tLmb);
		VecDestroy(bAlp);
		VecDestroy(alpha);
	}

	VecScale(lmb, -1);
	VecDestroy(d);

	if (isVerbose) {

		//Vec precTempV, pTV;
		Vec tLmb;
		//VecDuplicate(b, &precTempV);
		//VecDuplicate(b, &pTV);
		VecDuplicate(lmb, &tLmb);

		//VecCopy(u, tempInv);
		//MatMult(Aloc, tempInvGh, tempInvGhB);
		//VecCopy(tempInvGhB, tempInvGh);

		//VecAXPY(tempInv, -1, b);
		//MatMultTransposeAdd(B, lmb, tempInv, tempInv);

		PetscReal bNorm, feasErr, uNorm;
		//PetscReal error;
		//VecNorm(tempInv, NORM_2, &error);
		VecNorm(b, NORM_2, &bNorm);

		MatMult(B, u, tLmb);
		VecNorm(tLmb, NORM_2, &feasErr);
		VecNorm(u, NORM_2, &uNorm);

		PetscPrintf(comm, "\n");
		PetscPrintf(comm, "FETI finished   Outer it: %d   Inner it: %d\n", outIterations, inIterations);
		PetscPrintf(comm, "Feasibility err: %e \n", feasErr / uNorm);
	}
}

void AFeti::copySolution(Vec out) {

	VecCopy(u, out);
}

void AFeti::copyLmb(Vec out) {

	VecCopy(lmb, out);
}

bool AFeti::isConverged(PetscInt itNumber, PetscReal norm, PetscReal bNorm,
		Vec *vec) {
	lastNorm = norm;

	return norm / bNorm < 1e-4 || itNumber > 125;
}

Feti1::Feti1(Mat A, Vec b, Mat B, Vec lmb, NullSpaceInfo *nullSpace,
		PetscInt localNodeCount, MPI_Comm comm, CoarseProblemMethod cpM) :
	AFeti(b, B, lmb, nullSpace, comm, cpM) {

	PetscInt lNodeCount;
	VecGetLocalSize(b, &lNodeCount);

	this->A = A;
	extractLocalAPart(A, &Aloc);

	//Sestaveni Nuloveho prostoru lokalni casti matice tuhosti A
	if (isLocalSingular) {
		MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, nullSpace->localDimension, nullSpace->localBasis, &locNS);
	}

	//Matrix regularization!

	PetscInt firstRow, lastRow, nCols, locNullDim, commSize, rank, nodeDim;

	VecGetOwnershipRange(b, &firstRow, &lastRow);
	MatGetSize(R, PETSC_NULL, &nCols);
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &commSize);

	locNullDim = nCols / commSize;
	nodeDim = (lastRow - firstRow) / localNodeCount; //Number of rows for each node

	PetscInt FIX_NODE_COUNT;
	PetscInt fixingNodes[5];

	if (nodeDim == 1) { //Laplace
		FIX_NODE_COUNT = 1;
		fixingNodes[0] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2;
	} else if (nodeDim == 2) { //Elasticity
		FIX_NODE_COUNT = 5;
		fixingNodes[0] = firstRow / nodeDim + 40;
		fixingNodes[1] = lastRow / nodeDim - 40;
		fixingNodes[2] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2 + 45;
		fixingNodes[3] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2 - 45;
		fixingNodes[4] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2;
	}

	Mat REG;
	MatCreateSeqAIJ(PETSC_COMM_SELF, lastRow - firstRow, locNullDim, FIX_NODE_COUNT
			* nodeDim * locNullDim, PETSC_NULL, &REG);

	const PetscReal *values;

	for (int i = 0; i < FIX_NODE_COUNT; i++)
		for (int d = 0; d < nodeDim; d++) {

			MatGetRow(R, fixingNodes[i] * nodeDim + d, PETSC_NULL, PETSC_NULL, &values);

			for (int j = 0; j < locNullDim; j++) {
				MatSetValue(REG, fixingNodes[i] * nodeDim + d - firstRow, j, values[j
						+ rank * locNullDim], INSERT_VALUES);
			}

			MatRestoreRow(R, fixingNodes[i] * nodeDim + d, PETSC_NULL, PETSC_NULL, &values);

		}

	MatAssemblyBegin(REG, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(REG, MAT_FINAL_ASSEMBLY);

	Mat REGREGT;
	Mat REGT;
	MatTranspose(REG, MAT_INITIAL_MATRIX, &REGT);
	MatMatMult(REG, REGT, MAT_INITIAL_MATRIX, 1, &REGREGT);

	//MatView(REGREGT, PETSC_VIEWER_STDOUT_SELF);

	Mat Areg;
	MatDuplicate(Aloc, MAT_COPY_VALUES, &Areg);

	MatAXPY(Areg, 1, REGREGT, DIFFERENT_NONZERO_PATTERN);

	PC pc;
	PCCreate(PETSC_COMM_SELF, &pc);
	PCSetOperators(pc, Aloc, Areg, SAME_PRECONDITIONER);
	PCSetFromOptions(pc);
	PCSetUp(pc);

	KSPCreate(PETSC_COMM_SELF, &kspA);
	KSPSetTolerances(kspA, 1e-10, 1e-10, 1e7, 3);
	KSPSetPC(kspA, pc);
	KSPSetOperators(kspA, Aloc, Areg, SAME_PRECONDITIONER);

	MatDestroy(Areg);
	MatDestroy(REGREGT);
	MatDestroy(REGT);

	PCDestroy(pc);

	if (isLocalSingular) KSPSetNullSpace(kspA, locNS);

	VecCreateGhost(comm, lNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &tempInv);
	VecSet(tempInv, 0);
	VecGhostGetLocalForm(tempInv, &tempInvGh);
	VecDuplicate(tempInvGh, &tempInvGhB);
}

void Feti1::solve() {
	AFeti::solve();

	if (isVerbose) {
		Vec precTempV, pTV;

		VecDuplicate(b, &precTempV);
		VecDuplicate(b, &pTV);

		VecCopy(u, tempInv);
		MatMult(Aloc, tempInvGh, tempInvGhB);
		VecCopy(tempInvGhB, tempInvGh);

		VecAXPY(tempInv, -1, b);
		MatMultTransposeAdd(B, lmb, tempInv, tempInv);

		PetscReal error, normB;
		VecNorm(tempInv, NORM_2, &error);
		VecNorm(b, NORM_2, &normB);

		PetscPrintf(comm, "Relative error: %e \n\n", error / normB);
	}
}

Feti1::~Feti1() {

	KSPDestroy(kspA);
	MatDestroy(Aloc);
	if (isLocalSingular) {
		MatNullSpaceDestroy(locNS);
	}
	VecDestroy(tempInv);
	VecDestroy(tempInvGh);
	VecDestroy(tempInvGhB);
}

void Feti1::applyInvA(Vec in, IterationManager *itManager) {

	VecCopy(in, tempInv);
	//if (isLocalSingular) MatNullSpaceRemove(locNS, tempInvGh, PETSC_NULL);

	KSPSolve(kspA, tempInvGh, tempInvGhB);

	PetscInt itNumber;
	KSPGetIterationNumber(kspA, &itNumber);

	inIterations += itNumber;

	VecCopy(tempInvGhB, tempInvGh);
	VecCopy(tempInv, in);

	if (itManager != NULL) {
		itManager->setIterationData("InCG.count", itNumber);
	}
}

void Feti1::applyPC(Vec g, Vec z) {
	//	VecCopy(g, z);
	if (isSingular) projectGOrth(g);

	MatMultTranspose(B, g, tempInv);

	MatMult(Aloc, tempInvGh, tempInvGhB);
	VecCopy(tempInvGhB, tempInvGh);

	MatMult(B, tempInv, z);

	if (isSingular) projectGOrth(z);

}

void Feti1::applyPrimalMult(Vec in, Vec out) {

}

InexactFeti1::InexactFeti1(Mat A, Vec b, Mat B, Vec lmb,
		NullSpaceInfo *nullSpace, PetscInt localNodeCount, MPI_Comm comm) :
	Feti1(A, b, B, lmb, nullSpace, localNodeCount, comm) {
	//	KSPSetType(kspA, KSPCG);

	//	PC prec;
	//	KSPGetPC(kspA, &prec);
	//	PCSetType(prec, PCILU);
	//	PCSetUp(prec);
	//	KSPSetPC(kspA, prec);
	//	KSPSetUp(kspA);

	outerPrec = 1e-7;
}

Solver* mFeti1::instanceOuterSolver(Vec d, Vec lmb) {
	//outerPrec = 1e-4;
	//lastNorm = 1e-4;
	//inCounter = 0;

	if (outerSolver == NULL) {
		outerSolver = new ReCGSolver(this, d, lmb, this);
	} else {
		outerSolver->reset(d, lmb);
	}

	return outerSolver;
}

Solver* InexactFeti1::instanceOuterSolver(Vec d, Vec lmb) {
	//outerPrec = 1e-4;
	//lastNorm = 1e-4;
	//inCounter = 0;
	return new ASinStep(this, d, lmb);
}

void InexactFeti1::applyInvA(Vec in, IterationManager *itManager) {

	//	KSPSetTolerances(kspA, outerPrec, outerPrec, 1e10, 1000);
	Feti1::applyInvA(in, itManager);

	//	PetscInt itNumber;
	//	KSPGetIterationNumber(kspA, &itNumber);
	//	inCounter += itNumber;

	//	if (itManager != NULL) {
	//		itManager->setIterationData("OuterPrecision", outerPrec);
	//		itManager->setIterationData("Inner CG it. count", itNumber);
	//	}
}

void InexactFeti1::setRequiredPrecision(PetscReal reqPrecision) {
	outerPrec = reqPrecision;
}

HFeti::HFeti(Mat A, Vec b, Mat BGlob, Mat BClust, Vec lmbGl, Vec lmbCl,
		SubdomainCluster *cluster, PetscInt localNodeCount, MPI_Comm comm) :
	AFeti(b, BGlob, lmbGl, cluster->outerNullSpace, comm) {

	VecCreateGhost(comm, localNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &globTemp);
	VecSet(globTemp, 0);
	VecGhostGetLocalForm(globTemp, &globTempGh);

	VecCreateGhost(cluster->clusterComm, localNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &clustTemp);
	VecSet(clustTemp, 0);
	VecGhostGetLocalForm(clustTemp, &clustTempGh);

	VecCreateMPI(cluster->clusterComm, localNodeCount, PETSC_DECIDE, &clustb);
	VecCopy(b, globTemp);
	VecCopy(globTempGh, clustTempGh);
	VecCopy(clustTemp, clustb);

	clustNullSpace = new NullSpaceInfo();
	clustNullSpace->R = cluster->Rin;
	clustNullSpace->isDomainSingular = cluster->isClusterSingular;
	clustNullSpace->isSubDomainSingular = cluster->isSubDomainSingular;

	subClusterSystem
			= new mFeti1(A, clustb, BClust, lmbCl, clustNullSpace, localNodeCount, cluster->clusterComm);
	subClusterSystem->setSystemSingular();

	//Sestaveni Nuloveho prostoru lokalni casti matice tuhosti A
	if (isLocalSingular) {
		MatNullSpaceCreate(cluster->clusterComm, PETSC_TRUE, 0, PETSC_NULL, &clusterNS);
	}
}

HFeti::~HFeti() {
	delete subClusterSystem;
}

void HFeti::applyInvA(Vec in, IterationManager *itManager) {

	VecCopy(in, globTemp);
	VecCopy(globTempGh, clustTempGh);
	if (isLocalSingular) MatNullSpaceRemove(clusterNS, clustTemp, PETSC_NULL);
	subClusterSystem->setRequiredPrecision(outerPrec);
	subClusterSystem->solve(clustTemp);
	subClusterSystem->copySolution(clustTemp);

	if (itManager != NULL) {
		itManager->setIterationData("outFETIit", subClusterSystem->getOutIterations());
		itManager->setIterationData("inFETIit", subClusterSystem->getInIterations());
	}

	inIterations += subClusterSystem->getOutIterations();

	VecCopy(clustTempGh, globTempGh);
	VecCopy(globTemp, in);
}

Solver* HFeti::instanceOuterSolver(Vec d, Vec lmb) {
	outerPrec = 1e-3;
	lastNorm = 1e-4;
	inCounter = 0;
	return new ASinStep(this, d, lmb);
}

void HFeti::setRequiredPrecision(PetscReal reqPrecision) {
	outerPrec = reqPrecision;
}

void GenerateJumpOperator(Mesh *mesh, Mat &B, Vec &lmb) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, mesh->vetrices.size(), mesh->nPairs, PETSC_DECIDE, 2, PETSC_NULL, 2, PETSC_NULL, &B);

	if (!rank) {
		for (int i = 0; i < mesh->nPairs; i++) {
			MatSetValue(B, i, mesh->pointPairing[i * 2], 1, INSERT_VALUES);
			MatSetValue(B, i, mesh->pointPairing[i * 2 + 1], -1, INSERT_VALUES);
		}
	}

	MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

	VecCreate(PETSC_COMM_WORLD, &lmb);
	VecSetSizes(lmb, PETSC_DECIDE, mesh->nPairs);
	VecSetFromOptions(lmb);
	VecSet(lmb, 0);
}

void GenerateTotalJumpOperator(Mesh *mesh, int d, Mat &B, Vec &lmb,
		PDCommManager* commManager) {
	PetscInt rank, size;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	int dSize;
	std::set<PetscInt> indDirchlet;
	for (std::set<PetscInt>::iterator i = mesh->borderEdges.begin(); i
			!= mesh->borderEdges.end(); i++) {

		for (int j = 0; j < 2; j++) {
			indDirchlet.insert(mesh->edges[*i]->vetrices[j]);
		}
	}

	dSize = indDirchlet.size();

	int dNodeCounts[size];
	MPI_Allgather(&dSize, 1, MPI_INT, dNodeCounts, 1, MPI_INT, PETSC_COMM_WORLD);

	int dSum = 0;
	for (int i = 0; i < size; i++)
		dSum += dNodeCounts[i];

	MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, mesh->vetrices.size() * d, (mesh->nPairs
			+ dSum) * d, PETSC_DECIDE, 2, PETSC_NULL, 2, PETSC_NULL, &B);

	int sIndex = 0;
	for (int i = 0; i < rank; i++)
		sIndex += dNodeCounts[i] * d;

	for (std::set<PetscInt>::iterator i = indDirchlet.begin(); i
			!= indDirchlet.end(); i++) {
		for (int j = 0; j < d; j++) {

			MatSetValue(B, sIndex++, *i * d + j, 1, INSERT_VALUES);
		}
	}

	PetscReal boundVal = 1.0 / sqrt(2.0); // Value to keep the B ortonormal

	if (!rank) {

		std::set<PetscInt> cornerInd;

		for (int i = 0; i < mesh->corners.size(); i++) {
			for (int j = 0; j < mesh->corners[i]->cornerSize; j++) {
				cornerInd.insert(mesh->corners[i]->vetrices[j]);
			}
		}

		PetscInt rowCounter = dSum;
		for (int i = 0; i < mesh->nPairs; i++) {
			if (cornerInd.count(mesh->pointPairing[i * 2]) == 0
					&& cornerInd.count(mesh->pointPairing[i * 2 + 1]) == 0) {
				for (int j = 0; j < d; j++) {

					MatSetValue(B, rowCounter * d + j, mesh->pointPairing[i * 2] * d + j, boundVal, INSERT_VALUES);
					MatSetValue(B, rowCounter * d + j, mesh->pointPairing[i * 2 + 1] * d
							+ j, -boundVal, INSERT_VALUES);

				}
				rowCounter++;
			}
		}

		//FIX more dimensions!!!
		for (int i = 0; i < mesh->corners.size(); i++) {

			PetscInt *vetrices = mesh->corners[i]->vetrices;
			PetscInt cornerSize = mesh->corners[i]->cornerSize;

			for (int j = 0; j < cornerSize - 1; j++) {

				PetscReal norm = sqrt((PetscReal) (cornerSize - j - 1) * (cornerSize
						- j - 1) + (cornerSize - j - 1));

				for (int dim = 0; dim < d; dim++) {
					MatSetValue(B, rowCounter * d + dim, vetrices[j] * d + dim, -(cornerSize
							- j - 1) / norm, INSERT_VALUES);
					for (int k = j + 1; k < cornerSize; k++) {
						MatSetValue(B, rowCounter * d + dim, vetrices[k] * d + dim, 1
								/ norm, INSERT_VALUES);
					}
				}

				rowCounter++;
			}
		}
	}

	MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

	VecCreate(PETSC_COMM_WORLD, &lmb);
	VecSetSizes(lmb, PETSC_DECIDE, (mesh->nPairs + dSum) * d);
	VecSetFromOptions(lmb);
	VecSet(lmb, 0);

}

void GenerateClusterJumpOperator(Mesh *mesh, SubdomainCluster *cluster,
		Mat &BGlob, Vec &lmbGlob, Mat &BCluster, Vec &lmbCluster) {
	PetscInt rank, subRank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_rank(cluster->clusterComm, &subRank);

	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	int dSize;
	std::set<PetscInt> indDirchlet;
	for (std::set<PetscInt>::iterator i = mesh->borderEdges.begin(); i
			!= mesh->borderEdges.end(); i++) {

		for (int j = 0; j < 2; j++) {
			indDirchlet.insert(mesh->edges[*i]->vetrices[j]);
		}
	}
	dSize = indDirchlet.size();

	int dNodeCounts[size];
	MPI_Allgather(&dSize, 1, MPI_INT, dNodeCounts, 1, MPI_INT, PETSC_COMM_WORLD);

	int dSum = 0;
	for (int i = 0; i < size; i++)
		dSum += dNodeCounts[i];

	//Preparation - scatter of the informations
	PetscInt globalPairCount, localPairCount;
	if (!rank) {
		globalPairCount = cluster->globalPairing.size() / 2;
	}
	MPI_Bcast(&globalPairCount, 1, MPI_INT, 0, PETSC_COMM_WORLD);
	if (!subRank) {
		localPairCount = cluster->localPairing.size() / 2;
	}
	MPI_Bcast(&localPairCount, 1, MPI_INT, 0, cluster->clusterComm);

	MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, mesh->vetrices.size(), globalPairCount
			+ dSum, PETSC_DECIDE, 2, PETSC_NULL, 2, PETSC_NULL, &BGlob);
	VecCreateMPI(PETSC_COMM_WORLD, mesh->vetrices.size(), PETSC_DECIDE, &lmbGlob);

	int sIndex = 0;
	for (int i = 0; i < rank; i++)
		sIndex += dNodeCounts[i];

	for (std::set<PetscInt>::iterator i = indDirchlet.begin(); i
			!= indDirchlet.end(); i++) {
		MatSetValue(BGlob, sIndex++, *i, 1, INSERT_VALUES);
	}

	PetscReal boundVal = 1.0 / sqrt(2.0); // Value is normalized for keeping the B matrix ortonormal

	if (!rank) {
		std::vector<PetscInt>::iterator i = cluster->globalPairing.begin();
		for (int rowCounter = 0; rowCounter < globalPairCount; rowCounter++) {
			MatSetValue(BGlob, rowCounter + dSum, *(i++), boundVal, INSERT_VALUES);
			MatSetValue(BGlob, rowCounter + dSum, *(i++), -boundVal, INSERT_VALUES);
		}
	}

	MatAssemblyBegin(BGlob, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(BGlob, MAT_FINAL_ASSEMBLY);
	VecSet(lmbGlob, 0);

	MatCreateMPIAIJ(cluster->clusterComm, PETSC_DECIDE, mesh->vetrices.size(), localPairCount, PETSC_DECIDE, 2, PETSC_NULL, 2, PETSC_NULL, &BCluster);
	VecCreateMPI(cluster->clusterComm, mesh->vetrices.size(), PETSC_DECIDE, &lmbCluster);

	if (!subRank) {
		std::vector<PetscInt>::iterator i = cluster->localPairing.begin();
		for (int rowCounter = 0; rowCounter < localPairCount; rowCounter++) {
			PetscInt globalIndex = *(i++);
			PetscInt clusterIndex = globalIndex
					+ cluster->startIndexesDiff[mesh->getNodeDomain(globalIndex)];
			MatSetValue(BCluster, rowCounter, clusterIndex, boundVal, INSERT_VALUES);
			globalIndex = *(i++);
			clusterIndex = globalIndex
					+ cluster->startIndexesDiff[mesh->getNodeDomain(globalIndex)];
			MatSetValue(BCluster, rowCounter, clusterIndex, -boundVal, INSERT_VALUES);
		}
	}

	MatAssemblyBegin(BCluster, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(BCluster, MAT_FINAL_ASSEMBLY);
	VecSet(lmbCluster, 0);
}

void getLocalJumpPart(Mat B, Mat *Bloc) {

	PetscInt m, n, rows;
	MatGetOwnershipRangeColumn(B, &m, &n);
	MatGetSize(B, &rows, PETSC_NULL);
	PetscInt size = n - m;

	IS ISlocal, ISlocalRows;
	ISCreateStride(PETSC_COMM_SELF, size, m, 1, &ISlocal);
	ISCreateStride(PETSC_COMM_SELF, rows, 0, 1, &ISlocalRows);
	Mat *sm;
	MatGetSubMatrices(B, 1, &ISlocalRows, &ISlocal, MAT_INITIAL_MATRIX, &sm);
	*Bloc = *sm;

}

void Generate2DLaplaceNullSpace(Mesh *mesh, bool &isSingular,
		bool &isLocalSingular, Mat *R, MPI_Comm comm) {
	PetscInt rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	//Zjisti, zda ma subdomena na tomto procesoru dirchletovu hranici (zda je regularni)
	PetscInt hasDirchBound = 0;
	isLocalSingular = true;

	if (mesh->borderEdges.size() > 0) {
		hasDirchBound = 1;
		isLocalSingular = false;
	}

	PetscInt nullSpaceDim;
	MPI_Allreduce(&hasDirchBound, &nullSpaceDim, 1, MPI_INT, MPI_SUM, comm); //Sum number of regular subdomains
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
				MPI_Recv(nsDomInd + counter, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &stats);
		} else {
			if (hasDirchBound == 0) MPI_Send(&rank, 1, MPI_INT, 0, 0, comm);
		}
		MPI_Bcast(nsDomInd, nullSpaceDim, MPI_INT, 0, comm);

		//Creating of matrix R - null space basis
		MatCreateMPIDense(comm, mesh->vetrices.size(), PETSC_DECIDE, PETSC_DECIDE, nullSpaceDim, PETSC_NULL, R);
		for (int i = 0; i < nullSpaceDim; i++) {
			if (nsDomInd[i] == rank) {
				for (std::map<PetscInt, Point*>::iterator v = mesh->vetrices.begin(); v
						!= mesh->vetrices.end(); v++) {
					MatSetValue(*R, v->first, i, 1, INSERT_VALUES);
				}
			}
		}

		MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);
		isSingular = true;
		PetscPrintf(comm, "Null space dimension: %d \n", nullSpaceDim);
	} else {
		isSingular = false;
	}
}

void Generate2DLaplaceTotalNullSpace(Mesh *mesh, NullSpaceInfo *nullSpace,
		MPI_Comm comm) {
	PetscInt rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	nullSpace->localDimension = 1;
	nullSpace->isDomainSingular = true;
	nullSpace->isSubDomainSingular = true;

	nullSpace->localBasis = new Vec[1];

	VecCreateSeq(PETSC_COMM_SELF, mesh->vetrices.size(), &(nullSpace->localBasis[0]));
	VecSet(nullSpace->localBasis[0], 1 / sqrt((double) mesh->vetrices.size()));

	Mat *R = &(nullSpace->R);
	//Creating of matrix R - null space basis
	MatCreateMPIDense(comm, mesh->vetrices.size(), PETSC_DECIDE, PETSC_DECIDE, size, PETSC_NULL, R);
	for (int i = 0; i < size; i++) {
		if (i == rank) {
			for (std::map<PetscInt, Point*>::iterator v = mesh->vetrices.begin(); v
					!= mesh->vetrices.end(); v++) {
				MatSetValue(*R, v->first, i, 1 / sqrt((double) mesh->vetrices.size()), INSERT_VALUES);
			}
		}
		MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);
	}

}

void Generate2DElasticityNullSpace(Mesh *mesh, NullSpaceInfo *nullSpace,
		MPI_Comm comm) {

	PetscInt rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	nullSpace->localDimension = 3;
	nullSpace->isDomainSingular = true;
	nullSpace->isSubDomainSingular = true;

	//Creating of matrix R - null space basis
	Mat *R = &(nullSpace->R);

	nullSpace->localBasis = new Vec[3];
	for (int i = 0; i < 3; i++) {
		VecCreateSeq(PETSC_COMM_SELF, mesh->vetrices.size() * 2, &(nullSpace->localBasis[i]));
	}

	for (int i = 0; i < size; i++) {
		if (i == rank) {
			for (std::map<PetscInt, Point*>::iterator v = mesh->vetrices.begin(); v
					!= mesh->vetrices.end(); v++) {
				VecSetValue(nullSpace->localBasis[0], (v->first - mesh->startIndexes[i])
						* 2, 1, INSERT_VALUES);
				VecSetValue(nullSpace->localBasis[1], (v->first - mesh->startIndexes[i])
						* 2 + 1, 1, INSERT_VALUES);
				VecSetValue(nullSpace->localBasis[2], (v->first - mesh->startIndexes[i])
						* 2, -v->second->y, INSERT_VALUES);
				VecSetValue(nullSpace->localBasis[2], (v->first - mesh->startIndexes[i])
						* 2 + 1, v->second->x, INSERT_VALUES);
			}
		}
	}

	//Null space ortonormalization
	PetscReal vecNorm1, vecNorm2;

	VecNorm(nullSpace->localBasis[0], NORM_2, &vecNorm1);
	VecScale(nullSpace->localBasis[0], 1 / vecNorm1);
	VecScale(nullSpace->localBasis[1], 1 / vecNorm1);

	for (int i = 0; i < 2; i++) {
		PetscReal a0;
		VecDot(nullSpace->localBasis[i], nullSpace->localBasis[2], &a0);
		VecAXPY(nullSpace->localBasis[2], -a0, nullSpace->localBasis[i]);
	}

	VecNorm(nullSpace->localBasis[2], NORM_2, &vecNorm2);
	VecScale(nullSpace->localBasis[2], 1 / vecNorm2);

	PetscInt rowIndL[mesh->vetrices.size() * 2];
	PetscInt rowIndG[mesh->vetrices.size() * 2];

	PetscInt localRSize = mesh->vetrices.size() * 2;

	for (int i = 0; i < localRSize; i++) {
		rowIndL[i] = i;
		rowIndG[i] = i + mesh->startIndexes[rank] * 2;
	}

	MatCreateMPIDense(comm, mesh->vetrices.size() * 2, PETSC_DECIDE, PETSC_DECIDE, size
			* 3, PETSC_NULL, R);

	for (int j = 0; j < 3; j++) {
		PetscReal *values;
		PetscInt colIndG = rank * 3 + j;
		VecGetArray(nullSpace->localBasis[j], &values);
		MatSetValues(*R, localRSize, rowIndG, 1, &colIndG, values, INSERT_VALUES);
		VecRestoreArray(nullSpace->localBasis[j], &values);
	}

	MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);

	//MatView(*R, PETSC_VIEWER_STDOUT_WORLD);

}

void genClusterNullSpace(Mesh *mesh, SubdomainCluster *cluster, Mat *R) {
	PetscInt rank, size;
	MPI_Comm comm = cluster->clusterComm;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	cluster->isSubDomainSingular = true;

	//Creating of matrix R - null space basis
	MatCreateMPIDense(comm, mesh->vetrices.size(), PETSC_DECIDE, PETSC_DECIDE, size, PETSC_NULL, R);
	for (int i = 0; i < size; i++) {
		if (i == rank) {
			for (std::map<PetscInt, Point*>::iterator v = mesh->vetrices.begin(); v
					!= mesh->vetrices.end(); v++) {
				MatSetValue(*R, v->first + cluster->indexDiff, i, 1, INSERT_VALUES);
			}
		}
	}

	MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);
	cluster->isClusterSingular = true;
}

void Generate2DLaplaceClusterNullSpace(Mesh *mesh, SubdomainCluster *cluster) {
	Mat RClust, RGlob;
	PetscInt rank, size, clusterRank, clusterSize;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);
	MPI_Comm_rank(cluster->clusterComm, &clusterRank);
	MPI_Comm_size(cluster->clusterComm, &clusterSize);

	genClusterNullSpace(mesh, cluster, &RClust);

	//Creating of matrix R - null space basis
	MatCreateMPIDense(PETSC_COMM_WORLD, mesh->vetrices.size(), PETSC_DECIDE, PETSC_DECIDE, cluster->clusterCount, PETSC_NULL, &RGlob);
	for (int i = 0; i < cluster->clusterCount; i++) {
		if (i == cluster->clusterColor) {
			for (std::map<PetscInt, Point*>::iterator v = mesh->vetrices.begin(); v
					!= mesh->vetrices.end(); v++) {
				MatSetValue(RGlob, v->first, i, 1, INSERT_VALUES);
			}
		}
	}

	MatAssemblyBegin(RGlob, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(RGlob, MAT_FINAL_ASSEMBLY);
	cluster->isDomainSingular = true;

	NullSpaceInfo *outNullSpace = new NullSpaceInfo();
	outNullSpace->R = RGlob;
	outNullSpace->isDomainSingular = true;
	outNullSpace->isSubDomainSingular = true;

	cluster->outerNullSpace = outNullSpace;
	cluster->Rin = RClust;
}

Feti1* createFeti(Mesh *mesh, PetscReal(*f)(Point), PetscReal(*K)(Point),
		MPI_Comm comm) {
	Mat A, B;
	Vec b, lmb;
	NullSpaceInfo nullSpace;
	FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh, A, b, f, K);
	GenerateJumpOperator(mesh, B, lmb);
	Generate2DLaplaceNullSpace(mesh, nullSpace.isDomainSingular, nullSpace.isSubDomainSingular, &(nullSpace.R));

	return new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), comm);
}
