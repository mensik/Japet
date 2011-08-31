#include "feti.h"

GGLinOp::GGLinOp(Mat B, Mat R) {
	this->B = B;
	this->R = R;

	MatGetVecs(B, NULL, &temp2);
	MatGetVecs(R, NULL, &temp1);
}

void GGLinOp::applyMult(Vec in, Vec out, IterationManager *info) {

	MatMult(R, in, temp1);
	MatMult(B, temp1, temp2);
	MatMultTranspose(B, temp2, temp1);
	MatMultTranspose(R, temp1, out);
}

bool GGLinOp::isConverged(PetscInt itNumber, PetscReal norm, PetscReal bNorm,
		Vec *vec) {
	return norm / bNorm < 1e-8;
}

AFeti::AFeti(PDCommManager* comMan, Vec b, Mat BT, Mat B, Vec lmb,
		NullSpaceInfo *nullSpace, CoarseProblemMethod cpM) {

	this->cpMethod = cpM;
	this->cMan = comMan;


	outerSolver = NULL;
	isVerbose = false;

	isSingular = true; /// FIX - only works for total FETI

	this->b = b;
	this->BT = BT;
	this->B = B;

	isLocalSingular = nullSpace->isSubDomainSingular;
	isSingular = nullSpace->isDomainSingular;
	R = nullSpace->R;

	PetscInt lNodeCount;
	VecGetLocalSize(b, &lNodeCount);

	//Priprava ghostovaneho vektoru TEMP a kopie vektoru b do lokalnich casti bloc
	VecCreateGhost(cMan->getPrimal(), lNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &temp);
	VecCopy(b, temp);
	VecGhostGetLocalForm(temp, &tempLoc);

	//Ghostovany vektor reseni
	VecCreateGhost(cMan->getPrimal(), lNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &u);
	VecSet(u, 0);

	PetscInt dualSize;
	MatGetSize(B, &dualSize, PETSC_NULL);
	VecCreateMPI(cMan->getPrimal(), PETSC_DECIDE, dualSize, &pBGlob);
	//VecScatterCreateToZero(pBGlob, &pBScat, &pBLoc);


	this->lmb = lmb;

	//If the matrix A is singular, the matrix G and G'G has to be prepared.
	if (isSingular) initCoarse();

}

void AFeti::initCoarse() {
	//PetscInt rank;
	Mat GTemp, GLOC, GTGloc;

	MyLogger::Instance()->getTimer("Coarse init")->startTimer();
	MyTimer* timer = MyLogger::Instance()->getTimer("Coarse init");


	PetscInt nD;
	MatGetSize(R, PETSC_NULL, &nD);
	VecCreateMPI(cMan->getPrimal(), PETSC_DECIDE, nD, &e);

	//VecCreateGhost(cMan->getPrimal(), PETSC_DECIDE, nD,  0, PETSC_NULL, &eG);

	MatMultTranspose(R, b, e);
	VecScale(e, -1);

	GGLinOp *linOp;
	switch (cpMethod) {
	case ParaCG:

		MatGetVecs(R, &parT2, &parT1);

		VecSet(parT1, 0);
		VecSet(parT2, 0);

		linOp = new GGLinOp(B, R);
		ggParSol = new CGSolver(linOp, parT2, parT2, NULL, cMan->getDual());
		ggParSol->setSolverCtr(linOp);

		VecDuplicate(parT2, &tgA);
		VecDuplicate(tgA, &tgB);

		break;

	case MasterWork:

		if (!cMan->isSameComm()) {
			if (cMan->isPrimal()) {

				MatMatMult(B, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GTemp);

				timer->markTime("BRmult done");

				if (cMan->isDualRoot()) {
					//
					// TODO The roots has to be the same process
					//
					MatGetSize(GTemp, &gM, &gN);

					IS ISrows, IScols;
					ISCreateStride(PETSC_COMM_SELF, gM, 0, 1, &ISrows);
					ISCreateStride(PETSC_COMM_SELF, gN, 0, 1, &IScols);
					Mat *gl;
					MatGetSubMatrices(GTemp, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);

					ISDestroy(ISrows);
					ISDestroy(IScols);

					GLOC = *gl;
					timer->markTime("GLOC obtained");

					MatMatMultTranspose(GLOC, GLOC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GTGloc);

					timer->markTime("GTGloc comp.");

					PC pcGTG;
					PCCreate(PETSC_COMM_SELF, &pcGTG);
					PCSetOperators(pcGTG, GTGloc, GTGloc, SAME_PRECONDITIONER);
					KSPCreate(PETSC_COMM_SELF, &kspG);
					KSPSetOperators(kspG, GTGloc, GTGloc, SAME_PRECONDITIONER);

					MatDestroy(GTGloc);

					PCSetType(pcGTG, "lu");
					PCSetUp(pcGTG);

					KSPSetPC(kspG, pcGTG);
					PCDestroy(pcGTG);

					timer->markTime("Factor.done");

				} else {
					IS ISrows, IScols;
					ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &ISrows);
					ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &IScols);
					Mat *gl;
					MatGetSubMatrices(GTemp, 1, &ISrows, &IScols, MAT_INITIAL_MATRIX, &gl);

					ISDestroy(ISrows);
					ISDestroy(IScols);
				}

				MatDestroy(GTemp);

			}

			if (cMan->isDual()) {

				MPI_Bcast(&gM, 1, MPI_INT, 0, cMan->getDual());
				MPI_Bcast(&gN, 1, MPI_INT, 0, cMan->getDual());

				VecCreateMPI(cMan->getDual(), PETSC_DECIDE, gN, &tgA);
				VecDuplicate(tgA, &tgB);

				VecScatterCreateToZero(tgA, &tgScat, &tgLocIn);
				VecDuplicate(tgLocIn, &tgLocOut);

				timer->markTime("Prep.GforD");
				//MatCreateMPIDense(cMan->getDual(), PETSC_DECIDE, PETSC_DECIDE, gM, gN, PETSC_NULL, &G);
				MatCreateMPIAIJ(cMan->getDual(), PETSC_DECIDE, PETSC_DECIDE, gM, gN, 6, PETSC_NULL, 6, PETSC_NULL, &G);

				if (cMan->isDualRoot()) {

					for (int row = 0; row < gM; row++) {
						PetscInt nCols;
						const PetscInt *idx;
						const PetscScalar *vals;
						MatGetRow(GLOC, row, &nCols, &idx, &vals);
						MatSetValuesBlocked(G, 1, &row, nCols, idx, vals, INSERT_VALUES);
						MatRestoreRow(GLOC, row, &nCols, &idx, &vals);
					}

					MatDestroy(GLOC);
				}

				MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);
				MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);

				timer->markTime("G in dual done");
			}
		} else {
			//
			//
			//  Variant for identical comunicators
			//
			//


			MatMatMult(B, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &G);

			MatGetLocalMat(G, MAT_INITIAL_MATRIX, &GTemp);

			PetscScalar *val;
			PetscInt *ia, *ja;
			PetscInt n;
			PetscTruth done;
			PetscInt lm, ln;

			MatGetSize(GTemp, &lm, &ln);
			MatGetArray(GTemp, &val);
			MatGetRowIJ(GTemp, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done);

			if (cMan->isPrimalRoot()) {

				PetscInt lNumRow[cMan->getPrimalSize()], lNNZ[cMan->getPrimalSize()],
						firstRowIndex[cMan->getPrimalSize() + 1],
						displ[cMan->getPrimalSize()], totalNNZ, totalRows;
				MPI_Gather(&n, 1, MPI_INT, lNumRow, 1, MPI_INT, 0, cMan->getPrimal());
				MPI_Gather(&ia[n], 1, MPI_INT, lNNZ, 1, MPI_INT, 0, cMan->getPrimal());

				totalNNZ = 0;
				totalRows = 0;

				for (int i = 0; i < cMan->getPrimalSize(); i++) {
					firstRowIndex[i] = totalRows;
					displ[i] = totalNNZ;
					totalNNZ += lNNZ[i];
					totalRows += lNumRow[i];
				}
				firstRowIndex[cMan->getPrimalSize()] = totalRows;

				PetscInt *locJA = new PetscInt[totalNNZ];
				PetscInt *locIA = new PetscInt[totalRows + 1];
				PetscScalar *locVal = new PetscScalar[totalNNZ];

				locIA[0] = 0;
				MPI_Gatherv(ja, ia[n], MPI_INT, locJA, lNNZ, displ, MPI_INT, 0, cMan->getPrimal());
				MPI_Gatherv(ia + 1, n, MPI_INT, locIA + 1, lNumRow, firstRowIndex, MPI_INT, 0, cMan->getPrimal());
				MPI_Gatherv(val, ia[n], MPI_DOUBLE, locVal, lNNZ, displ, MPI_DOUBLE, 0, cMan->getPrimal());

				for (int j = 0; j < cMan->getPrimalSize(); j++) {
					for (int i = firstRowIndex[j] + 1; i < firstRowIndex[j + 1] + 1; i++) {
						locIA[i] += displ[j];
					}
				}

				MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, totalRows, ln, locIA, locJA, locVal, &GLOC);

				MatMatMultTranspose(GLOC, GLOC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GTGloc);

				MatDestroy(GLOC);
				delete[] locJA;
				delete[] locIA;
				delete[] locVal;

				PC pcGTG;
				PCCreate(PETSC_COMM_SELF, &pcGTG);
				PCSetOperators(pcGTG, GTGloc, GTGloc, SAME_PRECONDITIONER);
				KSPCreate(PETSC_COMM_SELF, &kspG);
				KSPSetOperators(kspG, GTGloc, GTGloc, SAME_PRECONDITIONER);

				MatDestroy(GTGloc);

				PCSetType(pcGTG, "lu");

				PCSetUp(pcGTG);

				KSPSetPC(kspG, pcGTG);
				PCDestroy(pcGTG);

			} else {
				MPI_Gather(&n, 1, MPI_INT, PETSC_NULL, 1, MPI_INT, 0, cMan->getPrimal());
				MPI_Gather(&ia[n], 1, MPI_INT, PETSC_NULL, 1, MPI_INT, 0, cMan->getPrimal());

				MPI_Gatherv(ja, ia[n], MPI_INT, NULL, NULL, NULL, MPI_INT, 0, cMan->getPrimal());
				MPI_Gatherv(ia + 1, n, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, cMan->getPrimal());
				MPI_Gatherv(val, ia[n], MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, cMan->getPrimal());
			}

			MatRestoreRowIJ(GTemp, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done);
			MatRestoreArray(GTemp, &val);

			MatDestroy(GTemp);
			MatGetSize(G, &gM, &gN);

			VecCreateMPI(cMan->getDual(), PETSC_DECIDE, gN, &tgA);
			VecDuplicate(tgA, &tgB);

			VecScatterCreateToZero(tgA, &tgScat, &tgLocIn);
			VecDuplicate(tgLocIn, &tgLocOut);

		}
		break;
	case ORTO:
		//
		//So far only prepared for same comm
		//
		if (cMan->isSameComm()) {

			PetscInt primSize;
			MatGetSize(R, &primSize, &gN);
			MatGetSize(B, &gM, PETSC_NULL);

			Vec rTemp;
			Vec gCols[gN];

			VecDuplicate(b, &rTemp);

			for (int i = 0; i < gN; i++) {
				MatGetColumnVector(R, rTemp, i);
				VecCreateMPI(cMan->getDual(), PETSC_DECIDE, gM, &gCols[i]);
				MatMult(B, rTemp, gCols[i]);
			}
			VecDestroy(rTemp);

			VecScatter eScat;
			Vec eMaster;
			VecScatterCreateToZero(e, &eScat, &eMaster);

			VecScatterBegin(eScat, e, eMaster, INSERT_VALUES, SCATTER_FORWARD);
			VecScatterEnd(eScat, e, eMaster, INSERT_VALUES, SCATTER_FORWARD);

			PetscScalar *eValues;
			if (cMan->isDualRoot()) {

				VecGetArray(eMaster, &eValues);
			}
			//
			// Orthogonalization
			//

			// Gram-Schmidt

			PetscReal r;
			VecNorm(gCols[0], NORM_2, &r);
			VecScale(gCols[0], 1 / r);

			if (cMan->isDualRoot()) {
				eValues[0] = eValues[0] / r;
			}

			for (int j = 1; j < gN; j++) {

				PetscScalar eDif = 0;

				for (int i = 0; i < j; i++) {
					VecDot(gCols[i], gCols[j], &r);
					VecAXPY(gCols[j], -r, gCols[i]);

					if (cMan->isDualRoot()) {
						eDif += -r * eValues[i];
					}
				}

				VecNorm(gCols[j], NORM_2, &r);
				VecScale(gCols[j], 1 / r);

				if (cMan->isDualRoot()) {
					eValues[j] = (eValues[j] + eDif) / r;
				}
			}

			if (cMan->isDualRoot()) {
				VecRestoreArray(eMaster, &eValues);
			}

			VecScatterBegin(eScat, eMaster, e, INSERT_VALUES, SCATTER_REVERSE);
			VecScatterEnd(eScat, eMaster, e, INSERT_VALUES, SCATTER_REVERSE);

			VecScatterDestroy(eScat);
			if (cMan->isDualRoot()) VecDestroy(eMaster);

			PetscInt localRows, firstRow;
			VecGetLocalSize(gCols[0], &localRows);
			VecGetOwnershipRange(gCols[0], &firstRow, PETSC_NULL);

			//MatCreateMPIAIJ(cMan->getDual(), localRows, PETSC_DECIDE, PETSC_DECIDE, gN, gN, PETSC_NULL, 0, PETSC_NULL, &G);
			MatCreateMPIDense(cMan->getDual(), localRows, PETSC_DECIDE, PETSC_DECIDE, gN, PETSC_NULL, &G);

			PetscInt rowIndices[localRows];
			PetscScalar *val;
			for (int i = 0; i < localRows; i++) {
				rowIndices[i] = i + firstRow;
			}

			for (int i = 0; i < gN; i++) {
				VecGetArray(gCols[i], &val);
				MatSetValues(G, localRows, rowIndices, 1, &i, val, INSERT_VALUES);
				VecRestoreArray(gCols[i], &val);
				VecDestroy(gCols[i]);
			}
			MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);
			MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);

			//VecView(e, PETSC_VIEWER_STDOUT_WORLD);


			VecCreateMPI(cMan->getDual(), PETSC_DECIDE, gN, &tgA);
			VecDuplicate(tgA, &tgB);

			//VecScatterCreateToZero(tgA, &tgScat, &tgLocIn);
			//VecDuplicate(tgLocIn, &tgLocOut);

		}
		break;
	default:
		break;
	}
	MyLogger::Instance()->getTimer("Coarse init")->stopTimer();


}

void AFeti::applyInvGTG(Vec in, Vec out) {

	switch (cpMethod) {
	case ParaCG:

		ggParSol->reset(in, out);
		ggParSol->solve();
		ggParSol->getX(out);

		break;
	case MasterWork:
		VecScatterBegin(tgScat, in, tgLocIn, INSERT_VALUES, SCATTER_FORWARD);
		VecScatterEnd(tgScat, in, tgLocIn, INSERT_VALUES, SCATTER_FORWARD);

		if (cMan->isDualRoot()) {
			KSPSetTolerances(kspG, 1e-19, 1e-19, 1e7, 2);
			KSPSolve(kspG, tgLocIn, tgLocOut);
		}

		VecScatterBegin(tgScat, tgLocOut, out, INSERT_VALUES, SCATTER_REVERSE);
		VecScatterEnd(tgScat, tgLocOut, out, INSERT_VALUES, SCATTER_REVERSE);
		break;
	case ORTO:
		VecCopy(in, out);
		break;
	}

}

void AFeti::projectGOrth(Vec in) {

	MyLogger::Instance()->getTimer("Coarse problem")->startTimer();

	if (cpMethod != ParaCG) {
		MatMultTranspose(G, in, tgA);
	} else {

		MatMultTranspose(B, in, parT1);
		MatMultTranspose(R, parT1, tgA);

	}

	applyInvGTG(tgA, tgB);

	VecScale(in, -1);

	if (cpMethod != ParaCG) {
		MatMultAdd(G, tgB, in, in);
	} else {
		MatMult(R, tgB, parT1);
		MatMultAdd(B, parT1, in, in);
	}
	VecScale(in, -1);
	MyLogger::Instance()->getTimer("Coarse problem")->stopTimer();
}

AFeti::~AFeti() {

	if (cMan->isPrimal()) {
		VecDestroy(u);
		VecDestroy(temp);
		VecDestroy(tempLoc);

		if (isSingular) {
			MatDestroy(R);
		}

		VecScatterDestroy(pBScat);
	}

	if (cMan->isDual()) {
		MatDestroy(G);
		VecDestroy(lmb);
		if (cMan->isDualRoot()) {
			KSPDestroy(kspG);
		}

		VecScatterDestroy(dBScat);
		VecScatterDestroy(tgScat);

		VecDestroy(tgA);
		VecDestroy(tgB);

		VecDestroy(tgLocIn);
		VecDestroy(tgLocOut);
	}

	if (outerSolver != NULL) delete outerSolver;

}

void AFeti::dumpSolution(PetscViewer v) {
	//VecView(u, v);
	//VecView(lmb, v);
}

void AFeti::dumpSystem(PetscViewer v) {
	//View A
	//VecView(b, v);
	//MatView(B, v);
}

void AFeti::applyMult(Vec in, Vec out, IterationManager *info) {

	outIterations++;

	if (cMan->isPrimalRoot()) MyLogger::Instance()->getTimer("BA+BT")->startTimer();

	MatMult(BT, in, temp);
	applyInvA(temp, info);
	MatMult(B, temp, out);

	if (cMan->isPrimalRoot()) MyLogger::Instance()->getTimer("BA+BT")->stopTimer();

}

void AFeti::applyPrimalMult(Vec in, Vec out) {
	VecCopy(in, out);
}

Solver* AFeti::instanceOuterSolver(Vec d, Vec l) {
	return new CGSolver(this, d, l, this, cMan->getDual());
}

void AFeti::solve() {

	inIterations = 0;
	outIterations = 0;

	Vec d;
	//	VecScatter peScat, deScat;

	VecDuplicate(lmb, &d);

	if (cpMethod != ParaCG) MatScale(G, -1);


	VecCopy(b, temp);
	//applyInvA(temp, NULL);
	//MatMult(B, temp, d);

	//Preparation of the right-hand side vector d=PBA^+b
	//The matrix P is projector on the space orthogonal to range(G)

	//if (isSingular) projectGOrth(d); //Projection


	//Feasible lambda_0 preparation

	Vec lmbIm, lmbKer, dAlt;

	if (isSingular) { //Je li singularni, je treba pripavit vhodne vstupni lambda

		Vec eTemp;
		VecDuplicate(e, &eTemp);
		applyInvGTG(e, eTemp);

		if (cpMethod != ParaCG) {
			MatMult(G, eTemp, lmb);
		} else {
			MatMult(R, eTemp, parT1);
			MatMult(B, parT1, lmb);
			VecScale(lmb, -1);
		}

		Vec lmbTemp;
		VecDuplicate(lmb, &lmbKer);
		VecDuplicate(d, &dAlt);
		applyMult(lmb, lmbKer, NULL);

		VecCopy(d, dAlt);
		VecAXPY(dAlt, -1, lmbKer);

		VecSet(lmbKer, 0);
		projectGOrth(dAlt);
	}

	outerSolver = instanceOuterSolver(dAlt, lmbKer);
	outerSolver->setSolverCtr(this);
	outerSolver->setIsVerbose(isVerbose);

	//Solve!!!
	outerSolver->solve();
	outerSolver->getX(lmbKer);

	projectGOrth(lmbKer);

	VecAXPY(lmb, 1, lmbKer);

	//
	// Rigid body motions
	//
	if (isSingular) {

		Vec lmbTemp;
		VecDuplicate(lmb, &lmbTemp);
		applyMult(lmb, lmbTemp, NULL);

		VecAYPX(lmbTemp, -1, d);

		Vec bAlp, alpha;

		VecCreateMPI(cMan->getDual(), PETSC_DECIDE, gN, &bAlp);
		VecDuplicate(bAlp, &alpha);

		if (cpMethod == ORTO) {
			Vec tA1, tA2;

			MatGetVecs(R, &tA2, &tA1);
			MatMultTranspose(B, lmbTemp, tA1);
			MatMultTranspose(R, tA1, tA2);

			GGLinOp * linOp = new GGLinOp(B, R);
			CGSolver *ggCgSolver =
					new CGSolver(linOp, tA2, bAlp, NULL, cMan->getDual());
			ggCgSolver->setSolverCtr(linOp);

			ggCgSolver->solve();

			ggCgSolver->getX(alpha);
			VecScale(alpha, -1);

			//MatMultTranspose(G, lmbTemp, bAlp);
		} else if (cpMethod == ParaCG) {

			MatMultTranspose(B, lmbTemp, parT1);
			MatMultTranspose(R, parT1, bAlp);

			VecScale(bAlp, -1);
			applyInvGTG(bAlp, alpha);

		} else {
			MatMultTranspose(G, lmbTemp, bAlp);
			applyInvGTG(bAlp, alpha);
		}

		VecCopy(b, u);

		VecScale(lmb, -1);
		MatMultAdd(BT, lmb, u, u);
		VecScale(lmb, -1);
		applyInvA(u, NULL);
		MatMultAdd(R, alpha, u, u);

	}

	//VecScale(lmb, -1);
	//VecDestroy(d);

	if (isVerbose) {
		PetscReal feasErr, uNorm;

		MatMult(B, u, d);

		VecNorm(d, NORM_2, &feasErr);
		VecNorm(u, NORM_2, &uNorm);

		PetscPrintf(cMan->getPrimal(), "\n");
		PetscPrintf(cMan->getPrimal(), "FETI finished   Outer it: %d   Inner it: %d\n", outIterations, inIterations);
		PetscPrintf(cMan->getPrimal(), "Feasibility err: %e \n", feasErr / uNorm);
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

	return norm / bNorm < 1e-4 || itNumber > 60;
}

Feti1::Feti1(PDCommManager *comMan, Mat A, Vec b, Mat BT, Mat B, Vec lmb,
		NullSpaceInfo *nullSpace, PetscInt localNodeCount, PetscInt fNodesCount,
		PetscInt *fNodes, CoarseProblemMethod cpM) :
	AFeti(comMan, b, BT, B, lmb, nullSpace, cpM) {

	if (cMan->isPrimal()) {

		PetscInt lNodeCount;
		VecGetLocalSize(b, &lNodeCount);

		//Sestaveni Nuloveho prostoru lokalni casti matice tuhosti A
		if (isLocalSingular) {
			MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, nullSpace->localDimension, nullSpace->localBasis, &locNS);
		}

		this->A = A;

		Aloc = A;
		//extractLocalAPart(A, &Aloc);
		//Matrix regularization!

		PetscInt firstRow, lastRow, nCols, locNullDim, nodeDim;

		VecGetOwnershipRange(b, &firstRow, &lastRow);
		MatGetSize(R, PETSC_NULL, &nCols);

		locNullDim = nCols / cMan->getPrimalSize();
		nodeDim = (lastRow - firstRow) / localNodeCount; //Number of rows for each node

		PetscInt FIX_NODE_COUNT;
		PetscInt fixingNodes[5];

		if (fNodesCount == 0) {
			if (nodeDim == 1) { //Laplace
				FIX_NODE_COUNT = 1;
				fixingNodes[0] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2;
			} else if (nodeDim == 2) { //Elasticity
				FIX_NODE_COUNT = 3;
				fixingNodes[0] = firstRow / nodeDim + 4;
				fixingNodes[1] = lastRow / nodeDim - 4;
				fixingNodes[2] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2 + 5;
				fixingNodes[3] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2 - 5;
				fixingNodes[4] = (firstRow / nodeDim + lastRow / nodeDim - 1) / 2;
			}
		} else {
			for (int i = 0; i < fNodesCount; i++) {
				fixingNodes[i] = fNodes[i] + firstRow / nodeDim;
			}
		}

		Mat REG;
		MatCreateSeqAIJ(PETSC_COMM_SELF, lastRow - firstRow, locNullDim, FIX_NODE_COUNT
				* nodeDim * locNullDim, PETSC_NULL, &REG);

		PetscReal values[locNullDim];
		PetscInt idx[locNullDim];
		for (int i = 0; i < locNullDim; i++) {
			idx[i] = cMan->getPrimalRank() * locNullDim + i;
		}

		for (int i = 0; i < FIX_NODE_COUNT; i++)
			for (int d = 0; d < nodeDim; d++) {

				PetscInt colInd = fixingNodes[i] * nodeDim + d;
				MatGetValues(R, 1, &colInd, locNullDim, idx, values);

				for (int j = 0; j < locNullDim; j++) {
					MatSetValue(REG, colInd - firstRow, j, values[j], INSERT_VALUES);
				}

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

		MyLogger::Instance()->getTimer("Factorization")->startTimer();
		PCCreate(PETSC_COMM_SELF, &pc);
		PCSetOperators(pc, Areg, Areg, SAME_PRECONDITIONER);
		PCSetFromOptions(pc);
		PCSetUp(pc);

		KSPCreate(PETSC_COMM_SELF, &kspA);
		KSPSetTolerances(kspA, 1e-10, 1e-10, 1e7, 1);
		KSPSetPC(kspA, pc);
		KSPSetOperators(kspA, Areg, Areg, SAME_PRECONDITIONER);

		MyLogger::Instance()->getTimer("Factorization")->stopTimer();

		MatDestroy(Areg);
		MatDestroy(REGREGT);
		MatDestroy(REGT);

		PCDestroy(pc);

		if (isLocalSingular) KSPSetNullSpace(kspA, locNS);

		VecCreateGhost(cMan->getPrimal(), lNodeCount, PETSC_DECIDE, 0, PETSC_NULL, &tempInv);
		VecSet(tempInv, 0);
		VecGhostGetLocalForm(tempInv, &tempInvGh);
		VecDuplicate(tempInvGh, &tempInvGhB);

	}

}

void Feti1::solve() {

	AFeti::solve();

	PetscReal normB, error;

	if (isVerbose) {

		VecNorm(b, NORM_2, &normB);

		applyPrimalMult(u, temp);
		VecAXPY(temp, -1, b);
		MatMult(BT, lmb, tempInv);
		VecAXPY(temp, 1, tempInv);

		VecNorm(temp, NORM_2, &error);

		PetscPrintf(cMan->getPrimal(), "Relative error: %e\n\n", error / normB);

	}

}

Feti1::~Feti1() {

	if (cMan->isPrimal()) {
		KSPDestroy(kspA);
		MatDestroy(Aloc);
		if (isLocalSingular) {
			MatNullSpaceDestroy(locNS);
		}
		VecDestroy(tempInv);
		VecDestroy(tempInvGh);
		VecDestroy(tempInvGhB);
	}

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

	if (cMan->isPrimalRoot()) MyLogger::Instance()->getTimer("F^-1")->startTimer();

	if (isSingular) projectGOrth(g);

	MatMult(BT, g, temp);
	applyPrimalMult(temp, temp);
	MatMult(B, temp, z);

	if (isSingular) projectGOrth(z);

	if (cMan->isPrimalRoot()) MyLogger::Instance()->getTimer("F^-1")->stopTimer();

}

void Feti1::applyPrimalMult(Vec in, Vec out) {

	VecCopy(in, tempInv);

	MatMult(Aloc, tempInvGh, tempInvGhB);
	VecCopy(tempInvGhB, tempInvGh);

	VecCopy(tempInv, out);

}
/*
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
 */
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

HFeti::HFeti(PDCommManager* pdMan, Mat A, Vec b, Mat BGlob, Mat BTGlob,
		Mat BClust, Mat BTClust, Vec lmbGl, Vec lmbCl, SubdomainCluster *cluster,
		PetscInt localNodeCount) :
	AFeti(pdMan, b, BTGlob, BGlob, lmbGl, cluster->outerNullSpace, MasterWork) {

	VecCreateGhost(cMan->getParen(), localNodeCount * 2, PETSC_DECIDE, 0, PETSC_NULL, &globTemp);
	VecSet(globTemp, 0);
	VecGhostGetLocalForm(globTemp, &globTempGh);

	VecCreateGhost(cluster->clusterComm, localNodeCount * 2, PETSC_DECIDE, 0, PETSC_NULL, &clustTemp);
	VecSet(clustTemp, 0);
	VecGhostGetLocalForm(clustTemp, &clustTempGh);

	VecCreateMPI(cluster->clusterComm, localNodeCount * 2, PETSC_DECIDE, &clustb);

	VecCopy(b, globTemp);
	VecCopy(globTempGh, clustTempGh);
	VecCopy(clustTemp, clustb);

	PDCommManager *clustComMan =
			new PDCommManager(cluster->clusterComm, SAME_COMMS);

	//
	// TODO GTG ve vnitrnim feti neni regularni!!!
	//
	subClusterSystem
			= new Feti1(clustComMan, A, clustb, BTClust, BClust, lmbCl, cluster->clusterNullSpace, localNodeCount, 0, NULL, MasterWork);
	subClusterSystem->setSystemSingular();

	//Sestaveni Nuloveho prostoru lokalni casti matice tuhosti A
	//if (isLocalSingular) {
	//	MatNullSpaceCreate(cluster->clusterComm, PETSC_TRUE, 0, PETSC_NULL, &clusterNS);
	//}

}

HFeti::~HFeti() {
	delete subClusterSystem;
}

void HFeti::applyInvA(Vec in, IterationManager *itManager) {

	VecCopy(in, globTemp);
	VecCopy(globTempGh, clustTempGh);
//	if (isLocalSingular) MatNullSpaceRemove(clusterNS, clustTemp, PETSC_NULL);
//	subClusterSystem->setIsVerbose(true);
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

//
//
//
// ******************************************************************************
//
//


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

void GenerateTotalJumpOperator(Mesh *mesh, int d, Mat &B, Mat &BT, Vec &lmb,
		PDCommManager* commManager) {

	int dSum;

	if (commManager->isPrimal()) {

		//
		// Compute overall primal size
		//
		PetscInt localNodeCount, globalNodeCount;
		localNodeCount = mesh->vetrices.size();
		MPI_Reduce(&localNodeCount, &globalNodeCount, 1, MPI_INT, MPI_SUM, 0, commManager->getPrimal());

		//
		//Put all dirchlet nodes to root
		//
		int dSize;
		std::set<PetscInt> indDirchlet;
		for (std::set<PetscInt>::iterator i = mesh->borderEdges.begin(); i
				!= mesh->borderEdges.end(); i++) {

			for (int j = 0; j < 2; j++) {
				indDirchlet.insert(mesh->edges[*i]->vetrices[j]);
			}
		}
		dSize = indDirchlet.size();

		PetscInt locDirch[dSize];
		int counter = 0;
		for (std::set<PetscInt>::iterator i = indDirchlet.begin(); i
				!= indDirchlet.end(); i++) {
			locDirch[counter++] = *i;
		}

		int dNodeCounts[commManager->getPrimalSize()];

		MPI_Allgather(&dSize, 1, MPI_INT, dNodeCounts, 1, MPI_INT, commManager->getPrimal());

		dSum = 0;
		for (int i = 0; i < commManager->getPrimalSize(); i++)
			dSum += dNodeCounts[i];

		//
		// Create BT matrix on PRIMAL
		//
		MatCreateMPIAIJ(commManager->getPrimal(), mesh->vetrices.size() * d, PETSC_DECIDE, PETSC_DECIDE, (mesh->nPairs
				+ dSum) * d, 2, PETSC_NULL, 2, PETSC_NULL, &BT);
		MatCreateMPIAIJ(commManager->getPrimal(), PETSC_DECIDE, mesh->vetrices.size()
				* d, (mesh->nPairs + dSum) * d, PETSC_DECIDE, 2, PETSC_NULL, 2, PETSC_NULL, &B);

		if (commManager->isPrimalRoot()) {
			//
			// TODO I suppose here, that pRank is in root in dual too!
			//

			PetscInt globDirch[dSum];
			int displac[commManager->getPrimalSize()];
			displac[0] = 0;
			for (int i = 1; i < commManager->getPrimalSize(); i++)
				displac[i] = displac[i - 1] + dNodeCounts[i - 1];

			MPI_Gatherv(locDirch, dSize, MPI_INT, globDirch, dNodeCounts, displac, MPI_INT, 0, commManager->getPrimal());

			int sIndex = 0;
			for (int i = 0; i < dSum; i++) {
				for (int j = 0; j < d; j++) {
					MatSetValue(B, sIndex, globDirch[i] * d + j, 1, INSERT_VALUES);
					MatSetValue(BT, globDirch[i] * d + j, sIndex, 1, INSERT_VALUES);
					sIndex++;
				}
			}

			PetscReal boundVal = 1.0 / sqrt(2.0); // Value to keep the B ortonormal


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

						MatSetValue(B, rowCounter * d + j, mesh->pointPairing[i * 2] * d
								+ j, boundVal, INSERT_VALUES);
						MatSetValue(B, rowCounter * d + j, mesh->pointPairing[i * 2 + 1]
								* d + j, -boundVal, INSERT_VALUES);

						MatSetValue(BT, mesh->pointPairing[i * 2] * d + j, rowCounter * d
								+ j, boundVal, INSERT_VALUES);
						MatSetValue(BT, mesh->pointPairing[i * 2 + 1] * d + j, rowCounter
								* d + j, -boundVal, INSERT_VALUES);

					}
					rowCounter++;
				}
			}

			for (int i = 0; i < mesh->corners.size(); i++) {

				PetscInt *vetrices = mesh->corners[i]->vetrices;
				PetscInt cornerSize = mesh->corners[i]->cornerSize;

				for (int j = 0; j < cornerSize - 1; j++) {

					PetscReal norm = sqrt((PetscReal) (cornerSize - j - 1) * (cornerSize
							- j - 1) + (cornerSize - j - 1));

					for (int dim = 0; dim < d; dim++) {
						MatSetValue(B, rowCounter * d + dim, vetrices[j] * d + dim, -(cornerSize
								- j - 1) / norm, INSERT_VALUES);
						MatSetValue(BT, vetrices[j] * d + dim, rowCounter * d + dim, -(cornerSize
								- j - 1) / norm, INSERT_VALUES);

						for (int k = j + 1; k < cornerSize; k++) {
							MatSetValue(B, rowCounter * d + dim, vetrices[k] * d + dim, 1
									/ norm, INSERT_VALUES);
							MatSetValue(BT, vetrices[k] * d + dim, rowCounter * d + dim, 1
									/ norm, INSERT_VALUES);
						}
					}

					rowCounter++;
				}
			}

		} else {
			//
			//Primal nonroots
			//
			MPI_Gatherv(locDirch, dSize, MPI_INT, NULL, 0, NULL, MPI_INT, 0, commManager->getPrimal());
		}

		MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
		MatAssemblyBegin(BT, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(BT, MAT_FINAL_ASSEMBLY);

	}

	if (commManager->isDual()) {
		int BSize;

		if (commManager->isDualRoot()) {
			BSize = (mesh->nPairs + dSum) * d;
		}

		MPI_Bcast(&BSize, 1, MPI_INT, 0, commManager->getDual());

		VecCreateMPI(commManager->getDual(), PETSC_DECIDE, BSize, &lmb);
		VecSet(lmb, 0);
	}

}

void GenerateClusterJumpOperator(Mesh *mesh, SubdomainCluster *cluster,
		Mat &BGlob, Mat &BTGlob, Vec &lmbGlob, Mat &BCluster, Mat &BTCluster,
		Vec &lmbCluster) {

	PetscInt rank, subRank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_rank(cluster->clusterComm, &subRank);

	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	int d = 2;
	//
	//Put all dirchlet nodes to root
	//

	int dSize;
	std::set<PetscInt> indDirchlet;
	for (std::set<PetscInt>::iterator i = mesh->borderEdges.begin(); i
			!= mesh->borderEdges.end(); i++) {

		for (int j = 0; j < 2; j++) {
			indDirchlet.insert(mesh->edges[*i]->vetrices[j]);
		}

	}
	dSize = indDirchlet.size();

	PetscInt locDirch[dSize];
	int counter = 0;
	for (std::set<PetscInt>::iterator i = indDirchlet.begin(); i
			!= indDirchlet.end(); i++) {
		locDirch[counter++] = *i;
	}
	int dNodeCounts[size];
	MPI_Allgather(&dSize, 1, MPI_INT, dNodeCounts, 1, MPI_INT, PETSC_COMM_WORLD);

	int dSum = 0;
	for (int i = 0; i < size; i++)
		dSum += dNodeCounts[i];

	int globalPairsCount = cluster->globalPairing.size() / 2;
	MPI_Bcast(&globalPairsCount, 1, MPI_INT, 0, PETSC_COMM_WORLD);

	MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, mesh->vetrices.size() * d, (globalPairsCount
			+ dSum) * d, PETSC_DECIDE, 2, PETSC_NULL, 2, PETSC_NULL, &BGlob);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, mesh->vetrices.size() * d, PETSC_DECIDE, PETSC_DECIDE, (globalPairsCount
			+ dSum) * d, 2, PETSC_NULL, 2, PETSC_NULL, &BTGlob);

	if (!rank) {

		PetscInt globDirch[dSum];
		int displac[size];
		displac[0] = 0;
		for (int i = 1; i < size; i++)
			displac[i] = displac[i - 1] + dNodeCounts[i - 1];

		MPI_Gatherv(locDirch, dSize, MPI_INT, globDirch, dNodeCounts, displac, MPI_INT, 0, PETSC_COMM_WORLD);

		int sIndex = 0;
		for (int i = 0; i < dSum; i++) {
			for (int j = 0; j < d; j++) {
				MatSetValue(BGlob, sIndex, globDirch[i] * d + j, 1, INSERT_VALUES);
				MatSetValue(BTGlob, globDirch[i] * d + j, sIndex, 1, INSERT_VALUES);
				sIndex++;
			}
		}

		typedef std::map<PetscInt, std::set<PetscInt> > PGroupMap;
		PGroupMap equalGroups;

		for (std::vector<PetscInt>::iterator i = cluster->globalPairing.begin(); i
				!= cluster->globalPairing.end(); i = i + 2) {

			std::set<PetscInt> newSet;

			PetscInt pair[] = { *i, *(i + 1) };

			for (int j = 0; j < 2; j++) {
				newSet.insert(pair[j]);

				for (std::set<PetscInt>::iterator k = equalGroups[pair[j]].begin(); k
						!= equalGroups[pair[j]].end(); k++) {
					newSet.insert(*k);
				}
			}

			for (std::set<PetscInt>::iterator j = newSet.begin(); j != newSet.end(); j++) {
				for (std::set<PetscInt>::iterator k = newSet.begin(); k != newSet.end(); k++) {
					equalGroups[*j].insert(*k);
				}
			}
		}

		// for (PGroupMap::iterator i = equalGroups.begin(); i != equalGroups.end(); i++) {
		// PetscPrintf(PETSC_COMM_SELF, "[%d] - ", i->first);
		// for (std::set<PetscInt>::iterator j = i->second.begin(); j
		// != i->second.end(); j++) {
		// PetscPrintf(PETSC_COMM_SELF, " %d ", *j);
		// }
		// PetscPrintf(PETSC_COMM_SELF, "\n");
		// }


		PetscInt rowCounter = dSum;

		while (equalGroups.size() > 0) {

			std::set<PetscInt> p = equalGroups.begin()->second;

			PetscInt cornerSize = p.size();
			PetscInt *vetrices = new PetscInt[cornerSize];

			int cc = 0;
			for (std::set<PetscInt>::iterator j = p.begin(); j != p.end(); j++) {
				vetrices[cc++] = *j;
				equalGroups.erase(*j);
			}

			for (int j = 0; j < cornerSize - 1; j++) {

				PetscReal norm = sqrt((PetscReal) (cornerSize - j - 1) * (cornerSize
						- j - 1) + (cornerSize - j - 1));

				for (int dim = 0; dim < d; dim++) {
					MatSetValue(BGlob, rowCounter * d + dim, vetrices[j] * d + dim, -(cornerSize
							- j - 1) / norm, INSERT_VALUES);
					MatSetValue(BTGlob, vetrices[j] * d + dim, rowCounter * d + dim, -(cornerSize
							- j - 1) / norm, INSERT_VALUES);

					for (int k = j + 1; k < cornerSize; k++) {
						MatSetValue(BGlob, rowCounter * d + dim, vetrices[k] * d + dim, 1
								/ norm, INSERT_VALUES);
						MatSetValue(BTGlob, vetrices[k] * d + dim, rowCounter * d + dim, 1
								/ norm, INSERT_VALUES);
					}
				}
				rowCounter++;
			}
			delete[] vetrices;
		}
	} else {
		MPI_Gatherv(locDirch, dSize, MPI_INT, NULL, 0, NULL, MPI_INT, 0, PETSC_COMM_WORLD);
	}

	MatAssemblyBegin(BGlob, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(BGlob, MAT_FINAL_ASSEMBLY);

	MatAssemblyBegin(BTGlob, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(BTGlob, MAT_FINAL_ASSEMBLY);

	VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, (globalPairsCount + dSum) * d, &lmbGlob);
	VecSet(lmbGlob, 0);

	int clusterPairCount = cluster->localPairing.size() / 2;
	MPI_Bcast(&clusterPairCount, 1, MPI_INT, 0, cluster->clusterComm);

	MatCreateMPIAIJ(cluster->clusterComm, PETSC_DECIDE, mesh->vetrices.size() * d, clusterPairCount
			* d, PETSC_DECIDE, 2, PETSC_NULL, 2, PETSC_NULL, &BCluster);
	MatCreateMPIAIJ(cluster->clusterComm, mesh->vetrices.size() * d, PETSC_DECIDE, PETSC_DECIDE, clusterPairCount
			* d, 2, PETSC_NULL, 2, PETSC_NULL, &BTCluster);

	VecCreateMPI(cluster->clusterComm, PETSC_DECIDE, clusterPairCount, &lmbCluster);

	if (!subRank) {

		typedef std::map<PetscInt, std::set<PetscInt> > PGroupMap;
		PGroupMap equalGroups;

		for (std::vector<PetscInt>::iterator i = cluster->localPairing.begin(); i
				!= cluster->localPairing.end(); i = i + 2) {

			std::set<PetscInt> newSet;

			PetscInt pair[] = { *i, *(i + 1) };

			for (int j = 0; j < 2; j++) {
				newSet.insert(pair[j]);

				for (std::set<PetscInt>::iterator k = equalGroups[pair[j]].begin(); k
						!= equalGroups[pair[j]].end(); k++) {
					newSet.insert(*k);
				}
			}

			for (std::set<PetscInt>::iterator j = newSet.begin(); j != newSet.end(); j++) {
				for (std::set<PetscInt>::iterator k = newSet.begin(); k != newSet.end(); k++) {
					equalGroups[*j].insert(*k);
				}
			}
		}

		PetscInt rowCounter = 0;
		while (equalGroups.size() > 0) {

			std::set<PetscInt> p = equalGroups.begin()->second;

			PetscInt cornerSize = p.size();
			PetscInt *vetrices = new PetscInt[cornerSize];

			int cc = 0;
			for (std::set<PetscInt>::iterator j = p.begin(); j != p.end(); j++) {

				PetscInt globalIndex = *j;
				PetscInt clusterIndex = globalIndex
						+ cluster->startIndexesDiff[mesh->getNodeDomain(globalIndex)];
				vetrices[cc++] = clusterIndex;

				equalGroups.erase(*j);
			}

			for (int j = 0; j < cornerSize - 1; j++) {

				PetscReal norm = sqrt((PetscReal) (cornerSize - j - 1) * (cornerSize
						- j - 1) + (cornerSize - j - 1));

				for (int dim = 0; dim < d; dim++) {

					MatSetValue(BCluster, rowCounter * d + dim, vetrices[j] * d + dim, -(cornerSize
							- j - 1) / norm, INSERT_VALUES);
					MatSetValue(BTCluster, vetrices[j] * d + dim, rowCounter * d + dim, -(cornerSize
							- j - 1) / norm, INSERT_VALUES);

					for (int k = j + 1; k < cornerSize; k++) {
						MatSetValue(BCluster, rowCounter * d + dim, vetrices[k] * d + dim, 1
								/ norm, INSERT_VALUES);
						MatSetValue(BTCluster, vetrices[k] * d + dim, rowCounter * d + dim, 1
								/ norm, INSERT_VALUES);
					}
				}

				rowCounter++;
			}

			delete[] vetrices;
		}
	}

	MatAssemblyBegin(BCluster, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(BCluster, MAT_FINAL_ASSEMBLY);

	MatAssemblyBegin(BTCluster, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(BTCluster, MAT_FINAL_ASSEMBLY);

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

	PetscInt rowIndG[mesh->vetrices.size() * 2];

	PetscInt localRSize = mesh->vetrices.size() * 2;

	for (int i = 0; i < localRSize; i++) {
		rowIndG[i] = i + mesh->startIndexes[rank] * 2;
	}

	//MatCreateMPIDense(comm, mesh->vetrices.size() * 2, PETSC_DECIDE, PETSC_DECIDE, size
	//		* 3, PETSC_NULL, R);
	MatCreateMPIAIJ(comm, mesh->vetrices.size() * 2, 3, PETSC_DECIDE, PETSC_DECIDE, 3, PETSC_NULL, 0, PETSC_NULL, R);

	for (int j = 0; j < 3; j++) {
		PetscReal *values;
		PetscInt colIndG = rank * 3 + j;
		VecGetArray(nullSpace->localBasis[j], &values);
		MatSetValues(*R, localRSize, rowIndG, 1, &colIndG, values, INSERT_VALUES);
		VecRestoreArray(nullSpace->localBasis[j], &values);
	}

	MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);

	//	MatView(*R, PETSC_VIEWER_STDOUT_WORLD);

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

void Generate2DElasticityClusterNullSpace(Mesh *mesh, SubdomainCluster *cluster) {

	MPI_Comm comm = cluster->clusterComm;
	NullSpaceInfo *nullSpace = new NullSpaceInfo();
	NullSpaceInfo *gNullSpace = new NullSpaceInfo();

	PetscInt rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	nullSpace->localDimension = 3;
	nullSpace->isDomainSingular = true;
	nullSpace->isSubDomainSingular = true;

	gNullSpace->localDimension = 3;
	gNullSpace->isDomainSingular = true;
	gNullSpace->isSubDomainSingular = true;

	//Creating of matrix R - null space basis
	Mat *R = &(nullSpace->R);
	Mat *Rglob = &(gNullSpace->R);

	nullSpace->localBasis = new Vec[3];
	for (int i = 0; i < 3; i++) {
		VecCreateSeq(PETSC_COMM_SELF, mesh->vetrices.size() * 2, &(nullSpace->localBasis[i]));
	}

	PetscInt localRSize = mesh->vetrices.size() * 2;
	PetscInt *rowIndG = new PetscInt[localRSize];
	PetscInt *rowIndGlob = new PetscInt[localRSize];

	int cRank;
	MPI_Comm_rank(cluster->clusterComm, &cRank);

	int counter = 0;
	for (std::map<PetscInt, Point*>::iterator v = mesh->vetrices.begin(); v
			!= mesh->vetrices.end(); v++) {

		rowIndGlob[counter] = v->first * 2;
		rowIndG[counter++] = (v->first + cluster->indexDiff) * 2;
		rowIndGlob[counter] = v->first * 2 + 1;
		rowIndG[counter++] = (v->first + cluster->indexDiff) * 2 + 1;

		VecSetValue(nullSpace->localBasis[0], (v->first - mesh->startIndexes[rank])
				* 2, 1, INSERT_VALUES);
		VecSetValue(nullSpace->localBasis[1], (v->first - mesh->startIndexes[rank])
				* 2 + 1, 1, INSERT_VALUES);
		VecSetValue(nullSpace->localBasis[2], (v->first - mesh->startIndexes[rank])
				* 2, -v->second->y, INSERT_VALUES);
		VecSetValue(nullSpace->localBasis[2], (v->first - mesh->startIndexes[rank])
				* 2 + 1, v->second->x, INSERT_VALUES);
	}

	MatCreateMPIAIJ(PETSC_COMM_WORLD, mesh->vetrices.size() * 2, PETSC_DECIDE, PETSC_DECIDE, cluster->clusterCount
			* 3, 3, PETSC_NULL, 0, PETSC_NULL, Rglob);

	for (int j = 0; j < 3; j++) {
		PetscReal *values;
		PetscInt colIndG = cluster->clusterColor * 3 + j;
		VecGetArray(nullSpace->localBasis[j], &values);
		MatSetValues(*Rglob, localRSize, rowIndGlob, 1, &colIndG, values, INSERT_VALUES);
		VecRestoreArray(nullSpace->localBasis[j], &values);
	}

	delete[] rowIndGlob;

	MatAssemblyBegin(*Rglob, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*Rglob, MAT_FINAL_ASSEMBLY);

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

	//MatCreateMPIDense(comm, mesh->vetrices.size() * 2, PETSC_DECIDE, PETSC_DECIDE, size
	//		* 3, PETSC_NULL, R);
	MatCreateMPIAIJ(cluster->clusterComm, mesh->vetrices.size() * 2, 3, PETSC_DECIDE, PETSC_DECIDE, 3, PETSC_NULL, 0, PETSC_NULL, R);

	for (int j = 0; j < 3; j++) {
		PetscReal *values;
		PetscInt colIndG = cRank * 3 + j;
		VecGetArray(nullSpace->localBasis[j], &values);
		MatSetValues(*R, localRSize, rowIndG, 1, &colIndG, values, INSERT_VALUES);
		VecRestoreArray(nullSpace->localBasis[j], &values);
	}

	delete[] rowIndG;

	MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);

	cluster->clusterNullSpace = nullSpace;
	cluster->outerNullSpace = gNullSpace;
}

Feti1* createFeti(Mesh *mesh, PetscReal(*f)(Point), PetscReal(*K)(Point),
		MPI_Comm comm) {
	Mat A, B;
	Vec b, lmb;
	NullSpaceInfo nullSpace;
	FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh, A, b, f, K);
	GenerateJumpOperator(mesh, B, lmb);
	Generate2DLaplaceNullSpace(mesh, nullSpace.isDomainSingular, nullSpace.isSubDomainSingular, &(nullSpace.R));

	//return new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), comm);
	return NULL;
}
