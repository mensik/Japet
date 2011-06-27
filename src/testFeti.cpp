static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "fem.h"
#include "solver.h"
#include "feti.h"

static PetscReal den = 7.85e-9;

void funGravity(Element* e, PetscReal density, PetscReal *fs) {
	fs[0] = 0;
	fs[1] = -9800 * density;
}

PetscReal funDensity(Element* e) {
	return den;
}

PetscReal funConst(Point n) {
	return 1;
}

int main(int argc, char *argv[]) {
	PetscErrorCode ierr;
	PetscMPIInt rank, size;
	PetscLogStage assembly, fetiStage;

	PetscInitialize(&argc, &argv, 0, help);

	{
		PetscPrintf(PETSC_COMM_WORLD, "***************************************************\n");
		PetscPrintf(PETSC_COMM_WORLD, "                    TEST FETI \n");
		PetscPrintf(PETSC_COMM_WORLD, "***************************************************\n");
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		PetscViewer v;
		Mesh *mesh = new Mesh();

		bool bound[] = { false, false, false, true };

		ConfigManager *conf = ConfigManager::Instance();

		int problemType = conf->problem;
		PDCommManager* commManager =
				new PDCommManager(PETSC_COMM_WORLD, conf->pdStrategy);

		PetscLogStageRegister("Assembly", &assembly);
		PetscLogStagePush(assembly);

		PetscReal h = conf->Hx / (PetscReal) ((PetscReal) (conf->m)
				* (PetscReal) (conf->reqSize));

		mesh->generateTearedRectMesh(0, conf->Hx, 0.0, conf->Hy, h, conf->m, conf->n, bound, commManager);
		if (conf->saveOutputs) {
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
			mesh->dumpForMatlab(v);
			PetscViewerDestroy(v);
		}

		//***********************************************************************************************


		Mat A;
		Vec b;

		if (commManager->isPrimal()) {
			if (problemType == 1) {
				FEMAssemble2DElasticity(commManager->getPrimal(), mesh, A, b, conf->E, conf->mu, funDensity, funGravity);
			} else {
				FEMAssembleTotal2DLaplace(commManager->getPrimal(), mesh, A, b, funConst, funConst);
			}
		}

		Mat B, BT;
		Vec lmb;
		NullSpaceInfo nullSpace;

		if (problemType == 1) {
			GenerateTotalJumpOperator(mesh, 2, B, BT, lmb, commManager);
			if (commManager->isPrimal()) Generate2DElasticityNullSpace(mesh, &nullSpace, commManager->getPrimal());
		} else {
			GenerateTotalJumpOperator(mesh, 1, B, BT, lmb, commManager);
			if (commManager->isPrimal()) Generate2DLaplaceTotalNullSpace(mesh, &nullSpace, commManager->getPrimal());
		}
		PetscLogStagePop();

		if (commManager->isPrimalRoot()) {
			PetscInt dimPrim, dimDual, dimNull;

			MatGetSize(A, &dimPrim, PETSC_NULL);
			MatGetSize(B, &dimDual, PETSC_NULL);
			MatGetSize(nullSpace.R, PETSC_NULL, &dimNull);

			PetscPrintf(PETSC_COMM_SELF, "\nPrimal var. : %d \nDual var.   : %d\nCoarse dim. : %d \n\n", dimPrim, dimDual, dimNull);
		}

		PetscLogStageRegister("FETI", &fetiStage);
		PetscLogStagePush(fetiStage);

		MyLogger::Instance()->getTimer("feti")->startTimer();
		Feti1
				*feti =
						new Feti1(commManager, A, b, BT, B, lmb, &nullSpace, mesh->vetrices.size(), 0, NULL, conf->coarseProblemMethod);

		feti->setIsVerbose(true);

		MyLogger::Instance()->getTimer("Solving")->startTimer();

		feti->solve();

		MyLogger::Instance()->getTimer("Solving")->stopTimer();
		MyLogger::Instance()->getTimer("feti")->stopTimer();

		if (commManager->isPrimalRoot()) {

			PetscPrintf(PETSC_COMM_SELF, "Total time             : %e \n", MyLogger::Instance()->getTimer("feti")->getTotalTime());
			PetscPrintf(PETSC_COMM_SELF, "Solve time             : %e \n", MyLogger::Instance()->getTimer("Solving")->getTotalTime());
		}

		if (commManager->isPrimal()) {

			MyTimer* fTimer = MyLogger::Instance()->getTimer("Factorization");

			PetscPrintf(commManager->getPrimal(), "Factorization - average: %e \n", fTimer->getAverageOverComm(commManager->getPrimal()));
			PetscPrintf(commManager->getPrimal(), "                   max : %e \n", fTimer->getMaxOverComm(commManager->getPrimal()));
		}
		if (commManager->isDual()) {

			MyTimer* cTimer = MyLogger::Instance()->getTimer("Coarse init");
			PetscPrintf(commManager->getDual(), "Coarse p. init - avg.  : %e \n", cTimer->getAverageOverComm(commManager->getDual()));
			PetscPrintf(commManager->getDual(), "                 max.  : %e \n", cTimer->getMaxOverComm(commManager->getDual()));

		}
		if (commManager->isDualRoot()) {

			MyTimer* iTimer = MyLogger::Instance()->getTimer("Iteration");
			PetscPrintf(PETSC_COMM_SELF, "Avg. iteration         : %e \n", iTimer->getAverageTime());

			MyTimer* cTimer = MyLogger::Instance()->getTimer("Coarse problem");
			PetscPrintf(PETSC_COMM_SELF, "Projection (2x)        : %e \n", cTimer->getAverageTime()
					* 2);
		}

		if (commManager->isPrimalRoot()) {
			MyTimer* mTimer = MyLogger::Instance()->getTimer("BA+BT");
			PetscPrintf(PETSC_COMM_SELF, "F                      : %e \n", mTimer->getAverageTime());
			MyTimer* fTimer = MyLogger::Instance()->getTimer("F^-1");
			PetscPrintf(PETSC_COMM_SELF, "F^-1                   : %e \n", fTimer->getAverageTime());

			PetscPrintf(PETSC_COMM_SELF, "D->P Scatter           : %e \n", MyLogger::Instance()->getTimer("DP scatter")->getAverageTime());
			PetscPrintf(PETSC_COMM_SELF, "P->D Scatter           : %e \n\n", MyLogger::Instance()->getTimer("PD scatter")->getAverageTime());

			MyLogger::Instance()->getTimer("Coarse init")->printMarkedTime(PETSC_COMM_SELF);
		}

		if (conf->saveOutputs) {
			if (commManager->isPrimal()) {

				Vec x;
				VecDuplicate(b, &x);
				feti->copySolution(x);

				PetscViewerBinaryOpen(commManager->getPrimal(), "../matlab/outP.m", FILE_MODE_WRITE, &v);
				MatView(A, v);
				VecView(b, v);
				MatView(BT, v);
				MatView(B, v);
				MatView(nullSpace.R, v);
				VecView(x, v);
				PetscViewerDestroy(v);

				VecDestroy(x);
			}

			if (commManager->isDual()) {
				PetscViewerBinaryOpen(commManager->getDual(), "../matlab/outD.m", FILE_MODE_WRITE, &v);
				VecView(lmb, v);
				PetscViewerDestroy(v);
			}
		}

		PetscLogStagePop();

		//delete feti;

		/*
		 feti->saveIterationInfo(conf->name);

		 if (conf->saveOutputs) {
		 PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		 mesh->dumpForMatlab(v);
		 PetscViewerDestroy(v);

		 feti->copyLmb(lmb);

		 PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/out.m", FILE_MODE_WRITE, &v);
		 MatView(A, v);
		 VecView(b, v);
		 MatView(B, v);
		 VecView(x, v);
		 VecView(lmb, v);
		 PetscViewerDestroy(v);
		 }

*/

		//MatDestroy(A);
		//MatDestroy(B);

		if (commManager->isPrimal()) {
			MatDestroy(A);
			MatDestroy(B);
			MatDestroy(BT);
			delete mesh;
		}

		delete commManager;
	}
	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
