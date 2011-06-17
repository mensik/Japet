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
	PetscLogStage meshing, assembly, fetiStage;

	PetscInitialize(&argc, &argv, 0, help);

	{

		PDCommManager* commManager = new PDCommManager(PETSC_COMM_WORLD, TEST);

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

		PetscLogStageRegister("Meshing", &meshing);
		PetscLogStagePush(meshing);
		mesh->generateTearedRectMesh(0, conf->m * conf->H, 0, conf->n * conf->H, conf->h, conf->m, conf->n, bound, commManager);
		PetscLogStagePop();

		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		//mesh->dumpForMatlab(v);
		//PetscViewerDestroy(v);

		//***********************************************************************************************


		PetscLogStageRegister("Assembly", &assembly);
		PetscLogStagePush(assembly);
		Mat A;
		Vec b;

		if (commManager->isPrimal()) {
			if (problemType == 1) {
				FEMAssemble2DElasticity(commManager->getPrimal(), mesh, A, b, conf->E, conf->mu, funDensity, funGravity);
			} else {
				FEMAssembleTotal2DLaplace(commManager->getPrimal(), mesh, A, b, funConst, funConst);
			}
		}

		PetscPrintf(PETSC_COMM_WORLD, "\nMass matrix assembled \n");

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

		PetscPrintf(PETSC_COMM_WORLD, "Jump operator assembled \n");
		PetscLogStagePop();

		if (commManager->isPrimal()) {
			PetscViewerBinaryOpen(commManager->getPrimal(), "../matlab/outP.m", FILE_MODE_WRITE, &v);
			MatView(A, v);
			VecView(b, v);
			MatView(BT, v);
			PetscViewerDestroy(v);
		}

		if (commManager->isDual()) {
			PetscViewerBinaryOpen(commManager->getDual(), "../matlab/outD.m", FILE_MODE_WRITE, &v);
			MatView(B, v);
			VecView(lmb, v);
			PetscViewerDestroy(v);
		}

		PetscLogStageRegister("FETI", &fetiStage);
		// PetscLogStagePush(fetiStage);


		Feti1
				*feti =
						new mFeti1(commManager, A, b, BT, B, lmb, &nullSpace, mesh->vetrices.size(), conf->coarseProblemMethod);

		 feti->setIsVerbose(true);

		 feti->solve();
		/*
		 PetscLogStagePop();

		 feti->saveIterationInfo(conf->name);

		 if (conf->saveOutputs) {
		 PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		 mesh->dumpForMatlab(v);
		 PetscViewerDestroy(v);

		 Vec x;
		 VecDuplicate(b, &x);
		 feti->copySolution(x);
		 feti->copyLmb(lmb);

		 PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/out.m", FILE_MODE_WRITE, &v);
		 MatView(A, v);
		 VecView(b, v);
		 MatView(B, v);
		 VecView(x, v);
		 VecView(lmb, v);
		 PetscViewerDestroy(v);
		 }

		 MatDestroy(A);
		 MatDestroy(B);

		 delete mesh;
		 delete feti;
		 */

		delete commManager;
	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
