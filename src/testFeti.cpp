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

	PDCommManager commManager(PETSC_COMM_WORLD, ALL_ALL_SAMEROOT);
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

		PetscLogStageRegister("Meshing", &meshing);
		PetscLogStagePush(meshing);
		mesh->generateTearedRectMesh(0, conf->m * conf->H, 0, conf->n * conf->H, conf->h, conf->m, conf->n, bound);
			PetscLogStagePop();

		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		//mesh->dumpForMatlab(v);
		//PetscViewerDestroy(v);

		//***********************************************************************************************

		PetscLogStageRegister("Assembly", &assembly);
		PetscLogStagePush(assembly);
		Mat A;
		Vec b;

		if (problemType == 1) {
			FEMAssemble2DElasticity(PETSC_COMM_WORLD, mesh, A, b, conf->E, conf->mu, funDensity, funGravity);
		} else {
			FEMAssembleTotal2DLaplace(PETSC_COMM_WORLD, mesh, A, b, funConst, funConst);
		}

		PetscPrintf(PETSC_COMM_WORLD, "\nMass matrix assembled \n");

		Mat B;
		Vec lmb;
		if (problemType == 1) {
			GenerateTotalJumpOperator(mesh, 2, B, lmb, &commManager);
		} else {
			GenerateTotalJumpOperator(mesh, 1, B, lmb,  &commManager);
		}

		PetscPrintf(PETSC_COMM_WORLD, "Jump operator assembled \n");
		PetscLogStagePop();

		NullSpaceInfo nullSpace;

		if (problemType == 1) {
			Generate2DElasticityNullSpace(mesh, &nullSpace, PETSC_COMM_WORLD);
		} else {
			Generate2DLaplaceTotalNullSpace(mesh, &nullSpace, PETSC_COMM_WORLD);
		}

		PetscLogStageRegister("FETI", &fetiStage);
		PetscLogStagePush(fetiStage);

		Feti1
				*feti =
						new mFeti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD, conf->coarseProblemMethod);

		feti->setIsVerbose(true);
		feti->solve();

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

	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
