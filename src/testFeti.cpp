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
	PetscReal E = 2.1e5, mu = 0.3, h = 2.0, H = 100;
	PetscInt m = 3, n = 3;
	PetscInitialize(&argc, &argv, 0, help);
	PetscTruth flg;
	PetscLogStage meshing, assembly, fetiStage;
	char name[PETSC_MAX_PATH_LEN] = "FetiTest.log";

	PetscOptionsGetInt(PETSC_NULL, "-japet_m", &m, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_n", &n, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_h", &h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_HH", &H, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-japet_name", name, PETSC_MAX_PATH_LEN
			- 1, &flg);
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

		int problemType = 0;

		PetscLogStageRegister("Meshing", &meshing);
		PetscLogStagePush(meshing);
		mesh->generateTearedRectMesh(0, m * H, 0, n * H, h, m, n, bound);
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
			FEMAssemble2DElasticity(PETSC_COMM_WORLD, mesh, A, b, E, mu, funDensity, funGravity);
		} else {
			FEMAssembleTotal2DLaplace(PETSC_COMM_WORLD, mesh, A, b, funConst, funConst);
		}

		PetscPrintf(PETSC_COMM_WORLD, "\nMass matrix assembled \n");

		Mat B;
		Vec lmb;
		if (problemType == 1) {
			GenerateTotalJumpOperator(mesh, 2, B, lmb);
		} else {
			GenerateTotalJumpOperator(mesh, 1, B, lmb);
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
						new mFeti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		//AFeti
		//		*ifeti =
		//				new InexactFeti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		//feti->setIsVerbose(true);
		//feti->solve();

		//feti->saveIterationInfo("feti.log");

		//PetscPrintf(PETSC_COMM_WORLD, "Ready to solve \n");
		feti->setIsVerbose(true);
		feti->solve();
		feti->saveIterationInfo(name);
		PetscLogStagePop();

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		mesh->dumpForMatlab(v);
		PetscViewerDestroy(v);
		delete mesh;

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

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
