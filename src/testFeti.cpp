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

int main(int argc, char *argv[]) {
	PetscErrorCode ierr;
	PetscMPIInt rank, size;
	PetscReal E = 2.1e5, mu = 0.3, h = 2.0;
	PetscInt m = 3, n = 3;
	PetscInitialize(&argc, &argv, 0, help);
	PetscTruth flg;
	PetscLogStage meshing, assembly, fetiStage;
	char fileName[PETSC_MAX_PATH_LEN] = "benchmarks/arc.med";

	PetscOptionsGetInt(PETSC_NULL, "-japet_m", &m, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_n", &n, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_h", &h, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-japet_mesh", fileName, PETSC_MAX_PATH_LEN
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

		PetscLogStageRegister("Meshing", &meshing);
		PetscLogStagePush(meshing);
		mesh->generateTearedRectMesh(0, 300, 0, 300, h, m, n, bound);
		PetscLogStagePop();

		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		//mesh->dumpForMatlab(v);
		//PetscViewerDestroy(v);

		//***********************************************************************************************

		PetscLogStageRegister("Assembly", &assembly);
		PetscLogStagePush(assembly);
		Mat A;
		Vec b;

		FEMAssemble2DElasticity(PETSC_COMM_WORLD, mesh, A, b, E, mu, funDensity, funGravity);

		PetscPrintf(PETSC_COMM_WORLD, "\nMass matrix assembled \n");

		Mat B;
		Vec lmb;

		GenerateTotalJumpOperator(mesh, 2, B, lmb);

		PetscPrintf(PETSC_COMM_WORLD, "Jump operator assembled \n");
		PetscLogStagePop();



		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/elast.m", FILE_MODE_WRITE, &v);
		MatView(A, v);
		VecView(b, v);
		MatView(B, v);
		PetscViewerDestroy(v);


		NullSpaceInfo nullSpace;


		Generate2DElasticityNullSpace(mesh, &nullSpace, PETSC_COMM_WORLD);

		PetscLogStageRegister("FETI", &fetiStage);
		PetscLogStagePush(fetiStage);
		//Feti1
		//*feti =
		//new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		AFeti
				*ifeti =
						new InexactFeti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		//feti->setIsVerbose(true);
		//feti->solve();

		//feti->saveIterationInfo("feti.log");


		//PetscPrintf(PETSC_COMM_WORLD, "Ready to solve \n");
		ifeti->setIsVerbose(true);
		ifeti->solve();
		ifeti->saveIterationInfo("ifeti.log");
		PetscLogStagePop();
		delete mesh;

		//Vec x;
		//VecDuplicate(b, &x);
		//ifeti->copySolution(x);
		//save2DResultHDF5("outMesh.med", "fetiResult", x);

	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
