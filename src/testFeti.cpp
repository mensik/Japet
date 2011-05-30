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
	PetscReal E = 2.1e5, mu = 0.3;
	PetscInitialize(&argc, &argv, 0, help);
	PetscTruth flg;
	char fileName[PETSC_MAX_PATH_LEN] = "benchmarks/arc.med";

	PetscOptionsGetReal(PETSC_NULL, "-japet_E", &E, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_mu", &mu, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_dens", &den, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-japet_mesh", fileName, PETSC_MAX_PATH_LEN
			- 1, &flg);
	{

		ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		PetscViewer v;
		Mesh *mesh = new Mesh();

		bool bound[] = { false, false, false, true };

		mesh->generateTearedRectMesh(0, 500, 0, 500, 1, 3, 3, bound);
		/*
		 PetscPrintf(PETSC_COMM_WORLD, "Loading mesh ... ");
		 //mesh->generateRectangularMesh(m, n, k, l, h);
		 mesh->loadHDF5(fileName);
		 PetscPrintf(PETSC_COMM_WORLD, "done.\n\nTearing mesh ...");
		 mesh->partition(size);
		 mesh->tear();
		 */
		PetscPrintf(PETSC_COMM_WORLD, "done.\n\n");

		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		//mesh->dumpForMatlab(v);
		//PetscViewerDestroy(v);

		//mesh->saveHDF5("outMesh.med");

		//***********************************************************************************************

		Mat A;
		Vec b;

		FEMAssemble2DElasticity(PETSC_COMM_WORLD, mesh, A, b, E, mu, funDensity, funGravity);

		PetscPrintf(PETSC_COMM_WORLD, "Mass matrix assembled \n");

		Mat B;
		Vec lmb;

		GenerateTotalJumpOperator(mesh, 2, B, lmb);

		PetscPrintf(PETSC_COMM_WORLD, "Hump assembled \n");

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/elast.m", FILE_MODE_WRITE, &v);
		MatView(A, v);
		VecView(b, v);
		MatView(B, v);
		PetscViewerDestroy(v);


		NullSpaceInfo nullSpace;


		Generate2DElasticityNullSpace(mesh, &nullSpace, PETSC_COMM_WORLD);

		Feti1
		*feti =
		new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		//AFeti
		//		*ifeti =
		//				new InexactFeti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		feti->setIsVerbose(true);
		feti->solve();
		feti->saveIterationInfo("feti.log");


		//PetscPrintf(PETSC_COMM_WORLD, "Ready to solve \n");
		//ifeti->setIsVerbose(true);
		//ifeti->solve();
		//ifeti->saveIterationInfo("ifeti.log");

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
