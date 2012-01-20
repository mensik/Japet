static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include <string>
#include <sstream>

#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "fem.h"
#include "solver.h"
#include "feti.h"

static PetscReal den = 7.85e-9;

void funGravity(Element* e, PetscReal density, PetscReal *fs) {
	fs[0] = 0;
	fs[1] = -9810 * density;
}

PetscReal funDensity(Element* e) {
	return den;
}

PetscReal funConst(Element *el) {
	return 1;
}

PetscReal funSin(Point n) {
	return sin(n.x + n.y);
}

PetscReal funStep(Point n) {

	PetscReal c = n.x > 2 ? -1 : 1;
	c *= n.y > 2 ? -1 : 1;
	return c;
}

bool cf(PetscInt itNumber, PetscReal rNorm, Vec *r) {
	PetscPrintf(PETSC_COMM_SELF, "%d - %e\n", itNumber, rNorm);
	return itNumber > 5;
}

int main(int argc, char *argv[]) {
	//PetscReal (*fList[])(Point) = {funConst, funSin, funStep};
	PetscErrorCode ierr;

	PetscInitialize(&argc, &argv, 0, help);

	{

		PetscPrintf(PETSC_COMM_WORLD, "***************************************************\n");
		PetscPrintf(PETSC_COMM_WORLD, "                    TEST hFETI \n");
		PetscPrintf(PETSC_COMM_WORLD, "***************************************************\n\n");

		ConfigManager *conf = ConfigManager::Instance();

		//Process ranking interchange

		int oRank; //old Rank
		MPI_Comm_rank(PETSC_COMM_WORLD, &oRank);

		int clXsize = conf->clustM;
		int clYsize = conf->clustN;
		int clXCount = conf->m / clXsize;
		int clYCount = conf->n / clYsize;

		int clInd = oRank % (clXsize * clYsize); // index inside cluster
		int clX = clInd % clXsize;
		int clY = (clInd - clX) / clXsize;

		int gInd = (oRank - clInd) / (clXsize * clYsize); //index of cluster
		int gX = gInd % clXCount; // X index of cluster
		int gY = (gInd - gX) / clXCount; // Y index of cluster

		int desiredRank = gY * clXsize * clYsize * clXCount + clY * conf->n + gX
				* clXsize + clX;

		MPI_Comm PERMUTATED_WORLD;

		MPI_Barrier(PETSC_COMM_WORLD);
		//int stat = MPI_Comm_split(PETSC_COMM_WORLD, 48, oRank, &PERMUTATED_WORLD);
		PERMUTATED_WORLD = PETSC_COMM_WORLD;
		MPI_Barrier(PERMUTATED_WORLD);

		int rank;
		MPI_Comm_rank(PERMUTATED_WORLD, &rank);

		char pName[MPI_MAX_PROCESSOR_NAME];
		int pLen;
		MPI_Get_processor_name(pName, &pLen);

		PDCommManager* commManager =
				new PDCommManager(PERMUTATED_WORLD, conf->pdStrategy);

		int size;
		MPI_Comm_size(PERMUTATED_WORLD, &size);

		Mesh *mesh = new Mesh();

		bool bound[] = { false, false, false, true };
		PetscReal h = conf->Hx / (PetscReal) ((PetscReal) (conf->m)
				* (PetscReal) (conf->reqSize));

		mesh->generateTearedRectMesh(0, conf->Hx, 0.0, conf->Hy, h, conf->m, conf->n, bound, commManager);

		Mat A;
		Vec b;

		FEMAssemble2DElasticity(commManager->getPrimal(), mesh, A, b, conf->E, conf->mu, funDensity, funGravity);
		PetscPrintf(PERMUTATED_WORLD, "Elasticity assembled \n\n");

		Mat Bl, Bg, BTg, BTl;
		Vec lmbG, lmbL;

		SubdomainCluster cluster;

		mesh->generateRectMeshCluster(&cluster, conf->m, conf->n, clXCount, clYCount, PERMUTATED_WORLD);
		PetscPrintf(PERMUTATED_WORLD, "Cluster constructed \n");

		GenerateClusterJumpOperator(mesh, &cluster, Bg, BTg, lmbG, Bl, BTl, lmbL, PERMUTATED_WORLD);
		PetscPrintf(PERMUTATED_WORLD, "Cluster jump operator constructed \n");

		Generate2DElasticityClusterNullSpace(mesh, &cluster, PERMUTATED_WORLD);
		PetscPrintf(PERMUTATED_WORLD, "Cluster null space constructed \n");

		HFeti
				*hFeti =
						new HFeti(commManager, A, b, Bg, BTg, Bl, BTl, lmbG, lmbL, &cluster, mesh->vetrices.size());

		//		Feti1
		//				*feti =
		//						new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		//		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/system.m", FILE_MODE_WRITE, &v);
		//		feti->dumpSystem(v);
		//		hFeti->dumpSolution(v);
		//		PetscViewerDestroy(v);


		hFeti->setIsVerbose(true);
		hFeti->solve();

		delete mesh;
		delete hFeti;

	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
