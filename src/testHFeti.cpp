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

#include "vt_user.h"

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

int getDesiredRank(int m, int n, int cm, int cn, int rank) {

	int cSize = cm * cn;
	int clustInRow = m / cm;

	int clusterNumber = rank / cSize;
	int clusterPos = rank % cSize;

	int cY = clusterPos / cm;
	int cX = clusterPos % cm;

	int desRank = (clusterNumber / clustInRow) * cSize * clustInRow + // Number of ranks in row o clusters
			cY * m + // Number of ranks in rows
			(clusterNumber % clustInRow) * cm + // Number of ranks in row preceding actual cluster
			cX;

	return desRank;
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

		MPI_Comm PERMUTATED_WORLD;

		MyLogger::Instance()->getTimer("comInit")->startTimer();
		int desRank =
				getDesiredRank(conf->m, conf->n, conf->clustM, conf->clustN, oRank);

		//MPI_Barrier(PETSC_COMM_WORLD);
		MPI_Comm_split(PETSC_COMM_WORLD, 0, desRank, &PERMUTATED_WORLD);
		//PERMUTATED_WORLD = PETSC_COMM_WORLD;
		//MPI_Barrier(PERMUTATED_WORLD);

		int rank;
		MPI_Comm_rank(PERMUTATED_WORLD, &rank);

		char pName[MPI_MAX_PROCESSOR_NAME];
		int pLen;
		MPI_Get_processor_name(pName, &pLen);

		PDCommManager* commManager =
				new PDCommManager(PERMUTATED_WORLD, conf->pdStrategy);

		int size;
		MPI_Comm_size(PERMUTATED_WORLD, &size);

		MyLogger::Instance()->getTimer("comInit")->stopTimer();

		Mesh *mesh = new Mesh();

		bool bound[] = { false, false, false, true };
		PetscReal h = conf->Hx / (PetscReal) ((PetscReal) (conf->m)
				* (PetscReal) (conf->reqSize));

		mesh->generateTearedRectMesh(0, conf->Hx, 0.0, conf->Hy, h, conf->m, conf->n, bound, commManager);

		Mat A;
		Vec b;

		{
		VT_TRACER("Elast. assemnle");	
		MyLogger::Instance()->getTimer("elastAss")->startTimer();
		FEMAssemble2DElasticity(commManager->getPrimal(), mesh, A, b, conf->E, conf->mu, funDensity, funGravity);
		MyLogger::Instance()->getTimer("elastAss")->stopTimer();
		PetscPrintf(PERMUTATED_WORLD, "\nElasticity assembled in              %e s\n", MyLogger::Instance()->getTimer("elastAss")->getTotalTime());
		}

		Mat Bl, Bg, BTg, BTl;
		Vec lmbG, lmbL;

		SubdomainCluster cluster;
		
		{
		VT_TRACER("Cluster init");
		MyLogger::Instance()->getTimer("clustInit")->startTimer();
		mesh->generateRectMeshCluster(&cluster, conf->m, conf->n, clXCount, clYCount, PERMUTATED_WORLD);
		MyLogger::Instance()->getTimer("clustInit")->stopTimer();
		PetscPrintf(PERMUTATED_WORLD, "Cluster constructed in               %e s\n", MyLogger::Instance()->getTimer("clustInit")->getTotalTime());
		}

		MyLogger::Instance()->getTimer("clustJump")->startTimer();
		GenerateClusterJumpOperator(mesh, &cluster, Bg, BTg, lmbG, Bl, BTl, lmbL, PERMUTATED_WORLD);
		MyLogger::Instance()->getTimer("clustJump")->stopTimer();
		PetscPrintf(PERMUTATED_WORLD, "Cluster jump operator constructed in %e s\n", MyLogger::Instance()->getTimer("clustJump")->getTotalTime());


		MyLogger::Instance()->getTimer("clustNull")->startTimer();
		Generate2DElasticityClusterNullSpace(mesh, &cluster, PERMUTATED_WORLD);
		MyLogger::Instance()->getTimer("clustNull")->stopTimer();
		PetscPrintf(PERMUTATED_WORLD, "Cluster null space constructed in    %e s \n", MyLogger::Instance()->getTimer("clustNull")->getTotalTime());
		{
		VT_TRACER("HFeti Init start");
		MyLogger::Instance()->getTimer("Initiation")->startTimer();
		}
		
		HFeti
				*hFeti;
		{
		VT_TRACER("HFETI initiation");	
		hFeti =
						new HFeti(commManager, A, b, Bg, BTg, Bl, BTl, lmbG, lmbL, &cluster, mesh->vetrices.size());
		}
		MyLogger::Instance()->getTimer("Initiation")->stopTimer();
		PetscPrintf(PERMUTATED_WORLD, "HFeti initiated in                   %e s \n\n\n", MyLogger::Instance()->getTimer("Initiation")->getTotalTime());

		//		Feti1
		//				*feti =
		//						new Feti1(A, b, B, lmb, &nullSpace, mesh->vetrices.size(), PETSC_COMM_WORLD);

		//		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/system.m", FILE_MODE_WRITE, &v);
		//		feti->dumpSystem(v);
		//		hFeti->dumpSolution(v);
		//		PetscViewerDestroy(v);





		hFeti->setIsVerbose(true);
		{
		VT_TRACER("HFeti solving");
		MyLogger::Instance()->getTimer("Solving")->startTimer();
		hFeti->solve();
		MyLogger::Instance()->getTimer("Solving")->stopTimer();
		}
		if (commManager->isPrimalRoot()) {

			PetscPrintf(PETSC_COMM_SELF, "Solve time             : %e \n", MyLogger::Instance()->getTimer("Solving")->getTotalTime());
			PetscPrintf(PETSC_COMM_SELF, "Init time              : %e \n", MyLogger::Instance()->getTimer("Initiation")->getTotalTime());

		}
/*
		PetscViewer v;
		PetscViewerBinaryOpen(PERMUTATED_WORLD, "../matlab/data/B.m", FILE_MODE_WRITE, &v);
		MatView(Bg, v);
		PetscViewerDestroy(v);
*/
		delete mesh;
		delete hFeti;

	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
