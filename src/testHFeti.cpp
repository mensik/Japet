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
	PetscMPIInt rank, size;

	PetscInitialize(&argc, &argv, 0, help);

	{

		PetscPrintf(PETSC_COMM_WORLD, "***************************************************\n");
		PetscPrintf(PETSC_COMM_WORLD, "                    TEST hFETI \n");
		PetscPrintf(PETSC_COMM_WORLD, "***************************************************\n\n");

		ConfigManager *conf = ConfigManager::Instance();

		//Process ranking interchange

		int oRank; //old Rank
		MPI_Comm_rank(PETSC_COMM_WORLD, &oRank);
		int clXCount = 2;
		int clYCount = 4;
		int clXsize = conf->m / clXCount;
		int clYsize = conf->n / clYCount;

		int clInd = oRank % (clXsize * clYsize); // index inside cluster
		int clX = clInd % clXsize;
		int clY = (clInd - clX) / clXsize;

		int gInd = (oRank - clInd) / (clXsize * clYsize); //index of cluster
		int gX = gInd % clXCount; // X index of cluster
		int gY = (gInd - gX) / clXCount; // Y index of cluster

		int desiredRank = gY * clXsize * clYsize * clXCount + clY * conf->n + gX
				* clXsize + clX;

		MPI_Comm PERMUTATED_WORLD;
		MPI_Comm_split(PETSC_COMM_WORLD, 0, desiredRank, &PERMUTATED_WORLD);

		MPI_Comm_rank(PERMUTATED_WORLD, &rank);

		char pName[MPI_MAX_PROCESSOR_NAME];
		int pLen;

		MPI_Get_processor_name(pName, &pLen);
		//PetscPrintf(PETSC_COMM_SELF, " processor: %s \n", pName);

		//	PetscPrintf(PETSC_COMM_SELF, "I am %d in cluster %d [%d,%d - %d,%d] , but I should be %d. Processor %s \n", oRank, gInd, gX, gY, clX, clY, rank, pName);

		PDCommManager* commManager =
				new PDCommManager(PERMUTATED_WORLD, conf->pdStrategy);

		//	CHKERRQ(ierr);
		MPI_Comm_size(PERMUTATED_WORLD, &size);

		PetscViewer v;
		Mesh *mesh = new Mesh();

		bool bound[] = { false, false, false, true };
		PetscReal h = conf->Hx / (PetscReal) ((PetscReal) (conf->m)
				* (PetscReal) (conf->reqSize));

		mesh->generateTearedRectMesh(0, conf->Hx, 0.0, conf->Hy, h, conf->m, conf->n, bound, commManager);

		Mat A;
		Vec b;

		FEMAssemble2DElasticity(commManager->getPrimal(), mesh, A, b, conf->E, conf->mu, funDensity, funGravity);
//
//		std::stringstream ss2;//
//
//		ss2 << "../matlab/data/A" << rank << ".m";
//		PetscViewerBinaryOpen(PETSC_COMM_SELF, ss2.str().c_str(), FILE_MODE_WRITE, &v);
//		MatView(A, v);
//		PetscViewerDestroy(v);

		Mat Bl, Bg, BTg, BTl;
		Vec lmbG, lmbL, lmb;

		SubdomainCluster cluster;
//		/*
//		 Mat B, BT;
//		 GenerateTotalJumpOperator(mesh, 2, B, BT, lmb, commManager);
//		 PetscViewerBinaryOpen(PERMUTATED_WORLD, "../matlab/B.m", FILE_MODE_WRITE, &v);
//		 MatView(B, v);
//		 //MatView(BTg, v);
//		 //MatView(cluster.outerNullSpace->R, v);
//		 PetscViewerDestroy(v);
//
		mesh->generateRectMeshCluster(&cluster, conf->m, conf->n, clXCount, clYCount, PERMUTATED_WORLD);

		GenerateClusterJumpOperator(mesh, &cluster, Bg, BTg, lmbG, Bl, BTl, lmbL, PERMUTATED_WORLD);
		Generate2DElasticityClusterNullSpace(mesh, &cluster, PERMUTATED_WORLD);

		PetscViewerBinaryOpen(PERMUTATED_WORLD, "../matlab/data/out.m", FILE_MODE_WRITE, &v);
		MatView(Bg, v);
		MatView(BTg, v);
		MatView(cluster.outerNullSpace->R, v);
		PetscViewerDestroy(v);

	//	std::stringstream ss;

	//	ss << "../matlab/data/out" << cluster.clusterColor << ".m";
	//	PetscViewerBinaryOpen(cluster.clusterComm, ss.str().c_str(), FILE_MODE_WRITE, &v);
	//	MatView(Bl, v);
		//MatView(BTl, v);
	//	MatView(cluster.clusterNullSpace->R, v);

	//	MatView(cluster.clusterR.systemR, v);

	//	VecView(cluster.clusterR.systemGNullSpace[0], v);
	//	VecView(cluster.clusterR.systemGNullSpace[1], v);
	//	VecView(cluster.clusterR.systemGNullSpace[2], v);

	//	VecView(cluster.outerNullSpace->localBasis[0], v);
	//	VecView(cluster.outerNullSpace->localBasis[1], v);
	//	VecView(cluster.outerNullSpace->localBasis[2], v);

	//	PetscViewerDestroy(v);

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


		// PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		// mesh->dumpForMatlab(v);
		// PetscViewerDestroy(v);

		//	delete mesh;
		/*
		 Vec u;
		 VecDuplicate(b, &u);
		 hFeti->copySolution(u);
		 //hFeti->copyLmb(lmb);

		 PetscViewerBinaryOpen(PERMUTATED_WORLD, "../matlab/out.m", FILE_MODE_WRITE, &v);
		 //MatView(A, v);
		 VecView(b, v);
		 //	MatView(B, v);
		 VecView(u, v);
		 //VecView(lmb, v);
		 PetscViewerDestroy(v);
		 */
		//	delete hFeti;

	}

	ierr = PetscFinalize();
	CHKERRQ(ierr);
	return 0;
}
