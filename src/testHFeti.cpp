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

		PDCommManager* commManager =
				new PDCommManager(PETSC_COMM_WORLD, conf->pdStrategy);

		ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);

		PetscViewer v;
		Mesh *mesh = new Mesh();

		bool bound[] = { false, false, false, true };

		mesh->generateTearedRectMesh(0, conf->Hx, 0.0, conf->Hy, conf->h, conf->m, conf->n, bound, commManager);

		Mat Bl, Bg, BTg, BTl;
		Vec lmbG, lmbL, lmb;

		SubdomainCluster cluster;
		mesh->generateRectMeshCluster(&cluster, conf->m, conf->n, 2, 2);
		GenerateClusterJumpOperator(mesh, &cluster, Bg, BTg, lmbG, Bl, BTl, lmbL);
		Generate2DElasticityClusterNullSpace(mesh, &cluster);

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/out.m", FILE_MODE_WRITE, &v);
		MatView(Bg, v);
		MatView(BTg, v);
		MatView(cluster.outerNullSpace->R, v);
		PetscViewerDestroy(v);

		std::stringstream ss;

		ss << "../matlab/out" << cluster.clusterColor << ".m";
		PetscViewerBinaryOpen(cluster.clusterComm, ss.str().c_str(), FILE_MODE_WRITE, &v);
		MatView(Bl, v);
		MatView(BTl, v);
		MatView(cluster.clusterNullSpace->R, v);
		MatView(cluster.clusterR.systemR, v);
		VecView(cluster.clusterR.systemGNullSpace[2], v);
		PetscViewerDestroy(v);

		Mat A;
		Vec b;

		FEMAssemble2DElasticity(commManager->getPrimal(), mesh, A, b, conf->E, conf->mu, funDensity, funGravity);

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
		PetscPrintf(PETSC_COMM_WORLD, "\n");
		hFeti->solve();

		/*
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/mesh.m", FILE_MODE_WRITE, &v);
		mesh->dumpForMatlab(v);
		PetscViewerDestroy(v);
		delete mesh;

		Vec u;
		VecDuplicate(b, &u);
		hFeti->copySolution(u);
		//hFeti->copyLmb(lmb);

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/out.m", FILE_MODE_WRITE, &v);
		MatView(A, v);
		VecView(b, v);
		MatView(B, v);
		VecView(u, v);
		VecView(lmb, v);
		PetscViewerDestroy(v);

		delete hFeti;
*/
}

ierr = PetscFinalize();
CHKERRQ(ierr);
return 0;
}
