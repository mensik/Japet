/**	@file		feti.h
 @brief	FETI Method
 @author Martin Mensik
 @date 	2010
 */

#ifndef FETI_H
#define FETI_H

#include <math.h>

#include "petscksp.h"
#include "petscmat.h"
#include "japetUtils.h"
#include "fem.h"
#include "solver.h"

/**
 * @brief Abstract FETI ancestor
 * @note		There is still some work to do. The messiest part of implementation is in dealing with
 object G matrix and computation of the projector P = I - G inv(G'G) G'
 *
 **/
class AFeti: public SolverApp, public SolverCtr {
protected:

	Solver *outerSolver; ///< Solver class for outer loop (default is CGSolver)
	bool isVerbose;

	Mat B; ///< Jump operator matrix
	Vec lmb; ///< Lambda vector
	Vec u; ///< solution
	Vec d; ///< dual right side

	Vec b; ///< Global force vector

	KSP kspG; ///< Global G'G solver

	bool isSingular; ///< is Matrix A singular
	bool isLocalSingular; ///< is local part of A singular
	Mat R; ///< Global null space of A
	Mat G; ///< BR
	PetscInt gM, gN; ///<	dimensions of G

	PetscReal lastNorm; ///< last computed norm

	Vec tgA;
	Vec tgB;

	Vec temp;
	Vec tempLoc;

	MPI_Comm comm; ///< Communication channel of processes

	PetscInt outIterations;
	PetscInt inIterations;

public:
	AFeti(Vec b, Mat B, Vec lmb, NullSpaceInfo *nullSpace, MPI_Comm comm);
	~AFeti();

	virtual void applyInvA(Vec in, IterationManager *itManager) = 0;
	void solve(Vec b) {
		this->b = b;
		solve();
	}
	virtual void solve();
	virtual void applyMult(Vec in, Vec out, IterationManager *info);
	virtual bool
	isConverged(PetscInt itNumber, PetscReal norm, PetscReal bNorm, Vec *vec);

	virtual Solver* instanceOuterSolver(Vec d, Vec l);

	void dumpSolution(PetscViewer v);
	void dumpSystem(PetscViewer v);
	void projectGOrth(Vec in); ///< Remove space spaned by G from vec in
	void copySolution(Vec out); /// <Copy solution to vector out
	void setIsVerbose(bool verbose) {
		isVerbose = verbose;
	}
	void saveIterationInfo(const char *fileName) {
		outerSolver->saveIterationInfo(fileName);
	}

	void setSystemSingular() {
		MatNullSpace NS;
		MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL, &NS);
		KSPSetNullSpace(kspG, NS);
	}

	PetscInt getOutIterations() {
		return outIterations;
	}
	PetscInt getInIterations() {
		return inIterations;
	}
};

/**
 @brief	Rather general (and functional) implementation of FETI-1 algorithm. Tearing of global
 domain and distribution of subdomains is done in preprocesing of Mesh object.

 Currently, CG implementation in solver.h is used for outer cycle and Petsc direct solver
 for inner cycle.

 **/

class Feti1: public AFeti {
protected:
	Mat A; ///< Global stifles matrix
	Mat Aloc; ///< Local part of stiffness matrix
	KSP kspA; ///< Local A solver

	MatNullSpace locNS; ///< local null space of local A

	Vec tempInv, tempInvGh, tempInvGhB;
public:
	Feti1(Mat A, Vec b, Mat B, Vec lmb, NullSpaceInfo *nullSpace,
			PetscInt localNodeCount, MPI_Comm comm);
	~Feti1();
	virtual void applyInvA(Vec in, IterationManager *itManager);
};

class InexactFeti1: public Feti1 {
	PetscReal outerPrec;
	PetscInt inCounter;
public:
	InexactFeti1(Mat A, Vec b, Mat B, Vec lmb, NullSpaceInfo *nullSpace,
			PetscInt localNodeCount, MPI_Comm comm);

	virtual void applyInvA(Vec in, IterationManager *itManager);
	virtual Solver* instanceOuterSolver(Vec d, Vec lmb);
	void setRequiredPrecision(PetscReal reqPrecision);
};

/**
 * @brief Hierarchical FETI implementation
 */
class HFeti: public AFeti {
	AFeti *subClusterSystem; ///< FETI1 system associated with cluste [cluster]
	SubdomainCluster *cluster; ///< Cluster info [cluster]

	Vec clustTemp, clustTempGh;
	Vec clustb;
	NullSpaceInfo *clustNullSpace;
	MatNullSpace clusterNS;

	Vec globTemp, globTempGh;

	PetscReal outerPrec;
	PetscInt inCounter;
public:
	HFeti(Mat A, Vec b, Mat BGlob, Mat BClust, Vec lmbGl, Vec lmbCl,
			SubdomainCluster *cluster, PetscInt localNodeCount, MPI_Comm comm);

	virtual void applyInvA(Vec in, IterationManager *itManager);
	void removeNullSpace(Vec in);
	virtual Solver* instanceOuterSolver(Vec d, Vec lmb);
	virtual void setRequiredPrecision(PetscReal reqPrecision);
};

void GenerateJumpOperator(Mesh *mesh, Mat &B, Vec &lmb);

void GenerateTotalJumpOperator(Mesh *mesh, int d, Mat &B, Vec &lmb);

void GenerateClusterJumpOperator(Mesh *mesh, SubdomainCluster *cluster,
		Mat &BGlob, Vec &lmbGlob, Mat &BCluster, Vec &lmbCluster);

void Generate2DLaplaceNullSpace(Mesh *mesh, bool &isSingular,
		bool &isLocalSingular, Mat *Rmat, MPI_Comm comm = PETSC_COMM_WORLD);

void Generate2DLaplaceTotalNullSpace(Mesh *mesh, bool &isSingular,
		bool &isLocalSingular, Mat *Rmat, MPI_Comm comm = PETSC_COMM_WORLD);

void Generate2DElasticityNullSpace(Mesh *mesh, NullSpaceInfo *nullSpace, MPI_Comm comm = PETSC_COMM_WORLD);

void Generate2DLaplaceClusterNullSpace(Mesh *mesh, SubdomainCluster *cluster);

void getLocalJumpPart(Mat B, Mat *Bloc);

Feti1* createFeti(Mesh *mesh, PetscReal(*f)(Point), PetscReal(*K)(Point),
		MPI_Comm comm);

#endif
