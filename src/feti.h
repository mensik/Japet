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
#include "fem.h"
#include "solver.h"

class AFeti: public SolverApp, public SolverCtr {
protected:

	Solver *outerSolver;

	Mat B;
	Vec lmb;
	Vec u;
	Vec d;

	Vec b; ///< Global force vector
	Vec bloc; ///< Ghosted local force vector
	Vec uloc; ///< Ghosted local solution

	KSP kspG; ///< Global G'G solver

	bool isSingular; ///< is Matrix A singular
	bool isLocalSingular; ///< is local part of A singular
	Mat R; ///< Global null space of A
	Mat G; ///< BR
	PetscInt gM, gN; ///<	dimensions of G

	PetscReal lastNorm;

	Vec tgA;
	Vec tgB;

	Vec temp;
	Vec tempLoc;
	Vec tempLocB;

	MPI_Comm comm; ///< Communication channel of processes

public:
	AFeti(Vec b, Mat B, Vec lmb, Laplace2DNullSpace *nullSpace,
			PetscInt localNodeCount, MPI_Comm comm);
	~AFeti();

	virtual void applyInvA(Vec in) = 0;
	void solve(Vec b) {
		this->b = b;
		solve();
	}
	virtual void solve();
	virtual void applyMult(Vec in, Vec out);
	virtual bool
	isConverged(PetscInt itNumber, PetscReal norm, PetscReal bNorm, Vec *vec);

	virtual Solver* instanceOuterSolver(Vec d, Vec l);

	void dumpSolution(PetscViewer v);
	void dumpSystem(PetscViewer v);
	void projectGOrth(Vec in); ///< Remove space spaned by G from vec in
	void copySolution(Vec out); /// <Copy solution to vector out
};

/**
 @brief	Rather general (and functional) implementation of FETI-1 algorythm. Tearing of global
 domain and distribution of subdomains is done in preprocesing of Mesh object.

 Currently, CG implemenation in solver.h is used for outer cycle and Petsc direct solver
 for inner cycle.

 @note		There is still some work to do. The messiest part of implementation is in dealing with
 object G matrix and computionfg of projector P = I - G inv(G'G) G'
 **/

class Feti1: public AFeti {
protected:
	Mat A; ///< Global stifles matrix
	Mat Aloc; ///< Local part of stiffness matrix
	KSP kspA; ///< Local A solver

	MatNullSpace locNS; ///< local null space of local A


	Vec tempInv, tempInvGh, tempInvGhB;
public:
	Feti1(Mat A, Vec b, Mat B, Vec lmb, Laplace2DNullSpace *nullSpace,
			PetscInt localNodeCount, MPI_Comm comm);
	~Feti1();
	virtual void applyInvA(Vec in);
};

class InexactFeti1: public Feti1 {
	PetscReal outerPrec;
	PetscInt inCounter;
public:
	InexactFeti1(Mat A, Vec b, Mat B, Vec lmb, Laplace2DNullSpace *nullSpace,
			PetscInt localNodeCount, MPI_Comm comm);

	virtual void applyInv(Vec in);
	virtual Solver* instanceOuterSolver(Vec d, Vec lmb);
	void setRequiredPrecision(PetscReal reqPrecision);
};

class HFeti: public AFeti {
	AFeti *subClusterSystem;
	SubdomainCluster *cluster;

	Vec clustTemp, clustTempGh;
	Vec clustb;
	Laplace2DNullSpace *clustNullSpace;
	MatNullSpace clusterNS;

	Vec globTemp, globTempGh;
public:
	HFeti(Mat A, Vec b, Mat BGlob, Mat BClust, Vec lmbGl, Vec lmbCl,
			SubdomainCluster *cluster, PetscInt localNodeCount, MPI_Comm comm);

	virtual void applyInvA(Vec in);
	void removeNullSpace(Vec in);
};

void GenerateJumpOperator(Mesh *mesh, Mat &B, Vec &lmb);
void GenerateClusterJumpOperator(Mesh *mesh, SubdomainCluster *cluster,
		Mat &BGlob, Vec &lmbGlob, Mat &BCluster, Vec &lmbCluster);
void Generate2DLaplaceNullSpace(Mesh *mesh, bool &isSingular,
		bool &isLocalSingular, Mat *Rmat, MPI_Comm comm = PETSC_COMM_WORLD);
void Generate2DLaplaceClusterNullSpace(Mesh *mesh, SubdomainCluster *cluster);
void getLocalJumpPart(Mat B, Mat *Bloc);

Feti1* createFeti(Mesh *mesh, PetscReal(*f)(Point), PetscReal(*K)(Point),
		MPI_Comm comm);

#endif
