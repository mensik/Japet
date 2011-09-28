/**	@file		feti.h
 @brief	FETI Method
 @author Martin Mensik
 @date 	2010
 */

#ifndef FETI_H
#define FETI_H

#include <math.h>
#include <map>
#include <set>
#include <string>
#include <sstream>

#include "petscksp.h"
#include "petscmat.h"
#include "japetUtils.h"
#include "fem.h"
#include "solver.h"

class GGLinOp: public SolverApp, public SolverCtr {
	Mat B;
	Mat R;

	Vec temp1, temp2, temp3;
public:
	GGLinOp(Mat B, Mat R);
	virtual void applyMult(Vec in, Vec out, IterationManager *info);
	virtual bool
	isConverged(PetscInt itNumber, PetscReal norm, PetscReal bNorm, Vec *vec);
};

/**
 * @brief Abstract FETI ancestor
 * @note		There is still some work to do. The messiest part of implementation is in dealing with
 object G matrix and computation of the projector P = I - G inv(G'G) G'
 *
 **/

class AFeti: public SolverApp, public SolverCtr, public SolverPreconditioner {
protected:

	PDCommManager *cMan;

	ASolver *outerSolver; ///< Solver class for outer loop (default is CGSolver)
	bool isVerbose;

	//
	// DUAL
	//

	VecScatter tgScat; ///< Scatter from dual group to master. For application of inv G'G
	Vec tgLocIn, tgLocOut;
	Vec tgA;
	Vec tgB;

	CGSolver *ggParSol;
	Vec parT1, parT2;

	Vec lmb; ///< Lambda vector
	Vec d; ///< dual right side
	Vec e;

	PetscInt gM, gN; ///<	dimensions of G
	Mat G, GT; ///< BR
	KSP kspG; ///< Global G'G solver
	MatNullSpace GTGNullSpace;

	VecScatter dBScat; ///< Scatter from dual group to master
	Vec dBGlob; ///< gloval version
	Vec dBLoc; ///< local (on root) version

	//
	// PRIMAL
	//

	Mat BT;
	Mat B; ///< Jump operator matrix
	Vec b; ///< Global force vector
	Vec u; ///< solution
	Mat R; ///< Global null space of A

	SystemR *systemR; ///< NullSpace of the whole system - even with constrains
	MatNullSpace systemNullSpace;

	Vec temp;
	Vec tempLoc;

	VecScatter pBScat; ///< Scatter from primal to primal root
	Vec pBGlob; ///< gloval version
	Vec pBLoc; ///< local (on root) version


	bool isSingular; ///< is Matrix A singular
	bool isLocalSingular; ///< is local part of A singular


	PetscReal lastNorm; ///< last computed norm

	PetscReal precision;

	PetscInt outIterations;
	PetscInt inIterations;

	PetscLogStage coarseStage, coarseInitStage, aInvStage, fetiInitStage;
	CoarseProblemMethod cpMethod;

	void initCoarse();

	void applyInvGTG(Vec in, Vec out);

public:

	//
	// Constants for identification and synchronization during the dual-primal solving
	//
	const static int P_ACTION_INVA = 1;
	const static int P_ACTION_MULTA = 2;
	const static int P_ACTION_BREAK = -1;

	AFeti(PDCommManager *comMan, Vec b, Mat BT, Mat B, Vec lmb,
			NullSpaceInfo *nullSpace, CoarseProblemMethod mcpM = ParaCG,
			SystemR *sR = PETSC_NULL);
	~AFeti();

	virtual void applyInvA(Vec in, IterationManager *itManager) = 0;
	virtual void applyPrimalMult(Vec in, Vec out);
	void solve(Vec b) {
		this->b = b;
		solve();
	}
	virtual void solve();
	virtual void applyMult(Vec in, Vec out, IterationManager *info);
	virtual bool
	isConverged(PetscInt itNumber, PetscReal norm, PetscReal bNorm, Vec *vec);

	virtual ASolver* instanceOuterSolver(Vec d, Vec l);
	virtual void applyPC(Vec g, Vec z) {
		projectGOrth(g);
		VecCopy(g, z);
	}

	void dumpSolution(PetscViewer v);
	void dumpSystem(PetscViewer v);
	void projectGOrth(Vec in); ///< Remove space spaned by G from vec in
	void copySolution(Vec out); /// <Copy solution to vector out
	void copyLmb(Vec out);
	void setPrec(PetscReal prec) {
		precision = prec;
	}
	void setIsVerbose(bool verbose) {
		isVerbose = verbose;
	}
	void saveIterationInfo(const char *fileName) {
		outerSolver->saveIterationInfo(fileName);
	}

	void testSomething();

	void setSystemSingular() {

		PetscPrintf(PETSC_COMM_SELF, "CODE TO WRITE you Moorons!!! setSystemSingular \n");
		//
		// FIX
		//

		//MatNullSpace NS;
		//MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL, &NS);
		//KSPSetNullSpace(kspG, NS);
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

	PetscLogStage aFactorStage;
public:
	Feti1(PDCommManager *comMan, Mat A, Vec b, Mat BT, Mat B, Vec lmb,
			NullSpaceInfo *nullSpace, PetscInt localNodeCount, PetscInt fNodesCount,
			PetscInt *fNodes, CoarseProblemMethod cpM = ParaCG,
			SystemR *sR = PETSC_NULL);
	~Feti1();
	virtual void applyInvA(Vec in, IterationManager *itManager);
	virtual void applyPC(Vec g, Vec z);

	virtual void solve();

	virtual void applyPrimalMult(Vec in, Vec out);
};

class FFeti: public Feti1, public SolverInvertor {
protected:

	VecScatter fToRoot;
	Vec fLocal, fGlobal;

	KSP kspF;
public:
	FFeti(PDCommManager *comMan, Mat A, Vec b, Mat BT, Mat B, Vec lmb,
			NullSpaceInfo *nullSpace, PetscInt localNodeCount, PetscInt fNodesCount,
			PetscInt *fNodes, CoarseProblemMethod cpM = ParaCG,
			SystemR *sR = PETSC_NULL);

	virtual void applyInversion(Vec b, Vec x);

	virtual ASolver* instanceOuterSolver(Vec d, Vec lmb);

//	virtual void solve();

};

class mFeti1: public Feti1 {

public:
	mFeti1(PDCommManager *comMan, Mat A, Vec b, Mat BT, Mat B, Vec lmb,
			NullSpaceInfo *nullSpace, PetscInt localNodeCount, PetscInt fNodesCount,
			PetscInt *fNodes, CoarseProblemMethod cpM = ParaCG) :
				Feti1(comMan, A, b, BT, B, lmb, nullSpace, localNodeCount, fNodesCount, fNodes, cpM) {
	}

	virtual ASolver* instanceOuterSolver(Vec d, Vec lmb);
};

class InexactFeti1: public Feti1 {
	PetscReal outerPrec;
	PetscInt inCounter;
public:
	InexactFeti1(Mat A, Vec b, Mat B, Vec lmb, NullSpaceInfo *nullSpace,
			PetscInt localNodeCount, MPI_Comm comm);

	virtual void applyInvA(Vec in, IterationManager *itManager);
	virtual ASolver* instanceOuterSolver(Vec d, Vec lmb);
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

	Vec tempInv, tempInvGh, tempInvGhB;

	Mat A;
public:
	HFeti(PDCommManager* pdMan, Mat A, Vec b, Mat BGlob, Mat BTGlob, Mat BClust,
			Mat BTClust, Vec lmbGl, Vec lmbCl, SubdomainCluster *cluster,
			PetscInt localNodeCount);

	~HFeti();
	void test();

	virtual void applyInvA(Vec in, IterationManager *itManager);
	virtual void applyPC(Vec g, Vec z);
	virtual void applyPrimalMult(Vec in, Vec out);
	void removeNullSpace(Vec in);
	virtual ASolver* instanceOuterSolver(Vec d, Vec lmb);
	virtual void setRequiredPrecision(PetscReal reqPrecision);
};

void GenerateJumpOperator(Mesh *mesh, Mat &B, Vec &lmb);

void GenerateTotalJumpOperator(Mesh *mesh, int d, Mat &B, Mat &BT, Vec &lmb,
		PDCommManager* commManager);

void GenerateClusterJumpOperator(Mesh *mesh, SubdomainCluster *cluster,
		Mat &BGlob, Mat &BTGlob, Vec &lmbGlob, Mat &BCluster, Mat &BTCluster,
		Vec &lmbCluster, MPI_Comm comm);

void Generate2DLaplaceNullSpace(Mesh *mesh, bool &isSingular,
		bool &isLocalSingular, Mat *Rmat, MPI_Comm comm = PETSC_COMM_WORLD);

void Generate2DLaplaceTotalNullSpace(Mesh *mesh, NullSpaceInfo *nullSpace,
		MPI_Comm comm = PETSC_COMM_WORLD);

void Generate2DElasticityNullSpace(Mesh *mesh, NullSpaceInfo *nullSpace,
		MPI_Comm comm = PETSC_COMM_WORLD);

void Generate2DLaplaceClusterNullSpace(Mesh *mesh, SubdomainCluster *cluster);

void
Generate2DElasticityClusterNullSpace(Mesh *mesh, SubdomainCluster *cluster, MPI_Comm comm);

void getLocalJumpPart(Mat B, Mat *Bloc);

Feti1* createFeti(Mesh *mesh, PetscReal(*f)(Point), PetscReal(*K)(Point),
		MPI_Comm comm);

#endif
