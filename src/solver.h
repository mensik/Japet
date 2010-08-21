/**	@file		solver.h
		@author Martin Mensik
		@date 	2010
		@brief	File containing solvers
*/

#ifndef SOLVER_H
#define SOLVER_H

#include "math.h"
#include "petscmat.h"

typedef bool (*ConvFunc)(PetscInt, PetscScalar, Vec*);
/// Basic Conjugate gradient implementation

class SolverApp {
public:
	virtual void applyMult(Vec in, Vec out) = 0;
};

class SolverCtr {
public:
	virtual bool isConverged(PetscInt itNum, PetscScalar rNorm, Vec *vec) = 0;
};

class CGSolver : public SolverApp, public SolverCtr{
	Mat A;	///< Matrix A in problem A*x = b
	Vec x;  ///< Vector x in problem A*x = b
	Vec b;	///< Vector b in problem A*x = b
	Vec r;	///< residual vector
	Vec p;	///< direction vector
	Vec temp;	///< just template vector
	PetscReal rNorm; ///< residual norm

	SolverApp *sApp;
	SolverCtr *sCtr;

	ConvFunc isCon;	///< function to decide about confergention
	PetscInt itCounter;		///< iteration counter
	void initSolver(Vec b, Vec x);
public:
	CGSolver(Mat A, Vec b, Vec x);
	CGSolver(SolverApp *sa, Vec b, Vec x);
	~CGSolver();
	void setIsConvergedFunc(ConvFunc f) { isCon = f; };	///< set different convergention deciding function
	void setSolverApp(SolverApp *sa) { sApp = sa; };
	void setSolverCtr(SolverCtr *sc) { sCtr = sc; };
	Vec getX() { return x;}	///< @return solution
	void getX(Vec r) { VecCopy(x,r); }	///< @return copy of solution
	void solve();						///< begin solving
	void applyMult(Vec in, Vec out);
	bool isConverged(PetscInt itNum, PetscScalar rNorm, Vec *vec);

};

class MPRGP {
	Mat A;
	Vec x;
	Vec b;
	Vec l;

	PetscReal G;
	PetscReal alp;

	PetscInt localRangeStart;
	PetscInt localRangeEnd;
	PetscInt localRangeSize;

	PetscReal e;

	Vec g;
	Vec p;
	Vec temp;

	PetscInt *localIndices;

	void projectFeas(Vec &v);
	void partGradient(Vec &freeG, Vec &chopG, Vec &rFreeG);
	PetscReal alpFeas();

public:
	MPRGP(Mat A, Vec b, Vec l, Vec x, PetscReal G, PetscReal alp);
	~MPRGP();
	void solve();
};

bool isConFun(PetscInt itNumber, PetscScalar rNorm, Vec *r);
 
#endif
