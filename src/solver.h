/**	@file		solver.h
 @author Martin Mensik
 @date 	2010
 @brief	File containing solvers
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <cmath>
#include <vector>
#include "japetUtils.h"
#include "petscmat.h"
#include "petscksp.h"

const double PI = 4.0 * std::atan(1.0);
const double MAXPREC = 1e-6;

class SolverApp {
public:
	virtual void applyMult(Vec in, Vec out, IterationManager* info = NULL) = 0;
	virtual void setRequiredPrecision(PetscReal reqPrecision);
};

class SolverCtr {
public:
	virtual bool isConverged(PetscInt itNum, PetscReal rNorm, PetscReal bNorm,
			Vec *vec) = 0;
};

class SolverPreconditioner {
public:
	virtual void applyPC(Vec g, Vec z) = 0;
};

enum StepType {
	CG, Expansion, Proportion
};

class Solver: public SolverApp, public SolverCtr, public SolverPreconditioner {
private:
	Mat A; ///< Matrix A in problem A*x = b
	PetscReal precision;

	void init();
protected:

	Vec x; ///< Vector x in problem A*x = b
	Vec b; ///< Vector b in problem A*x = b
	Vec g; ///< residual vector
	Vec z; ///< PC vector

	PetscReal rNorm;
	PetscReal bNorm;
	PetscReal r0Norm;

	SolverApp *sApp;
	SolverCtr *sCtr;

	SolverPreconditioner *sPC;
	bool isPCset;

	void nextIteration();

public:
	IterationManager itManager;

	Solver(Mat A, Vec b, Vec x, SolverPreconditioner *PC = NULL);
	Solver(SolverApp *sa, Vec b, Vec x, SolverPreconditioner *PC = NULL);
	virtual ~Solver();

	void setIterationData(std::string name, PetscReal value) {
		itManager.setIterationData(name, value);
	}
	void setIsVerbose(bool isVerbose) {
		itManager.setIsVerbose(isVerbose);
	}
	int getItCount() {
		return itManager.getItCount();
	}
	void applyMult(Vec in, Vec out, IterationManager *info = NULL);
	void applyPC(Vec r, Vec z);
	bool isConverged(PetscInt itNum, PetscReal rNorm, PetscReal bNorm, Vec *vec);
	void setSolverApp(SolverApp *sa) {
		sApp = sa;
	}
	void setSolverCtr(SolverCtr *sc) {
		sCtr = sc;
	}
	void setPrecision(PetscReal precision) {
		this->precision = precision;
	}
	void saveIterationInfo(const char *filename, bool rewrite = true) {
		itManager.saveIterationInfo(filename, rewrite);
	}
	void setTitle(std::string title) {
		itManager.setTitle(title);
	}
	PetscReal getNormG() {
		return rNorm;
	}
	Vec getX() {
		return x;
	} ///< @return solution
	void getX(Vec temp) {
		VecCopy(x, temp);
	} ///< @return copy of solution

	virtual void solve() = 0;
	virtual void reset(Vec newB, Vec newX);
};

class RichardsSolver: public Solver {
	PetscReal alpha;

public:
	RichardsSolver(SolverApp *sa, Vec b, Vec x, PetscReal alpha) :
		Solver(sa, b, x) {
		this->alpha = alpha;
	}
	;

	void solve();
};

class CGSolver: public Solver {

	Vec p; ///< direction vector
	Vec temp; ///< temp Vector

	void initSolver();
public:
	CGSolver(Mat A, Vec b, Vec x) :
		Solver(A, b, x) {
		initSolver();
	}
	;
	CGSolver(SolverApp *sa, Vec b, Vec x, SolverPreconditioner *pc = NULL) :
		Solver(sa, b, x, pc) {
		initSolver();
	}
	;
	~CGSolver();

	void solve(); ///< begin solving
};

class ReCGSolver: public Solver {

	std::vector<Vec> P;
	std::vector<Vec> AP;
	std::vector<PetscReal> PAP;

	Vec p; ///< direction vector
	Vec Ap; ///< temp Vector

	int maxSize;
	PetscInt gCounter;

	void initSolver();
	void project();
	void clearSubspace();
public:

	ReCGSolver(Mat A, Vec b, Vec x, SolverPreconditioner *pc = NULL) :
		Solver(A, b, x, pc) {
		initSolver();
	}
	ReCGSolver(SolverApp *sa, Vec b, Vec x, SolverPreconditioner *pc = NULL) :
		Solver(sa, b, x, pc) {
		initSolver();
	}

	~ReCGSolver();

	void solve();

};

class GLanczos: public Solver {

	PetscInt prevSize;
	Mat Vprev;
	PetscReal *lambda, *mju, *beta;

	MPI_Comm comm;

public:
	static const PetscInt MAXSTEPS = 500;

	GLanczos(MPI_Comm cm, Mat A, Vec b, Vec x) :
		Solver(A, b, x) {
		comm = cm;
		prevSize = 0;
	}
	;
	GLanczos(MPI_Comm cm, SolverApp *sa, Vec b, Vec x) :
		Solver(sa, b, x) {
		prevSize = 0;
		comm = cm;
	}
	;

	void solve();

};

class ASinStep: public Solver {
	PetscReal tau;
	PetscReal fi;
	PetscReal z;
	PetscReal m, M;

	void initSolver();
public:
	ASinStep(Mat A, Vec b, Vec x) :
		Solver(A, b, x) {
		initSolver();
	}
	ASinStep(SolverApp *sa, Vec b, Vec x) :
		Solver(sa, b, x) {
		initSolver();
	}
	void solve();
	PetscReal getRequiredMultPrecision();
};

class MPRGP: public Solver {
	Vec l;

	PetscReal G;
	PetscReal alp;

	PetscInt localRangeStart;
	PetscInt localRangeEnd;
	PetscInt localRangeSize;

	PetscReal e;

	Vec p;
	Vec temp;

	SolverPreconditioner *sPC;

	void projectFeas(Vec &v); //< @param[out] vector with changed infeasible parts to feasible
	void partGradient(Vec &freeG, Vec &chopG, Vec &rFreeG); //< divides gradient into its free, chopped and reduced parts
	PetscReal alpFeas(); //< @return feasible step length for current gradient
	void initSolver(Vec l, PetscReal G, PetscReal alp);
	void pcAction(Vec free, Vec z);

public:
	/**
	 @param[in] A stiffness (mass) matrix
	 @param[in] b force vector
	 @param[in] l "floor" vector, any part of solution x can't be less than according part of l
	 @param[in] G Gamma operator, used to decide about domination of free part of gradient
	 @param[in] alp fixed step length - <0, 2/||A||>
	 */
	MPRGP(Mat A, Vec b, Vec l, Vec x, PetscReal G, PetscReal alp) :
		Solver(A, b, x) {
		initSolver(l, G, alp);
	}
	;
	MPRGP(SolverApp *app, Vec b, Vec l, Vec x, PetscReal G, PetscReal alp) :
		Solver(app, b, x) {
		initSolver(l, G, alp);
	}
	;
	~MPRGP();

	void setPC(SolverPreconditioner *pc) {
		this->sPC = pc;
	}

	void solve();
};

#endif
