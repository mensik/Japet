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

class SolverProjector {
public:
	virtual void applyProjection(Vec v, Vec vp) = 0;
};

class SolverInvertor {
public:
	virtual void applyInversion(Vec b, Vec x) = 0;
};

enum StepType {
	CG, Expansion, Proportion
};

class ASolver {
protected:
	SolverCtr *sCtr;
	SolverProjector *sProj;
	SolverPreconditioner *sPC;

public:
	ASolver();

	virtual void solve(Vec b, Vec x) = 0;

	virtual void setIsVerbose(bool verbose) {

	}

	virtual void saveIterationInfo(const char *filename, bool rewrite = true) {
		//Has no meaning for factorization
	}

	void setSolverCtr(SolverCtr *sc) {
		sCtr = sc;
	}

	void setPreconditioner(SolverPreconditioner *sp) {
		sPC = sp;
	}

	void setProjector(SolverProjector *sp) {
		sProj = sp;
	}
};

class FinitSolverStub: public ASolver {

	SolverInvertor *invertor;
public:

	FinitSolverStub(SolverInvertor *inv) {
		this->invertor = inv;
	}

	virtual void solve(Vec b, Vec x) {
		invertor->applyInversion(b, x);
	}

};

class Solver: public ASolver,
		public SolverApp,
		public SolverCtr,
		public SolverPreconditioner {
private:
	Mat A; ///< Matrix A in problem A*x = b
	PetscReal precision;

protected:

	SolverApp *sApp;

	MPI_Comm comm;

	PetscReal rNorm, r0Norm, bNorm;

	void nextIteration();
	void init();

public:
	IterationManager itManager;

			Solver(Mat A, SolverPreconditioner *PC = NULL,
					MPI_Comm comm = MPI_COMM_WORLD);
	Solver(SolverApp *sa, SolverPreconditioner *PC = NULL,
			MPI_Comm comm = MPI_COMM_WORLD);
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

	void setPrecision(PetscReal precision) {
		this->precision = precision;
	}
	void saveIterationInfo(const char *filename, bool rewrite = true) {
		itManager.saveIterationInfo(filename, rewrite);
	}
	void setTitle(std::string title) {
		itManager.setTitle(title);
	}

	virtual void solve(Vec b, Vec x) = 0;
};

class CGSolver: public Solver {

	int restartRate;

	void initSolver();
public:
	CGSolver(Mat A) :
		Solver(A) {
	}

	CGSolver(SolverApp *sa, SolverPreconditioner *pc = NULL,
			MPI_Comm comm = MPI_COMM_WORLD, int restartRate = -1) :
		Solver(sa, pc, comm) {
		this->restartRate = restartRate;
	}

	~CGSolver();

	void solve(Vec b, Vec x); ///< begin solving
};

class BBSolver: public Solver {

	void projectedMult(Vec in, Vec out);

public:
	BBSolver(Mat A) :
		Solver(A) {
	}

	BBSolver(SolverApp *sa, SolverPreconditioner *pc = NULL,
			MPI_Comm comm = MPI_COMM_WORLD) :
		Solver(sa, pc, comm) {
	}

	void solve(Vec b, Vec x); ///< begin solving
};
/*
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

 SolverProjector *solProj;

 ReCGSolver(Mat A, Vec b, Vec x, SolverPreconditioner *pc = NULL) :
 Solver(A, b, x, pc) {
 initSolver();
 }
 ReCGSolver(SolverApp *sa, Vec b, Vec x, SolverPreconditioner *pc = NULL,
 SolverProjector *proj = NULL) :
 Solver(sa, b, x, pc) {
 solProj = proj;
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
 */
#endif
