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

/**
	@brief	Rather general (and functional) implementation of FETI-1 algorythm. Tearing of global
					domain and distribution of subdomains is done in preprocesing of Mesh object.

					Currently, CG implemenation in solver.h is used for outer cycle and Petsc direct solver
					for inner cycle.

	@note		There is still some work to do. The messiest part of implementation is in dealing with 
					object G matrix and computionfg of projector P = I - G inv(G'G) G'
**/

class Feti1 : public SolverApp, public SolverCtr {
protected:
	Mat A;			///< Global stifnes matrix
	Vec b;			///< Global force vector
	Mat B;			///< Jump operator matrix
	Vec lmb;		///< Lagrangian vector
	Vec u;			///< Solution
	
	Mat Aloc;		///< Local part of stifness matrix
	Vec bloc;		///< Ghosted local force vector
	Vec uloc;		///< Ghosted local solution

	KSP kspA;		///< Local A solver
	KSP kspG;		///< Global G'G solver

	bool isSingular;	///< is Matrix A singular
	bool isLocalSingular;	///< is local part of A singular
	Mat R;		///< Global null space of A
	Mat G;		///< BR
	PetscInt gM,gN; ///<	dimensions of G
	MatNullSpace locNS; ///< local null space of local A

	PetscReal lastNorm;
	
	Vec temp;
	Vec tempLoc;
	Vec tempLocB;

	Vec tgA;
	Vec tgB;
public:
	Feti1(Mesh *mesh,PetscReal (*f)(Point), PetscReal (*K)(Point));
	~Feti1();
	virtual void solve(); ///< solve the system ;-)
	void dumpSolution(PetscViewer v);
	void dumpSystem(PetscViewer v);
	virtual void applyMult(Vec in, Vec out); ///< Apply multiplication in outer (dual) CG iteration
	void projectGOrth(Vec in);			 ///< Remove space spaned by G from vec in
	bool isConverged(PetscInt itNumber, PetscReal norm, PetscReal bNorm, Vec *vec);
};

class InexactFeti1: public Feti1{
	PetscReal outerPrec;
	Solver *solver;
	PetscInt inCounter;
public:
	InexactFeti1(Mesh *mesh,PetscReal (*f)(Point), PetscReal (*K)(Point)) :  Feti1(mesh, f,K) {}
	void solve();
	void  applyMult(Vec in, Vec out);
};

void GenerateJumpOperator(Mesh *mesh,Mat &B, Vec &lmb);
void Generate2DLaplaceNullSpace(Mesh *mesh,bool &isSingular,bool &isLocalSingular, Mat *Rmat);
void getLocalJumpPart(Mat B, Mat *Bloc);

#endif
