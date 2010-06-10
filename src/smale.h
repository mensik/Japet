/**	@file		smale.h
		@author Martin Mensik
		@date 	2010
		@brief	File containing Smale(like) domain decomposition methods and structures
*/

#ifndef SMALE_H
#define SMALE_H

#include <math.h>

#include "petscmat.h"
#include "fem.h"
#include "solver.h"
/** 
		@brief SDRectSystem represents structure of subdomains and their mutual bounds
		

		@note Expresion "local" is used to point out structures not shared by all
					proceses, but rather by subset (most often by only one) of them.
*/
class SDRectSystem {
	PetscInt subMeshCount;///< number of subdomains in system
	DomainRectLayout *layout; ///< layout description
	PetscInt localIndex;	///< "local" subdomain index
	Mat A;								///< "local" Mass matrix
	Vec b;								///< "local" right side vector
	
	RectMesh **subMesh; 		///< all meshes
	Mat B;								///< jump operator matrix
	Vec c;								///< jump operator vector
public:
	SDRectSystem(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h, PetscInt xSize, PetscInt ySize, PetscScalar (*f)(Point), PetscScalar (*K)(Point));
	~SDRectSystem();

	/**
		@brief Defines which sides of the whole domain should have Dirchlet condition
		@param[in] n number of sides
		@param[in] sides array of sides
	*/
	void setDirchletBound(PetscInt n, BoundSide *sides);
	/**
		@return return index of current processes subdomain 
	*/
	PetscInt getLocalIndex() { return localIndex; }
	Mat getA() { return A; }	///< @return mass matrix A of local subdomain
	Vec getb() { return b; }	///< @return right side vector b of local subdomain
	Vec getc() { return c; }	///< @return vector c of Jump operator
	Mat getB() { return B; }	///< @return Jump operator B matrix
};

class Smale {
	static const PetscInt MAX_OUT_IT = 100;
	SDRectSystem	*sd;

	Vec	x;
	Vec	l;
	PetscReal mi;
	PetscReal ro;
	PetscReal beta;
	PetscReal M;

	Vec tempMSize,bxc;
	Vec g,p,temp;
	Vec lx,lg,lp,ltemp;

	PetscReal gNorm;
	PetscInt outItCount;
	PetscInt inItCount[MAX_OUT_IT];

	PetscReal aL;
	PetscReal prevL;
	PetscReal lPrec;

	void refreshGradient();
	void updateLagrange();
	bool isInerConverged();
	bool isOuterConverged();
	PetscReal	L();
public:
	Smale(SDRectSystem *sd, PetscReal mi = 1e-3, PetscReal ro = 5, PetscReal beta = 1.1, PetscReal M = 0.1);
	~Smale();
	void solve();
	void dump(PetscViewer v);
	void dumpSolution(PetscViewer v);
	Vec getx() { return x; }
	Vec getl() { return l; }
};

class SassiRectSystem {
	PetscInt subMeshCount;///< number of subdomains in system
	DomainRectLayout *layout; ///< layout description
	PetscInt localIndex;	///< "local" subdomain index
	RectMesh	*mesh;				///< "local" mesh
	

	Mat A;								///< "local" Mass matrix
	Vec b;								///< "local" right side vector
	Vec x_loc;						///< "local" ghost of solution vector
	PetscInt	numB;				///< "local" num of shared borders
	PetscInt	*indQ;			///< "local" indexes of shared borders
	Mat *B;								///< "local" mapings on shared borders
	Vec	*lag;							///< "local" lagrangian
	Vec *ghQ;							///< "local" ghost of border
	//Vec *ghRes;						///< "local" ghosted vector for residual estimation
		
	Vec x;								///< solution
	PetscInt numQ;				///< "global" total number of shared borders
	Vec *q;								///< Borders
	Vec *borderRes;				///< residual estimation

	PetscReal	r;
	
	void prepareBorder(Mat &B, Vec &lag, PetscInt numNodes, PetscInt borderLenght, PetscInt *indexes);
public:
	SassiRectSystem(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h, PetscInt xSize, PetscInt ySize, PetscScalar (*f)(Point), PetscScalar (*K)(Point), PetscReal r);
	~SassiRectSystem();

	void setDirchletBound(PetscInt n, BoundSide *sides);
	void solve();
	void dumpSolution(PetscViewer v);
};

#endif
