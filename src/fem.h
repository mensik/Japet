/**	@file		fem.h
		@brief	Declaration for Finite Element Method routines implementation
		@author Martin Mensik
		@date 	2010
*/


/**
	@mainpage
	@section intro_sec Introducion
	Welcome to Japet library. Main reason for development of this library is train my skills
	in C++ programing, enhance my knowledge of PETSc library and basic optimalization algorhytms. 
	I hope it could be used as foundation fo my future experiments as well.

*/
#ifndef FEM_H
#define FEM_H

#include "math.h"
#include "petscmat.h"
#include "structures.h"

 

/**
	@brief Assemble mass matrix A and right side vector b
	@param[in] comm set of proceses to share matrix A and vector b
	@param[in] mesh pointer to mesh
	@param[out] A reference to mass matrix 
	@param[out] b reference to reight side vector
	@param[in] f 	"force" function
	@param[in] K	"material" function  	
*/
PetscErrorCode FEMAssemble2D(MPI_Comm comm, Mesh *mesh, Mat &A, Vec &b, PetscScalar (*f)(Point), PetscScalar (*K)(Point));
/**
	@brief Assemble mass matrix A and right side vector b
	@param[in] comm set of proceses to share matrix A and vector b
	@param[in] mesh pointer to mesh
	@param[out] A reference to mass matrix 
	@param[out] b reference to reight side vector
	@param[in] f 	"force" function
	@param[in] K	"material" function  	
*/
PetscErrorCode FEMAssemble2DLaplace(MPI_Comm comm, Mesh *mesh, Mat &A, Vec &b, PetscScalar (*f)(Point), PetscScalar (*K)(Point));


/**
	@brief Alter (zeros) some members of A and b to enforce Dirchlets boundary
	
 @param[in,out] A 		mass matrix
 @param[in,out] b 		right side vector
 @param[in] n			number of indices
 @param[in] ind		indices of Dirchlet boundary nodes
*/
PetscErrorCode FEMSetDirchletBound(Mat &A, Vec &b, PetscInt n, PetscInt *ind);
#endif
