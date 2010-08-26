/**	@file		fem.h
		@brief	Declaration for Finite Element Method routines implementation
		@author Martin Mensik
		@date 	2010
*/


/**
	@mainpage
	@section intro_sec Introduction
	Welcome to Japet library. Main reason for development of this library is train my skills
	in C++ programming, enhance my knowledge of PETSc library and basic optimalization algorhitms.
	I hope it could be used as foundation for my future experiments as well.

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

PetscErrorCode FEMAssemble2DLaplace(MPI_Comm comm, Mesh *mesh, Mat &A, Vec &b, PetscReal (*f)(Point), PetscReal (*K)(Point));

#endif
