/**	@file		structures.h
		@brief	FETI Method
		@author Martin Mensik
		@date 	2010
*/

#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <math.h>
#include <set>
#include <map>
#include "petscmat.h"

/// Struct representing 2D Point

enum PointType {
	INTERMAL,
	BOUNDING,
	CORNER,
	BORDER
};

/**
	@brief Describes sides of boundary, used for example in SDRectSystem::setDirchletBound
*/
enum BoundSide {
	LEFT, 
	RIGHT,
	TOP,
	BOTTOM,
	ALL
};


struct Point {
	PetscScalar x;									///< X coordinate of the node
	PetscScalar y;									///< Y coordinate of the node
	PetscScalar z;								///< Z coordinate of the node
	
	std::set<PetscInt> edges;
	
	Point(PetscScalar xx = 0, PetscScalar yy = 0, PetscScalar zz = 0) {x = xx; y = yy; z = zz;};
	Point(const Point &p) {x = p.x; y = p.y; z = p.z;};
	~Point() {};
};

///Struct representing edge
struct Edge {
	PetscInt vetrices[2];
	std::set<PetscInt> elements;
};

static const int MAX_VETRICES = 3;
/// Struct representing general elements of mesh
struct Element {
	PetscInt numVetrices;
	PetscInt vetrices[MAX_VETRICES];
	PetscInt numEdges;
	PetscInt edges[MAX_VETRICES];
};


/**
	@brief	Complex system to keep information of mesh topology and its distribution
					on proceses.

	@note		There is open way to implementation of ParMetis system to tearing domain.

	@todo		TODO there is some "duality" of this object. Inherited object RectMesh uses 
					former parts of this strcture and it should be unitize.
**/

class Mesh {
	public:
	std::map<PetscInt, Point>	vetrices;		///< Map of vetrices on this processor. TODO Should replace *nodes 
	std::map<PetscInt, Edge> edges;				///< Mat of edges
	std::map<PetscInt, Element> elements;	///< Map of elements on this processor. TODO Should repplace *nodes
	
	std::set<PetscInt> borderEdges;	///< set of border edges

	void generateRectangularMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h); ///< generate rectangular mesh
	void regenerateEdges();	///< Regenerates edges according to vetrices and elements. <b>Destroys previusly generated edges!!! Use with caution</b>
	void findBorders();
	PetscInt getEdge(PetscInt nodeA, PetscInt nodeB); ///< -1 if it doesn't exist. 
	void dumpForMatlab(PetscViewer v);
};

/**
	@brief	Extract local part from global matrix A. Most likely mesh.localVetriceSet will work best for
					this purpose.

	@param[in] A global matrix
	@param[in] vetrices set of matrix rows x cols to extract on this procesor
	@param[out] Aloc local matrix (user is responsible for destroy) 
**/
void extractLocalAPart(Mat A, std::set<PetscInt> vetrices, Mat *Aloc); 

#endif
