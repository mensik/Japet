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
#include <cstdio>
#include "petscmat.h"

extern "C" {
	#include "metis.h"
}

static const int MAX_VETRICES = 3;
/// Struct representing general elements of mesh
struct Element {
	PetscInt numVetrices;
	PetscInt vetrices[MAX_VETRICES];
	PetscInt numEdges;
	PetscInt edges[MAX_VETRICES];
};

///Struct representing edge
struct Edge {
	PetscInt id;
	PetscInt vetrices[2];
	std::set<PetscInt> elements;
};

/// Struct representing Point
struct Point {
	PetscScalar x;									///< X coordinate of the node
	PetscScalar y;									///< Y coordinate of the node
	PetscScalar z;								///< Z coordinate of the node
	
	std::set<Element*> elements;
	std::set<Edge*> edges;

	PetscInt domainInd;
	
	Point(PetscScalar xx = 0, PetscScalar yy = 0, PetscScalar zz = 0) {x = xx; y = yy; z = zz; domainInd = 0; };
	Point(const Point &p) {x = p.x; y = p.y; z = p.z;};
	~Point() {};
};

struct SubDomain {
	std::map<PetscInt, Point> vetrices;
	std::map<PetscInt, Element> elements;
	std::set<PetscInt>	dirchletBorder;
};

/**
	@brief	Complex system to keep information of mesh topology and its distribution
					on proceses.

	@note		There is open way to implementation of ParMetis system to tearing domain.
**/

class Mesh {
	std::map<PetscInt, Point>	vetrices;		///< Map of vetrices 
	std::map<PetscInt, Edge> edges;				///< Mat of edges
	std::map<PetscInt, Element> elements;	///< Map of elements 
	std::set<PetscInt> borderEdges;	///< set of border edges
	void linkPointsToElements(); ///<Add element pointer to points
	void regenerateEdges();	///< Regenerates edges according to vetrices and elements
	void findBorders();
	PetscInt getEdge(PetscInt nodeA, PetscInt nodeB); ///< -1 if it doesn't exist.

	bool isPartitioned;
	idxtype *epart;
public:
	Mesh() { isPartitioned = false; }
	~Mesh() {if (isPartitioned) delete [] epart; }
	int getNumElements() { return elements.size(); }
	int getNumNodes() { return vetrices.size(); }

	void generateRectangularMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h); ///< generate rectangular mesh
	void dumpForMatlab(PetscViewer v);
	void save(const char *filename, bool withEdges);
	void load(const char *filename, bool withEdges);
	void partition(int numDomains);
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
