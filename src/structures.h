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
#include <vector>
#include <cstdio>
#include "petscmat.h"
#include "petscao.h"
#include "parmetis.h"

extern "C" {
	#include "metis.h"
}


class MyMultiMap {
public:
	PetscInt numOfPoints;
	std::map<PetscInt, std::map<PetscInt, PetscInt> > data;
	MyMultiMap() {
		numOfPoints = 0;
	}

	PetscInt getNumOfPoints() {
		return numOfPoints;
	}

	PetscInt getNewPointId(PetscInt oldPointId, PetscInt domainId);
	void saveNewPoint(PetscInt oldPoint, PetscInt domainId, PetscInt newPoint);
};

class DomainPairings {
public:
	std::map<PetscInt, std::map<PetscInt, std::vector<PetscInt> > > data;
	void insert(PetscInt domA, PetscInt domB, PetscInt* pair);
	void getPairs(PetscInt domA, PetscInt domB, std::vector<PetscInt>::iterator &begin, std::vector<PetscInt>::iterator &end);
};

class SubdomainCluster {
public:
	MPI_Comm clusterComm;
	PetscInt clusterColor;

	//Root only
	PetscInt *subdomainColors;
	std::vector<PetscInt> globalPairing;

	//cluster Roots
	std::vector<PetscInt> localPairing;

	//cluster Roots

	std::map<PetscInt, PetscInt> startIndexesDiff;
};


static const int MAX_VETRICES = 3;
static const int MAX_CORNER_SIZE = 6;
/// Struct representing general elements of mesh
struct Element {
	PetscInt id;
	PetscInt numVetrices;
	PetscInt vetrices[MAX_VETRICES];
	PetscInt numEdges;
	PetscInt edges[MAX_VETRICES];
};

///Struct representing edge
struct Edge {
	PetscInt id;
	PetscInt domainInd;
	PetscInt vetrices[2];
	std::set<PetscInt> elements;
};

/// Struct representing Point
struct Point {
	PetscReal x; ///< X coordinate of the node
	PetscReal y; ///< Y coordinate of the node
	PetscReal z; ///< Z coordinate of the node

	std::set<Element*> elements;
	std::set<Edge*> edges;

	PetscInt domainInd;

	Point(PetscReal xx = 0, PetscReal yy = 0, PetscReal zz = 0) {
		x = xx;
		y = yy;
		z = zz;
		domainInd = 0;
	}
	;
	Point(const Point &p) {
		x = p.x;
		y = p.y;
		z = p.z;
	}
	;
	~Point() {
	}
	;
};

/// struct representing corner - node shared by more than two domains
struct Corner {
	PetscInt cornerSize;
	PetscInt vetrices[MAX_CORNER_SIZE];
};

/**
 @brief	Complex system to keep information of mesh topology and its distribution
 on processes. Main drag of this implementation is that whole mesh is
 stored and processed by just one processor.

 @note		There is open way to implementation of ParMetis system to tearing domain.
 **/

class Mesh {
public:
	std::map<PetscInt, Point*> vetrices; ///< Map of vetrices
	std::map<PetscInt, Edge*> edges; ///< Mat of edges
	std::map<PetscInt, Element*> elements; ///< Map of elements
	std::set<PetscInt> borderEdges; ///< set of border edges
	PetscInt nPairs;

	//Root only
	PetscInt *pointPairing; ///< array with pairs of neighbor vetrices
	std::vector<PetscInt*> borderPairs; ///< vector of pairs (array length 2), pairs of neighbor edges
	std::vector<Corner*> corners; ///< vector of indices to corners - grouped by corner
	PetscInt *startIndexes;

	void linkPointsToElements(); ///<Add element pointer to points
	void regenerateEdges(); ///< Regenerates edges according to vetrices and elements
	void findBorders();
	PetscInt getEdge(PetscInt nodeA, PetscInt nodeB); ///< -1 if it doesn't exist.

	bool isPartitioned;
	PetscInt numOfPartitions; ///< number of partitions, obviously ;-)
	idxtype *epart; ///< elements domain indexes
	Mesh() {
		isPartitioned = false;
	}
	~Mesh();
	int getNumElements() {
		return elements.size();
	}
	int getNumNodes() {
		return vetrices.size();
	}

	void generateRectangularMesh(PetscReal m, PetscReal n, PetscReal k,
			PetscReal l, PetscReal h); ///< generate rectangular mesh
	void dumpForMatlab(PetscViewer v);
	void save(const char *filename, bool withEdges);
	void load(const char *filename, bool withEdges);
	PetscErrorCode partition(int numDomains); ///< call Metis and divide elements among processes, only marks elements
	/**
	 * @brief Actually tear mesh to part, make new points and edges a distribute them among processes  by MPI
	 *
	 * This complex function has two main phases:
	 * 1. tearing - mesh is teared to subdomains, edges and vertices are duplicated and reconected. Master gathers informations about
	 * 	interdomain bounds and finds corners;
	 *
	 * 2. distribution - master keeps just his part of mesh and distribute all other parts to slaves
	 */
	void tear();

	/**
	 * @param[in] f function to evaluate
	 * @param[out] fv vector with values of f in nodes
	 */
	void evalInNodes(PetscReal(*f)(Point), Vec *fv);

	void createCluster(SubdomainCluster *cluster);

	PetscInt getNodeDomain(PetscInt index);
};


/**
 @brief	Extract local part from global matrix A. Most likely will work best for
 this purpose.

 @param[in] 	A global matrix
 @param[out] 	Aloc local matrix (user is responsible for destroy)
 **/
void extractLocalAPart(Mat A, Mat *Aloc);

#endif
