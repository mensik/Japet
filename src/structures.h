/**	@file		structures.h
 @brief	FETI Method
 @author Martin Mensik
 @date 	2010
 */

#ifndef MESGERR
#define MESGERR 1
#endif

#ifndef STRUCTURES_H
#define STRUCTURES_H

#include "med.h"
#include "med_utils.h"
#include <math.h>
#include <set>
#include <map>
#include <vector>
#include <cstdio>
#include "petscmat.h"
#include "petscao.h"
#include "parmetis.h"

/**
 * @brief Keeps information needed for FETI about null space
 **/
struct NullSpaceInfo {
	int localDimension;
	bool isSubDomainSingular; ///< is sub-part associated with current process singular
	bool isDomainSingular; ///< is there any singular sub-part
	Mat R; ///< matrix with null space basis
	Vec *localBasis;
};

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

/**
 *	@brief Helper class for analysis of sub-domain relations
 **/
class DomainPairings {
public:
	std::map<PetscInt, std::map<PetscInt, std::vector<PetscInt> > > data;
	void insert(PetscInt domA, PetscInt domB, PetscInt* pair);
	void getPairs(PetscInt domA, PetscInt domB,
			std::vector<PetscInt>::iterator &begin,
			std::vector<PetscInt>::iterator &end);
};

/**
 * @brief Class holding all information related to sub-domain clustering, including pairings
 **/
class SubdomainCluster {
public:
	MPI_Comm clusterComm; ///< Communication channel [cluster]
	PetscInt clusterColor; ///< Cluster color [cluster]
	PetscInt clusterCount; ///< Global number of clusters [global]

	//Information required by FETI
	bool isSubDomainSingular; ///< Is local domain singular [local]
	bool isClusterSingular; ///< Is any domain in cluster singular [cluster]
	bool isDomainSingular; ///< Is any domain singular [global]

	PetscInt indexDiff; ///< Cluster - global index difference [local]

	NullSpaceInfo *outerNullSpace; ///< Domain outer null space info for FETI [global]
	Mat Rin; ///< Cluster null space basis [cluster]

	//Root only
	PetscInt *subdomainColors; ///< array with all process cluster colors
	std::vector<PetscInt> globalPairing; ///< paired extra-cluster nodes [root]

	//cluster Roots
	std::vector<PetscInt> localPairing; ///< paired inter-cluster nodes [cluster roots]

	//cluster Roots
	std::map<PetscInt, PetscInt> startIndexesDiff; ///< array with cluster-global node index differences [cluster roots]
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

	PetscInt family; ///< contains information about node border etc.
	PetscInt domainInd;

	Point(PetscReal xx = 0, PetscReal yy = 0, PetscReal zz = 0) {
		x = xx;
		y = yy;
		z = zz;
		domainInd = 0;
		family = 0;
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
	/**
	 * @param[in] bounded array with information about bounding sided [U, L, B, R]
	 */
	void generateTearedRectMesh(PetscReal x0, PetscReal x1, PetscReal y0,
			PetscReal y1, PetscReal h, PetscInt m, PetscInt n, bool *bounded);
	void dumpForMatlab(PetscViewer v);
	void save(const char *filename, bool withEdges);
	void saveHDF5(const char *filename);
	void load(const char *filename, bool withEdges);
	void loadHDF5(const char* filename);
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

	/**
	 * @brief Divide already teared mesh to clusters
	 *
	 * @param[out] cluster cluster object
	 */
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
