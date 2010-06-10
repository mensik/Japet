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
	Point(PetscScalar xx = 0, PetscScalar yy = 0, PetscScalar zz = 0) {x = xx; y = yy; z = zz;};
	Point(const Point &p) {x = p.x; y = p.y; z = p.z;};
	~Point() {};
};

/// Struct representing element of mesh
struct Element2D {
	PetscInt nodes[3];										///< indices of nodes of element
	Point getCentroid(Point *nodes);			///< Returns centroid of triangular element 
};

/// Struct representing general elements of mesh
struct Element {
	std::set<PetscInt> vetrices;
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
	PetscInt numElements;			///<Number of all elements in mesh
	PetscInt numPoints;				///<Number of all nodes in mesh

	PetscInt mlocal_nodes;		///< Number of local nodes
	PetscInt mlocal_elements;	///< Number of local elements
	Point *nodes;							///<Array of local nodes
	std::map<PetscInt, Point>	vetrices;	///< Map of vetrices on this processor. TODO Should replace *nodes 
	std::set<PetscInt> localVetricesSet;///< Set of localy managed vetrices. Map vetrices can possibly contain ghosted vetrices
	Element2D *elements;			///<Array of local elements
	std::map<PetscInt, Element> element;	///< Map of elements on this processor. TODO Should repplace *nodes

	PetscInt n_pairings;			///< Number of point pairings
	bool	keepPairing;				///< Local information, if Pairings are stored on current procesor
	PetscInt *pointPairings;	///<Array of all paired points

	std::set<PetscInt> indDirchlet;
	std::set<PetscInt> indDual;
	std::set<PetscInt> indPrimal;
	std::multimap<PetscInt, PetscInt>	indPrimalBound;
		
	Mesh(PetscInt mlocal_elements, PetscInt mlocal_nodes); ///<Basic constructor, only allocate memory
	Mesh(PetscInt mlocal_elements, PetscInt mlocal_nodes, PetscInt num_pairings); ///<Basic constructor, only allocate memory
	~Mesh() { delete[] nodes; delete[] elements; if (keepPairing) delete[] pointPairings;};

	void dumpForMatlab(PetscViewer v);
};

/// Rectangular mesh structure
class RectMesh : public Mesh {
public:
	PetscInt xPoints;			///<Number of nodes along axis X
	PetscInt yPoints;			///<Number of nodes along axis Y
	PetscInt *iL;					///<Array of indexes on left border of mesh
	PetscInt *iR;					///<Array of indexes on right border of mesh
	PetscInt *iT;					///<Array of indexes on top border of mesh
	PetscInt *iB;					///<Array of indexes on bottom border of mesh

/**
	 @param[in] m		x coords of begining	
	 @param[in] n		x coords of end
	 @param[in] k		y coords of begining
	 @param[in] l		y coords of end
	 @param[in] h		lenght of step of discretication mesh
*/

	RectMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h);
	~RectMesh();
protected:
	///Static function to determine number of nodes in mesh
	static PetscInt nPoints(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h){
		PetscInt xEdges = (PetscInt)ceil((n - m) / h);
		PetscInt yEdges = (PetscInt)ceil((l - k) / h);
		return (xEdges + 1)*(yEdges + 1);
	};
	///Static function to determine number of elements in mesh
	static PetscInt nElements(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h){
		PetscInt xEdges = (PetscInt)ceil((n - m) / h);
		PetscInt yEdges = (PetscInt)ceil((l - k) / h);
		return xEdges*yEdges*2; 
	};
};

///Rectangular grid structure used for generating rectangular Meshes
class RectGrid {
public:
	PetscInt numElements;			///<Number of all elements in mesh
	PetscInt numPoints;				///<Number of all nodes in mesh

	Point *nodes;							///<Array of local nodes
	Element2D *elements;			///<Array of local elements

	PetscInt xPoints;			///<Number of nodes along axis X
	PetscInt yPoints;			///<Number of nodes along axis Y
	PetscInt *iL;					///<Array of indexes on left border of mesh
	PetscInt *iR;					///<Array of indexes on right border of mesh
	PetscInt *iT;					///<Array of indexes on top border of mesh
	PetscInt *iB;					///<Array of indexes on bottom border of mesh

/**
	 @param[in] m		x coords of begining	
	 @param[in] n		x coords of end
	 @param[in] k		y coords of begining
	 @param[in] l		y coords of end
	 @param[in] h		lenght of step of discretication mesh
*/

	RectGrid(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h);
	~RectGrid();
protected:
	///Static function to determine number of nodes in mesh
	static PetscInt nPoints(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h){
		PetscInt xEdges = (PetscInt)ceil((n - m) / h);
		PetscInt yEdges = (PetscInt)ceil((l - k) / h);
		return (xEdges + 1)*(yEdges + 1);
	};
	///Static function to determine number of elements in mesh
	static PetscInt nElements(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h){
		PetscInt xEdges = (PetscInt)ceil((n - m) / h);
		PetscInt yEdges = (PetscInt)ceil((l - k) / h);
		return xEdges*yEdges*2; 
	};

};

/**
	@brief	

	@param[in] m		x coords of begining	
	@param[in] n		x coords of end
	@param[in] k		y coords of begining
	@param[in] l		y coords of end
	@param[in] h		lenght of step of discretication mesh
	@param[in] xSize number of subdomains along axis x
	@param[in] ySize number of subdomains along axis y
	@param[in] n_dirchletSidex	number of dirchlet bounded sides
	@param[in] dirchletBounds	array of dirchlet bounded sides
	@param[out]	newly constructed mesh
**/
void generateRectangularTearedMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h, PetscInt xSize, PetscInt ySize,PetscInt n_dirchletSides, BoundSide dirchletBounds[], Mesh **mesh);

/**
	@brief	holds information about bounding of two nodes (a,b)
*/
struct NodeBound {
	PetscInt aDomain;	///< a-node domain index
	PetscInt bDomain;	///< b-node domain index
	PetscInt aNode;		///< a-node local index
	PetscInt bNode;		///< b-node local index
};

/**
	@brief	DomainRectLayout describes position of subdomains in rectangular case
*/
class DomainRectLayout {
	PetscInt	xSize;					///< number of subdomains along x axis
	PetscInt	ySize;					///< number of subdomains along y axis
	PetscInt	*subDomains;		///< array of subdomain indexes 
public:
	DomainRectLayout(PetscInt xSize, PetscInt ySize); 
	~DomainRectLayout() { delete [] subDomains; }

	PetscInt getXSize() { return xSize; }		///< returns number of subdomains along x axis
	PetscInt getYSize() { return ySize; }		///< returns number of subdomains along y axis
	/**
		@param[in] x x coord of subdomain
		@param[in] y y coord of subdomain
		@return	index of subdomain
	*/
	PetscInt getSub(PetscInt x, PetscInt y) { return subDomains[y*xSize + x]; }
	/**
		@param[in] ind index of subdomain
		@param[out] x coord of subdomain
		@param[out] y coord of subdomain
	*/
	void getMyCoords(PetscInt ind, PetscInt &x, PetscInt &y);
};


#endif
