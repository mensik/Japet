#include "structures.h"
 
/**
	@param[in] nodeList List of all nodes (referenced by element, for example)
	@return Centroid of this element
*/
Point Element2D::getCentroid(Point *nodeList) {
	return Point((nodeList[nodes[0]].x + nodeList[nodes[1]].x + nodeList[nodes[2]].x)/3.0,
							 (nodeList[nodes[0]].y + nodeList[nodes[1]].y + nodeList[nodes[2]].y)/3.0);
}

Mesh::Mesh(PetscInt nPoints, PetscInt nElements) {
	numPoints = nPoints;
	numElements = nElements;
	nodes = new Point[numPoints];
	elements = new Element2D[numElements];
	keepPairing = false;
	//pointPairings = new PetscInt[1];
}

Mesh::Mesh(PetscInt mlocal_elements, PetscInt mlocal_nodes, PetscInt num_pairings){
	this->mlocal_elements = mlocal_elements;
	this->mlocal_nodes = mlocal_nodes;
	n_pairings = num_pairings;
	nodes = new Point[mlocal_nodes];
	elements = new Element2D[mlocal_elements];
	keepPairing = false;
	//pointPairings = new PetscInt[num_pairings * 2];
}

void generateRectangularTearedMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h, PetscInt xSize, PetscInt ySize,PetscInt n_dirchletSides, BoundSide dirchletBounds[], Mesh **mesh) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	if (!rank) {
	PetscInt subMeshCount = xSize * ySize;
	DomainRectLayout *layout = new DomainRectLayout(xSize,ySize);

	PetscReal xWidth = (n - m) / (PetscReal)xSize;
	PetscReal yWidth = (l - k) / (PetscReal)ySize;
	
	RectGrid **subMesh = new RectGrid *[subMeshCount];

	for (int i = 0; i < subMeshCount; i++) {
		PetscInt xCoords = 0, yCoords = 0;
		layout->getMyCoords(i, xCoords, yCoords);
		
		PetscReal xStart = m + xCoords*xWidth;
		PetscReal xEnd = m + (xCoords+1)*xWidth;
		PetscReal yStart = k + yCoords*yWidth;
		PetscReal yEnd = k + (yCoords+1)*yWidth;

		PetscPrintf(PETSC_COMM_WORLD, "Dom. num. %d: <%f-%f>x<%f-%f>\n", i, xStart,xEnd, yStart,yEnd);
		
		subMesh[i] = new RectGrid(xStart,xEnd,yStart,yEnd,h);
	}
	

		//Determination of starting indexes for subdomains and 
		PetscInt totalNodesCount = 0;
		PetscInt totalElementsCount = 0;
		PetscInt nodeBoundsCount = 0;
		PetscInt startIndexes[subMeshCount];
		PetscInt startElemIndexes[subMeshCount];
		
		for (int i = 0; i < subMeshCount; i++) {
			PetscInt xCoords = 0, yCoords = 0;
			layout->getMyCoords(i, xCoords, yCoords);
			startIndexes[i] = totalNodesCount;
			totalNodesCount += subMesh[i]->numPoints;
			startElemIndexes[i] = totalElementsCount;
			totalElementsCount += subMesh[i]->numElements;
			if (xCoords < xSize - 1) {
				nodeBoundsCount+=subMesh[i]->yPoints;	
				if (yCoords > 0) nodeBoundsCount--; //Vynechani nadbytecne vazby ve ctverci
			}
	
			if (yCoords < ySize - 1) {
				nodeBoundsCount+=subMesh[i]->xPoints;
			}
		}
		

		PetscInt sendBuf[subMeshCount*5];
		for (int i = 0; i < subMeshCount; i++) {
			sendBuf[i*5] = subMesh[rank]->numElements;
			sendBuf[i*5+1] = subMesh[rank]->numPoints;
			sendBuf[i*5+2] = nodeBoundsCount;
			sendBuf[i*5+3] = totalElementsCount;
			sendBuf[i*5+4] = totalNodesCount;
		}

		PetscInt rcvBuf[5];
		MPI_Scatter(sendBuf, 5, MPI_INT, rcvBuf, 5, MPI_INT, 0, PETSC_COMM_WORLD);

		*mesh = new Mesh(subMesh[rank]->numElements, subMesh[rank]->numPoints, nodeBoundsCount);
		(*mesh)->numElements = totalElementsCount;
		(*mesh)->numPoints = totalNodesCount;
		(*mesh)->keepPairing = true;
		(*mesh)->pointPairings = new PetscInt[nodeBoundsCount * 2];

		//Construction of dual bounding pairs, corners, dirchlet bound
		PetscInt counter = 0;
		for (int i = 0; i < subMeshCount; i++) {
			PetscInt xCoords = 0, yCoords = 0;
			layout->getMyCoords(i, xCoords, yCoords);
			PetscInt SI = startIndexes[i];

			//L-R dual pairing
			if (xCoords < xSize - 1) {
				PetscInt neibDomain = layout->getSub(xCoords+1, yCoords);
				PetscInt neibSI = startIndexes[neibDomain];
				for (int j = (yCoords > 0)?1:0; j < subMesh[i]->yPoints; j++) {
					(*mesh)->pointPairings[counter++] = subMesh[i]->iR[j] + SI;
					(*mesh)->indDual.insert(subMesh[i]->iR[j] + SI);
					(*mesh)->pointPairings[counter++] = subMesh[neibDomain]->iL[j] + neibSI;
					(*mesh)->indDual.insert(subMesh[neibDomain]->iL[j] + neibSI);
				}
			}
			//T-B dual pairing
			if (yCoords < ySize - 1) {
				PetscInt neibDomain = layout->getSub(xCoords, yCoords+1);
				PetscInt neibSI = startIndexes[neibDomain];
				for (int j = 0; j < subMesh[i]->xPoints; j++) {
					(*mesh)->pointPairings[counter++] = subMesh[i]->iT[j] + SI;
					(*mesh)->indDual.insert(subMesh[i]->iT[j] + SI);
					(*mesh)->pointPairings[counter++] = subMesh[neibDomain]->iB[j] + neibSI;
					(*mesh)->indDual.insert(subMesh[i]->iB[j] + neibSI);
				}
			}
			///TODO 
			///@todo Cornering
			if (xCoords > 0 && yCoords > 0) {
				
			}
			//Dirchlet
			for (int j = 0; j < n_dirchletSides; j++) {
				BoundSide side = dirchletBounds[j];
				if ((side == ALL || side == LEFT) && xCoords == 0) {
					for (int k = 0; k < subMesh[i]->yPoints; k++)
						(*mesh)->indDirchlet.insert(subMesh[i]->iL[k] + SI);
				}
				if ((side == ALL || side == RIGHT) && xCoords == xSize - 1) {
					for (int k = 0; k < subMesh[i]->yPoints; k++)
						(*mesh)->indDirchlet.insert(subMesh[i]->iR[k] + SI);
				}
				if ((side == ALL || side == BOTTOM) && yCoords == 0) {
					for (int k = 0; k < subMesh[i]->xPoints; k++)
						(*mesh)->indDirchlet.insert(subMesh[i]->iB[k] + SI);
				}
				if ((side == ALL || side == TOP) && yCoords == ySize - 1) {
					for (int k = 0; k < subMesh[i]->xPoints; k++)
						(*mesh)->indDirchlet.insert(subMesh[i]->iT[k] + SI);
				}
			}

			
		}

		for (int i = 0; i < subMesh[rank]->numElements; i++) {
			(*mesh)->elements[i] = subMesh[rank]->elements[i];
			Element el;
			for (int j = 0; j < 3; j++) {
				el.vetrices.insert(subMesh[rank]->elements[i].nodes[j] + startIndexes[rank]);
				(*mesh)->localVetricesSet.insert(subMesh[rank]->elements[i].nodes[j] + startIndexes[rank]);
			}

			(*mesh)->element[i + startElemIndexes[rank]] = el;
		}
		for (int i = 0; i < subMesh[rank]->numPoints; i++) {
			Point p = subMesh[rank]->nodes[i];
			(*mesh)->vetrices[i+startIndexes[rank]] = p;
		}

		for (int smIndex = 1; smIndex < subMeshCount; smIndex++) {
			{
				PetscInt sendElements[subMesh[smIndex]->numElements * 3];
				for (int i = 0; i < subMesh[smIndex]->numElements; i++) {
					for (int j = 0; j < 3; j++)
						sendElements[i*3 + j] = subMesh[smIndex]->elements[i].nodes[j] + startIndexes[smIndex];
				}
				MPI_Send(&startElemIndexes[smIndex],1,MPI_INT, smIndex, 0,PETSC_COMM_WORLD);
				MPI_Send(sendElements, subMesh[smIndex]->numElements * 3, MPI_INT, smIndex, 0, PETSC_COMM_WORLD);
			}

			{
				PetscInt sendPointIndex[subMesh[smIndex]->numPoints];
				PetscScalar sendPointCoords[subMesh[smIndex]->numPoints*3];
				
				for (int i = 0; i < subMesh[smIndex]->numPoints; i++) {
					Point p = subMesh[smIndex]->nodes[i];
					sendPointIndex[i] = i+startIndexes[smIndex];
					sendPointCoords[i*3] = p.x;
					sendPointCoords[i*3+1] = p.y;
					sendPointCoords[i*3+2] = p.z;
				}

				MPI_Send(sendPointIndex, subMesh[smIndex]->numPoints, MPI_INT, smIndex, 1, PETSC_COMM_WORLD);
				MPI_Send(sendPointCoords, subMesh[smIndex]->numPoints * 3, MPIU_SCALAR, smIndex, 0, PETSC_COMM_WORLD);
			}
		}
			
		PetscInt dirchletSize = (*mesh)->indDirchlet.size();
		PetscInt sndDirchlet[dirchletSize];
		PetscInt c = 0;
		for (std::set<PetscInt>::iterator i = (*mesh)->indDirchlet.begin();
				i != (*mesh)->indDirchlet.end(); i++) {
			sndDirchlet[c++] = *i;
		}
		MPI_Bcast(&dirchletSize, 1, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(sndDirchlet, dirchletSize, MPI_INT, 0, PETSC_COMM_WORLD);

		for (int i = 0; i < subMeshCount; i++)
			delete subMesh[i];
		delete[] subMesh;
		delete layout;
	
	} else {
		PetscInt rcvBuf[5];
		MPI_Scatter(NULL, 5, MPI_INT, rcvBuf, 5, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Status status;
		*mesh = new Mesh(rcvBuf[0], rcvBuf[1], rcvBuf[2]);
		(*mesh)->numElements = rcvBuf[3];
		(*mesh)->numPoints = rcvBuf[4];
		
		{
			PetscInt startElementIndex;
			PetscInt rcvElements[rcvBuf[0]*3];
			MPI_Recv(&startElementIndex,1,MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, PETSC_COMM_WORLD, &status);
			MPI_Recv(rcvElements, rcvBuf[0]*3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,PETSC_COMM_WORLD, &status);
			for (int i = 0; i < rcvBuf[0]; i++) {
				Element el;
				el.vetrices.insert(rcvElements + i*3, rcvElements + i*3 + 3);
				(*mesh)->element[i + startElementIndex] = el;
				(*mesh)->localVetricesSet.insert(rcvElements + i*3, rcvElements + i*3 + 3);
			}
		}


		PetscInt rcvPointIndexes[(*mesh)->mlocal_nodes];
		PetscScalar rcvPointCoords[rcvBuf[1]*3];
	
				
		MPI_Recv(rcvPointIndexes, (*mesh)->mlocal_nodes, MPI_INT, MPI_ANY_SOURCE, 1, PETSC_COMM_WORLD, &status);
		MPI_Recv(rcvPointCoords, rcvBuf[1]*3, MPIU_SCALAR, MPI_ANY_SOURCE, MPI_ANY_TAG, PETSC_COMM_WORLD, &status);
		
		for (int i = 0; i < (*mesh)->mlocal_nodes; i++) {
			Point p(rcvPointCoords[i*3], rcvPointCoords[i*3+1], rcvPointCoords[i*3+2]);
			(*mesh)->vetrices[rcvPointIndexes[i]]=p;
		}

		PetscInt dirchletSize;
		MPI_Bcast(&dirchletSize, 1, MPI_INT, 0, PETSC_COMM_WORLD);
		PetscInt rcvDirchletInd[dirchletSize];
		MPI_Bcast(rcvDirchletInd, dirchletSize, MPI_INT, 0, PETSC_COMM_WORLD);
		(*mesh)->indDirchlet.insert(rcvDirchletInd, rcvDirchletInd + dirchletSize);
	}
}

void Mesh::dumpForMatlab(PetscViewer v) {
	Mat x;
	Mat e;
	Vec dirch;
	Vec dual;

	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	{
		PetscInt indexes[] = {0,1,2,3};
		MatCreateMPIAIJ(PETSC_COMM_WORLD, mlocal_nodes, PETSC_DECIDE, PETSC_DECIDE, 4, 4,PETSC_NULL, 4,PETSC_NULL, &x);
		
		std::map<PetscInt, Point>::iterator point = vetrices.begin();
		for (int i = 0; i < mlocal_nodes;i++) {
			PetscInt rowNumber = point->first;
			PetscScalar	data[] = {point->second.x, point->second.y, point->second.z, rank};
			MatSetValues(x,1, &rowNumber,4, indexes ,data,INSERT_VALUES);
			point++;
		}
		MatAssemblyBegin(x,MAT_FINAL_ASSEMBLY);
	
		MatCreateMPIAIJ(PETSC_COMM_WORLD, mlocal_elements, PETSC_DECIDE, PETSC_DECIDE, 4,4,PETSC_NULL, 4, PETSC_NULL, &e);	
		
		std::map<PetscInt, Element>::iterator el = element.begin();
		for (int i = 0; i < mlocal_elements;i++) {
			PetscInt rowNumber = el->first;
			PetscScalar data[4];
			PetscInt counter = 0;
			for (std::set<PetscInt>::iterator j = el->second.vetrices.begin(); j != el->second.vetrices.end(); j++)
				data[counter++] = *j;
			data[3] = rank;
			MatSetValues(e,1, &rowNumber,4, indexes ,data,INSERT_VALUES);
			el++;
		}
		MatAssemblyBegin(e,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(e, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY);

		if (!rank) {
			VecCreateMPI(PETSC_COMM_WORLD, indDirchlet.size(), PETSC_DECIDE, &dirch);

			PetscInt counter = 0;
			for (std::set<PetscInt>::iterator i = indDirchlet.begin(); i != indDirchlet.end(); i++) {
				VecSetValue(dirch, counter++, *i, INSERT_VALUES);
			}

			VecCreateMPI(PETSC_COMM_WORLD, indDual.size(), PETSC_DECIDE, &dual);

			counter = 0;
			for (std::set<PetscInt>::iterator i = indDual.begin(); i != indDual.end(); i++) {
				VecSetValue(dual, counter++, *i, INSERT_VALUES);
			}

		} else {
			VecCreateMPI(PETSC_COMM_WORLD, 0, PETSC_DECIDE, &dirch);
			VecCreateMPI(PETSC_COMM_WORLD, 0, PETSC_DECIDE, &dual);
		}
		
		VecAssemblyBegin(dirch);
		VecAssemblyEnd(dirch);
		VecAssemblyBegin(dual);
		VecAssemblyEnd(dual);

	}

	MatView(x,v);
	MatView(e,v);	
	VecView(dirch,v);
	VecView(dual,v);

	MatDestroy(x);
	MatDestroy(e);
	VecDestroy(dirch);
	VecDestroy(dual);
}

RectMesh::~RectMesh() {
	delete [] iT;
	delete [] iB;
	delete [] iL;
	delete [] iR;
}

RectMesh::RectMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h) : Mesh(nPoints(m,n,k,l,h), nElements(m,n,k,l,h)) {
	PetscInt xEdges = (PetscInt)ceil((n - m) / h);
	PetscInt yEdges = (PetscInt)ceil((l - k) / h);

	xPoints = xEdges + 1;
	yPoints = yEdges + 1;

	PetscReal hx = (n - m) / xEdges;
	PetscReal hy = (l - k) / yEdges;

	PetscInt iLc=0,iRc=0,iTc=0,iBc=0,nodeC=0, elementC=0; //Counters

	iT = new PetscInt[xPoints];
	iB = new PetscInt[xPoints];
	iL = new PetscInt[yPoints];
	iR = new PetscInt[yPoints];
	
	//Discretization
	
	for (PetscInt j = 0; j < yPoints; j++)
		for (PetscInt i = 0; i < xPoints; i++) {
				PetscReal xPos = m + i*hx;
				PetscReal yPos = k + j*hy;			

				nodes[nodeC].x = xPos;
				nodes[nodeC].y = yPos;
				
				//Element creation
				if (j < yPoints - 1 && i < xPoints - 1) {
					elements[elementC].nodes[0] = nodeC;
					elements[elementC].nodes[1] = nodeC + xPoints + 1;
					elements[elementC].nodes[2] = nodeC + 1;
					elementC++;
					elements[elementC].nodes[0] = nodeC;
					elements[elementC].nodes[1] = nodeC + xPoints;
					elements[elementC].nodes[2] = nodeC + xPoints + 1;
					elementC++;				
				}

				//Defining boundaries
				if (yPos == k) iB[iBc++]=nodeC;
				if (yPos >= l) iT[iTc++]=nodeC;
				if (xPos == m) iL[iLc++]=nodeC;
				if (xPos >= n) iR[iRc++]=nodeC;
				
				nodeC++;
		}
 
}

DomainRectLayout::DomainRectLayout(PetscInt xSize, PetscInt ySize) {
	this->xSize = xSize;
	this->ySize = ySize;
	subDomains = new PetscInt[xSize * ySize];

	for (int j = 0; j < ySize; j++)
		for (int i = 0; i < xSize; i++) {
			subDomains[j*xSize + i] = j*xSize + i;
		}
}

void DomainRectLayout::getMyCoords(PetscInt ind, PetscInt &x, PetscInt &y) {
	for (int j = 0; j < ySize; j++)
		for (int i = 0; i < xSize; i++) {
			if (subDomains[j*xSize + i] == ind) {
				x = i;
				y = j;
				return;
			}
		}
}

RectGrid::RectGrid(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h) {
	PetscInt xEdges = (PetscInt)ceil((n - m) / h);
	PetscInt yEdges = (PetscInt)ceil((l - k) / h);

	xPoints = xEdges + 1;
	yPoints = yEdges + 1;

	PetscReal hx = (n - m) / xEdges;
	PetscReal hy = (l - k) / yEdges;

	numPoints = xPoints * yPoints;
	numElements = xEdges * yEdges * 2;

	nodes = new Point[numPoints];
	elements = new Element2D[numElements];
	
	PetscInt iLc=0,iRc=0,iTc=0,iBc=0,nodeC=0, elementC=0; //Counters

	iT = new PetscInt[xPoints];
	iB = new PetscInt[xPoints];
	iL = new PetscInt[yPoints];
	iR = new PetscInt[yPoints];
	
	//Discretization
	
	for (PetscInt j = 0; j < yPoints; j++)
		for (PetscInt i = 0; i < xPoints; i++) {
				PetscReal xPos = m + i*hx;
				PetscReal yPos = k + j*hy;			

				nodes[nodeC].x = xPos;
				nodes[nodeC].y = yPos;
				
				//Element creation
				if (j < yPoints - 1 && i < xPoints - 1) {
					elements[elementC].nodes[0] = nodeC;
					elements[elementC].nodes[1] = nodeC + xPoints + 1;
					elements[elementC].nodes[2] = nodeC + 1;
					elementC++;
					elements[elementC].nodes[0] = nodeC;
					elements[elementC].nodes[1] = nodeC + xPoints;
					elements[elementC].nodes[2] = nodeC + xPoints + 1;
					elementC++;				
				}

				//Defining boundaries
				if (yPos == k) iB[iBc++]=nodeC;
				if (yPos >= l) iT[iTc++]=nodeC;
				if (xPos == m) iL[iLc++]=nodeC;
				if (xPos >= n) iR[iRc++]=nodeC;
				
				nodeC++;
		}
 
}

RectGrid::~RectGrid() {
	delete [] elements;
	delete [] nodes;

	delete [] iT;
	delete [] iB;
	delete [] iL;
	delete [] iR;
}
