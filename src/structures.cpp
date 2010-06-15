#include "structures.h"
 
void Mesh::dumpForMatlab(PetscViewer v) {
/*	Mat x;
	Mat e;
	Vec dirch;
	Vec dual;
	Vec primal;

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

			VecCreateMPI(PETSC_COMM_WORLD, indPrimal.size(), PETSC_DECIDE, &primal);

			counter = 0;
			for (std::set<PetscInt>::iterator i = indPrimal.begin(); i != indPrimal.end(); i++) {
				VecSetValue(primal, counter++, *i, INSERT_VALUES);
			}

		} else {
			VecCreateMPI(PETSC_COMM_WORLD, 0, PETSC_DECIDE, &dirch);
			VecCreateMPI(PETSC_COMM_WORLD, 0, PETSC_DECIDE, &dual);
			VecCreateMPI(PETSC_COMM_WORLD, 0, PETSC_DECIDE, &primal);
		}
		
		VecAssemblyBegin(dirch);
		VecAssemblyEnd(dirch);
		VecAssemblyBegin(dual);
		VecAssemblyEnd(dual);
		VecAssemblyBegin(primal);
		VecAssemblyEnd(primal);

	}


	MatView(x,v);
	MatView(e,v);	
	VecView(dirch,v);
	VecView(dual,v);
	VecView(primal,v);

	MatDestroy(x);
	MatDestroy(e);
	VecDestroy(dirch);
	VecDestroy(dual);
	VecDestroy(primal);
*/
}

void Mesh::generateRectangularMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	PetscInt xEdges = (PetscInt)ceil((n - m) / h);
	PetscInt yEdges = (PetscInt)ceil((l - k) / h);

	PetscInt xPoints = xEdges + 1;
	PetscInt yPoints = yEdges + 1;

	PetscReal hx = (n - m) / xEdges;
	PetscReal hy = (l - k) / yEdges;

	//Discretization
	PetscInt nodeIndex = 0;
	PetscInt elementIndex = 0;
	for (PetscInt j = 0; j < yPoints; j++)
		for (PetscInt i = 0; i < xPoints; i++) {
				PetscReal xPos = m + i*hx;
				PetscReal yPos = k + j*hy;			

				Point node(xPos,yPos);
				vetrices.insert(std::pair<PetscInt, Point>(nodeIndex, node));
				
				//Element creation
				if (j < yPoints - 1 && i < xPoints - 1) {
					Element el1, el2;

					el1.numVetrices = 3;
					el1.numEdges = 0;
					el1.vetrices[0] = nodeIndex;
					el1.vetrices[1] = nodeIndex + xPoints + 1;
					el1.vetrices[2] = nodeIndex + 1;
					
					el2.numVetrices = 3;
					el2.numEdges = 0;
					el2.vetrices[0] = nodeIndex;
					el2.vetrices[1] = nodeIndex + xPoints;
					el2.vetrices[2] = nodeIndex + xPoints +	 1;

					elements.insert(std::pair<PetscInt, Element>(elementIndex++, el1));
					elements.insert(std::pair<PetscInt, Element>(elementIndex++, el2));
				}
				nodeIndex++;
		}
	regenerateEdges();
	findBorders();
}

void Mesh::regenerateEdges() {
	PetscInt edgeIndex = 0;
	for (std::map<PetscInt, Element>::iterator i = elements.begin(); i != elements.end(); i++) {
		for (PetscInt j = 0; j < i->second.numVetrices; j++) {
			PetscInt v1 = i->second.vetrices[j];
			PetscInt v2 = i->second.vetrices[(j+1) % (i->second.numVetrices)];
			
			PetscInt edgeInd = getEdge(v1,v2);
			if (edgeInd == -1) { // Edge doesn't exist yet
				Edge newEdge;
				edgeInd = edgeIndex++;
				newEdge.vetrices[0] = v1;
				newEdge.vetrices[1] = v2;
				newEdge.elements.insert(i->first);
				vetrices[v1].edges.insert(edgeInd);
				vetrices[v2].edges.insert(edgeInd);
				edges.insert(std::pair<PetscInt, Edge>(edgeInd, newEdge));
			}
			i->second.edges[i->second.numEdges] = edgeInd;
			edges[edgeInd].elements.insert(i->first);
		}
	}
}

void Mesh::findBorders() {
	for (std::map<PetscInt, Edge>::iterator i = edges.begin(); i != edges.end(); i++) {
		if (i->second.elements.size() == 1) borderEdges.insert(i->first);
	}
}

PetscInt Mesh::getEdge(PetscInt nodeA, PetscInt nodeB) {

	PetscInt resultIndex = -1;
	for (std::set<PetscInt>::iterator i = vetrices[nodeA].edges.begin(); i != vetrices[nodeA].edges.end(); i++) {
		if (edges[*i].vetrices[0] == nodeB || edges[*i].vetrices[1] == nodeB) {
			resultIndex = *i;
			break;
		}
	}

	return resultIndex;
}


void extractLocalAPart(Mat A, std::set<PetscInt> vetrices, Mat *Aloc) {
	PetscInt localIndexes[vetrices.size()];
	PetscInt counter = 0;
	for (std::set<PetscInt>::iterator i = vetrices.begin();
		i != vetrices.end(); i++) {
		localIndexes[counter++] = *i;
	}
	IS ISlocal;
	ISCreateGeneral(PETSC_COMM_SELF, vetrices.size(), localIndexes, &ISlocal);
	Mat *sm;
	MatGetSubMatrices(A, 1, &ISlocal, &ISlocal, MAT_INITIAL_MATRIX, &sm);
	*Aloc = *sm;
}

