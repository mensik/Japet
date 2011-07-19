//============================================================================
// Name        : dayOne.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "petscmat.h"

using namespace std;

double norm(int len, double *vec) {
	double normV = 0;

	for (int i = 0; i < len; i++) {
		normV += vec[i]*vec[i];
	}

	return sqrt(normV);
}

int main(int argc, char *argv[]) {

	double a[10];

	for (int i = 0; i < 10; i++) {
		a[i] = i + 1;
	}

	cout << norm(10, a) << endl;


	return 0;
}
