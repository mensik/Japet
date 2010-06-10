#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Mat
LOCDIR		 = 
MYLIB = solver.o fem.o smale.o feti.o structures.o
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

vpath %.h ../src
vpath %.cpp ../src

testFeti: testFeti.o chkopts $(MYLIB)
	-${CLINKER} -o testFeti testFeti.o $(MYLIB) ${PETSC_KSP_LIB}
#	rm $(MYLIB) testFeti.o 

test: test.o chkopts $(MYLIB)
	-${CLINKER} -o test test.o $(MYLIB) ${PETSC_KSP_LIB}
	${RM} test.o 

testSmale: testSmale.o chkopts $(MYLIB)
	-${CLINKER} -o testSmale testSmale.o $(MYLIB) ${PETSC_KSP_LIB}
	${RM} testSmale.o

testSassi: testSassi.o chkopts $(MYLIB)
	-${CLINKER} -o testSassi testSassi.o $(MYLIB) ${PETSC_KSP_LIB}
	${RM} testSassi.o

ex11: ex11.o  chkopts
	-${CLINKER}  -o ex11 ex11.o ${PETSC_MAT_LIB}
	 ${RM} ex11.o
