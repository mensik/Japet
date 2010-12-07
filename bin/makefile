#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         = -Wno-unused-result 
FPPFLAGS         =
MANSEC           = Mat
LOCDIR		 = 
MYLIB = smale.o solver.o fem.o feti.o structures.o smalbe.o japetUtils.o
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

vpath %.h ../src
vpath %.cpp ../src

testFeti: testFeti.o chkopts $(MYLIB)
	-${CLINKER} -o testFeti testFeti.o $(MYLIB) ${PETSC_KSP_LIB}

testMPRGP: testMPRGP.o chkopts $(MYLIB)
	-${CLINKER} -o testMPRGP testMPRGP.o $(MYLIB) ${PETSC_KSP_LIB}

smalbe: testSMALBE.o chkopts $(MYLIB)
	-${CLINKER} -o smalbe testSMALBE.o $(MYLIB) ${PETSC_KSP_LIB}

test: test.o chkopts structures.o fem.o feti.o solver.o japetUtils.o
	-${CLINKER} -o test test.o fem.o structures.o feti.o solver.o japetUtils.o ${PETSC_KSP_LIB}
	 ${RM} test.o 

testStructures: testStructures.o chkopts structures.o fem.o feti.o solver.o japetUtils.o
	-${CLINKER} -o testStructures testStructures.o fem.o structures.o feti.o solver.o japetUtils.o ${PETSC_KSP_LIB}
	${RM} testStructures.o

testSmale: testSmale.o chkopts $(MYLIB)
	-${CLINKER} -o testSmale testSmale.o $(MYLIB) ${PETSC_KSP_LIB}
	${RM} testSmale.o

testSassi: testSassi.o chkopts $(MYLIB)
	-${CLINKER} -o testSassi testSassi.o $(MYLIB) ${PETSC_KSP_LIB}
	${RM} testSassi.o

ex11: ex11.o  chkopts
	-${CLINKER}  -o ex11 ex11.o ${PETSC_MAT_LIB}
	 ${RM} ex11.o
