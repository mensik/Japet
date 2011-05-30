#
LINKERFLAGS = -lpetsc -lparmetis -lmed -g
CPPCOMP = mpic++
CLINKER = mpic++
CPPFLAGS = -Wno-unused-result -I/usr/include/petsc -I/usr/include/ -g 

headerDir = ../src
MYLIB = structures.o fem.o feti.o solver.o japetUtils.o japetLa.o

vpath %.cpp ../src
vpath %.h $(headerDir)

.cpp.o:
	$(CPPCOMP) $(CPPFLAGS) -o $@ -c $<

feti: $(MYLIB) testFeti.o
	$(CLINKER) $(LINKERFLAGS) $(MYLIB) testFeti.o -o testFeti
	mv testFeti ../deploy

hfeti: $(MYLIB) testHFeti.o
	$(CLINKER) $(LINKERFLAGS) $(MYLIB) testHFeti.o -o testHFeti
	mv testHFeti ../deploy

test: $(MYLIB) test.o
	$(CLINKER) $(LINKERFLAGS) $(MYLIB) test.o -o testSolver
	mv testSolver ../deploy

structures: $(MYLIB) testStructures.o
	$(CLINKER) $(LINKERFLAGS) $(MYLIB) testStructures.o -o testStructures
	mv testStructures ../deploy

clean:
	rm *.o
