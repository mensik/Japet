#
LINKERFLAGS = -lpetsc -lparmetis
CPPCOMP = mpic++ 
CLINKER = mpic++ 
CPPFLAGS = -I/usr/include/petsc -I/usr/include/ 

headerDir = ../src
MYLIB = structures.o fem.o feti.o solver.o japetUtils.o japetLa.o

vpath %.cpp ../src
vpath %.h $(headerDir)

.cpp.o:
	$(CPPCOMP) $(CPPFLAGS) -o $@ -c $<

feti: $(MYLIB) testFeti.o
	$(CLINKER) $(MYLIB) testFeti.o -o testFeti $(LINKERFLAGS)
	mv testFeti ../deploy

hfeti: $(MYLIB) testHFeti.o
	$(CLINKER) $(MYLIB) testHFeti.o -o testHFeti $(LINKERFLAGS) 
	mv testHFeti ../deploy

test: $(MYLIB) test.o
	$(CLINKER)  $(MYLIB) test.o -o testSolver $(LINKERFLAGS) 
	mv testSolver ../deploy

structures: $(MYLIB) testStructures.o
	$(CLINKER)  $(MYLIB) testStructures.o -o testStructures $(LINKERFLAGS) 
	mv testStructures ../deploy

clean:
	rm *.o
