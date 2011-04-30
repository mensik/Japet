#
LINKERFLAGS = -lpetsc -lparmetis -lmed
CPPCOMP = mpic++
CLINKER = mpic++
CPPFLAGS = -Wno-unused-result -I/usr/include/petsc -I/usr/include/

headerDir = ../src
MYLIB = structures.o fem.o feti.o solver.o japetUtils.o japetLa.o

vpath %.cpp ../src
vpath %.h $(headerDir)

.cpp.o:
	$(CPPCOMP) $(CPPFLAGS) -o $@ -c $<

feti: $(MYLIB) testFeti.o
	$(CLINKER) $(LINKERFLAGS) $(MYLIB) testFeti.o -o testFeti
	mv testFeti ../deploy
