#
LINKERFLAGS = -Wl,-rpath,/home/martin/documents/code/petsc-3.1-p8/linux-gnu-cxx-debug/lib -Wl,-rpath,/home/martin/documents/code/petsc-3.1-p8/linux-gnu-cxx-debug/lib -L/home/martin/documents/code/petsc-3.1-p8/linux-gnu-cxx-debug/lib -lpetsc -lX11 -Wl,-rpath,/home/martin/documents/code/petsc-3.1-p8/linux-gnu-cxx-debug/lib -L/home/martin/documents/code/petsc-3.1-p8/linux-gnu-cxx-debug/lib -lparmetis -lmetis -llapack -lblas -lnsl -lrt -lm -L/usr/lib/gcc/x86_64-linux-gnu/4.6.1 -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -ldl -lmpich -lopa -lmpl -lrt -lcr -lpthread -lgcc_s -lmpichf90 -lgfortran -lm -lgfortran -lm -lquadmath -lm -lmpichcxx -lstdc++ -ldl -lmpich -lopa -lmpl -lrt -lcr -lpthread -lgcc_s -ldl

CPPCOMP = vtcxx -vt:inst manual -DVTRACE
CLINKER = vtcxx -vt:inst manual -DVTRACE
CPPFLAGS = -I/home/martin/documents/code/petsc-3.1-p8/linux-gnu-cxx-debug/include -I/home/martin/documents/code/petsc-3.1-p8/include -I/home/martin/documents/code/petsc-3.1-p8/linux-gnu-cxx-debug/include -I/usr/include/mpich2

headerDir = ../src
MYLIB = structures.o fem.o feti.o solver.o japetUtils.o japetLa.o

vpath %.cpp ../src
vpath %.h $(headerDir)

.cpp.o:
	echo Compiling $<
	$(CPPCOMP) $(CPPFLAGS) -o $@ -c $<

.SILENT:

feti: $(MYLIB) testFeti.o
	$(CLINKER) $(MYLIB) testFeti.o -o testFeti $(LINKERFLAGS)
	mv testFeti ../deploy

hfeti: $(MYLIB) testHFeti.o
	echo Linking hFeti
	$(CLINKER) $(MYLIB) testHFeti.o -o testHFeti $(LINKERFLAGS) 
	mv testHFeti ../deploy

test: $(MYLIB) test.o
	$(CLINKER)  $(MYLIB) test.o -o testSolver $(LINKERFLAGS) 
	mv testSolver ../deploy

structures: $(MYLIB) testStructures.o
	echo Linking structures
	$(CLINKER)  $(MYLIB) testStructures.o -o testStructures $(LINKERFLAGS) 
	mv testStructures ../deploy

testMPI: testMPI.o
	$(CLINKER) testMPI.o -o testMPI $(LINKERFLAGS)
	mv testMPI ../deploy

clean:
	rm *.o
