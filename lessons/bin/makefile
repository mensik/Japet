#
LINKERFLAGS = -lpetsc -lparmetis
CPPCOMP = mpic++
CLINKER = mpic++
CPPFLAGS = -I/usr/include/petsc -I/usr/include/  

headerDir = ../src

vpath %.cpp ../src
vpath %.h $(headerDir)

.cpp.o:
	$(CPPCOMP) $(CPPFLAGS) -o $@ -c $<

hello: $(MYLIB) dayOne.o
	$(CLINKER) $(LINKERFLAGS) dayOne.o -o dayOne

clean:
	rm *.o
