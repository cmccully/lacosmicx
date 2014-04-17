CFITSINCDIR = /usr/local/include
PYTHONINCDIR = /usr/stsci/pyssg/Python-2.7.1/include/python2.7
NUMPYINCDIR = /usr/stsci/pyssg/2.7.1/numpy/core/include/numpy

CFITSLIBDIR = /usr/local/lib
PYTHONLIBDIR = /usr/stsci/pyssgx/Python-2.7.1/lib

COPTS = -funroll-loops -O3 -fopenmp -Wall -I$(CFITSINCDIR) -I$(PYTHONINCDIR) -I$(NUMPYINCDIR) -I/usr/include/malloc 
LIBS  = -L$(CFITSLIBDIR) -lcfitsio -L$(PYTHONLIBDIR) -lpython -funroll-loops -O3 -fopenmp -Wall -lgomp

CPP    = g++

all:
	$(CPP)  $(COPTS) -o lacosmicx.o -c lacosmicx.cpp 
	$(CPP)  $(COPTS) -o functions.o -c functions.cpp
	$(CPP)  $(COPTS) -o lacosmicx_py.o -c lacosmicx_py.cpp
	$(CPP) -o lacosmicx lacosmicx_py.o functions.o lacosmicx.o $(LIBS)
	python setup.py build
	find ./build -name "_lacosmicx.so" -exec cp {} ./ \;
	
install:
	cp ./lacosmicx /usr/bin/
	python setup.py install
	
clean:
	rm -rf build
	rm -rf *.o
	rm -rf lacosmicx
	rm -rf *.pyc
	rm -rf *.so
