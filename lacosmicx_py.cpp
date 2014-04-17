using namespace std;
#include "lacosmicx_py.h"
#include "lacosmicx.h"
#include "functions.h"
#include "Python.h"
#include "arrayobject.h"
#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>
#include<string.h>

/* ==== Set up the methods table ====================== */
PyMethodDef _lacosmicxMethods[] = { { "pyrun", pyrun, METH_VARARGS }, { NULL,
		NULL } /* Sentinel - marks the end of this structure */
};
PyMODINIT_FUNC init_lacosmicx() {
	(void) Py_InitModule("_lacosmicx", _lacosmicxMethods);
	import_array(); // Must be present for NumPy.  Called first after above line.
}

PyObject* pyrun(PyObject* self, PyObject* args) {

	//This is how the call looks in python
	Py_Initialize();
	//_Lacosmicx.pyrun(inmat,inmask,outmaskfile,nx,ny,sigclip,objlim,sigfrac,satlevel,gain,pssl,readnoise,robust,verbose,niter)
	PyArrayObject *indat, *inmask;
	char* outmaskfile;
	int nx, ny, i;
	float sigclip, objlim, satlevel, gain, pssl, readnoise, sigfrac;
	bool verbose;
	bool robust;
	int niter;
	// Parse tuples separately since args will differ
	if (!PyArg_ParseTuple(args, "O!O!siifffffffbbi", &PyArray_Type, &indat,
			&PyArray_Type, &inmask, &outmaskfile, &nx, &ny, &sigclip, &objlim,
			&sigfrac, &satlevel, &gain, &pssl, &readnoise, &robust, &verbose,
			&niter))
		return NULL;

	/* Get the dimension of the input */
	npy_intp dims[1] = { indat->dimensions[0] };
	/* Change contiguous arrays into C *arrays   */
	float* data;
	data = pyvector_to_Carrayptrs(indat);
	bool* mask;
	mask = pyvector_to_Carrayptrsbool(inmask);

	lacosmicx* l;
	l = new lacosmicx(data, mask, nx, ny, pssl, gain, readnoise, sigclip,
			sigfrac, objlim, satlevel, robust, verbose);

	l->run(niter);
	//cout << l;
	if (strcmp(outmaskfile, "") != 0) {
		booltofits((char *) outmaskfile, l->crmask, nx, ny);
	}

	/* Make a new double vector of same dimension */
	PyArrayObject* outdat;
	outdat = (PyArrayObject *) PyArray_SimpleNew(1, dims,NPY_FLOAT);
	float* cout;
	cout = pyvector_to_Carrayptrs(outdat);
	int nxny = nx * ny;
	for (i = 0; i < nxny; i++) {
		cout[i] = l->cleanarr[i];
	}
	delete l;

	return PyArray_Return(outdat);
}

/*Utility Functions: These functions are directly taken from the Scipy C-extensions cookbook
 */
/* ==== Create 1D Carray from PyArray ======================
 Assumes PyArray is contiguous in memory.             */
float* pyvector_to_Carrayptrs(PyArrayObject* arrayin) {
	int n;

	n = arrayin->dimensions[0];
	return (float*) arrayin->data; /* pointer to arrayin data as double */
}
bool* pyvector_to_Carrayptrsbool(PyArrayObject* arrayin) {
	int n;

	n = arrayin->dimensions[0];
	return (bool*) arrayin->data; /* pointer to arrayin data as double */
}

