/*
 * Lacosmicx_py.h
 *
 *  Created on: May 27, 2011
 *      Author: cmccully
 */

#ifndef LACOSMICX_PY_H_
#define LACOSMICX_PY_H_
#include "Python.h"
#include "arrayobject.h"
#include "lacosmicx.h"

PyMODINIT_FUNC init_lacosmicx();
static PyObject* pyrun(PyObject* self, PyObject* args);
float* pyvector_to_Carrayptrs(PyArrayObject* arrayin);
bool* pyvector_to_Carrayptrsbool(PyArrayObject* arrayin);

#endif /* LACOSMICX_PY_H_ */
