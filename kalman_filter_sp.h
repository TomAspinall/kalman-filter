#include <Python.h>

// Logical function:
int ckalman_filter(int n);

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter(PyObject *self, PyObject *args);

// PyMODINIT_FUNC
// initkalman_filter(void)
// {
//     (void)Py_InitModule(kalman_filter, mymethods);
//     import_array();
// }
// // The mymethods must be an array (usually statically declared) of PyMethodDef structures which contain method names, actual C-functions, a variable indicating whether the method uses keyword arguments or not, and docstrings.
