#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <kalman_filter_sp.h>

// Defining functions:
// The second argument passed in to the Py_InitModule function is a structure that makes it easy to define functions in the module.
// In the example given above, the mymethods structure would have been defined earlier in the file (usually right before the init{name} subroutine) to:

/* Define functions in module */
static PyMethodDef KalmanFilterMethods[] = {
    {"kalman_filter", kalman_filter, METH_VARARGS, "Calculate the Fibonacci numbers (in C)."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Create PyModuleDef structure */
static struct PyModuleDef KalmanFilterStruct = {
    PyModuleDef_HEAD_INIT,
    "kalman_filter",
    "",
    -1,
    KalmanFilterMethods,
    NULL,
    NULL,
    NULL,
    NULL};

/* Module initialization */
PyObject *PyInit_kalman_filter(void)
{
    return PyModule_Create(&KalmanFilterStruct);
}
