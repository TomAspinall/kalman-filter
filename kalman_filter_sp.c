#include <Python.h>
#include "numpy/arrayobject.h"

// Logical function:
int ckalman_filter_sequential(
    // n: the total number of observations
    int n,
    // m: the dimension of the state vector
    int m,
    // d: the dimension of observations
    int d,
    // a0:
    double *a0)
{

    printf("Debug: n is: (%d)\n", n);
    printf("Debug: m is: (%d)\n", m);
    printf("Debug: d is: (%d)\n", d);
    printf("Debug: *a0 is: (%d)\n", *a0);

    // Array dimensions:
    int m_x_m = m * m;
    int m_x_d = m * d;
    int na_sum;
    int t = 0;

    // integers and double precisions used in dcopy and dgemm
    int intone = 1;
    double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;
    // To transpose or not transpose matrix
    char *transpose = "T", *dont_transpose = "N";

    // SEQUENTIAL PROCESSING DEFINED VARIABLES:
    int N_obs = 0;

    // Doubles for the Sequential Processing iteration:
    double V;
    double Ft;
    double tmpFtinv;

    // PyArray_malloc();
    if (n < m)
        return n;
    else
        return m;
}

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter(PyObject *self, PyObject *args)
{
    PyObject *a0;
    int n, m, d;

    /* Parse the input, from Python integer to C int */
    if (!PyArg_ParseTuple(args, "iiO", &m, &d, &a0))
        return NULL;
    n = PyObject_Length(a0);
    // Exception handling:
    if (n < 0)
    {
        return NULL;
    }
    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */

    /* Construct the result: a Python integer object */
    return Py_BuildValue("ii", ckalman_filter_sequential(n, m, d, &a0), d);
}

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

    import_array(); // Inititalise Numpy
    if (PyErr_Occurred())
    {
        printf("Failed to import numpy Python module(s).");
        return NULL;
    }

    return PyModule_Create(&KalmanFilterStruct);
}
