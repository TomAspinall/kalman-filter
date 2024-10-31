#include <Python.h>

// Logical function:
int ckalman_filter(int n)
{
    // printf("Debug: calling ckalman_filter(%d)\n", n);
    if (n <= 1)
        return n;
    else
        return ckalman_filter(n - 1) + ckalman_filter(n - 2);
}

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter(PyObject *self, PyObject *args)
{
    int n;

    /* Parse the input, from Python integer to C int */
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    /* If the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */

    /* Construct the result: a Python integer object */
    return Py_BuildValue("i", ckalman_filter(n));
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
    return PyModule_Create(&KalmanFilterStruct);
}
