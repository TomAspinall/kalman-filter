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

// PyMODINIT_FUNC
// initkalman_filter(void)
// {
//     (void)Py_InitModule(kalman_filter, mymethods);
//     import_array();
// }
// // The mymethods must be an array (usually statically declared) of PyMethodDef structures which contain method names, actual C-functions, a variable indicating whether the method uses keyword arguments or not, and docstrings.
