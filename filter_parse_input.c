#include <Python.h>
#include "numpy/arrayobject.h"

// Given an input payload, validate input payload objects, such as conformed payload dimensionality.
int cfilter_parse_input(
    // input dictionary:
    PyObject *kwargs,
    // ndarrays:
    PyArrayObject **ndarrays,
    // output formatted double arrays for algorithm input:
    double **parsed_inputs,
    // Array of parsed input dimensions, again for algorithm input:
    int parsed_input_dim[9])
{

    const int total_keys = 9;

    // Input dictionary keys:
    const char *keys[] = {
        "x", "P", "yt", "dt", "ct", "GGt", "Tt", "Zt", "HHt"};
    // ndarray dims:
    npy_intp *array_dims[9];
    // ndarray number of dimensions:
    int array_ndims[9];

    // Extract NumPy arrays from input dictionary:
    for (int i = 0; i < total_keys; i++)
    {
        PyObject *item = PyDict_GetItemString(kwargs, keys[i]);
        if (item == NULL)
        {
            PyErr_Format(PyExc_KeyError, "Dictionary is missing '%s' key", keys[i]);
            return 1;
        }
        // if (!PyArray_Check(item))
        // {
        //     PyErr_Format(PyExc_TypeError, "'%s' is not a valid numpy.ndarray object", keys[i]);
        //     return 1;
        // }
        // Fetch ndarray object:
        ndarrays[i] = (PyArrayObject *)item;
        // Fetch dimension sizes:
        array_dims[i] = PyArray_DIMS(ndarrays[i]);
        // Fetch dimension size:
        array_ndims[i] = PyArray_NDIM(ndarrays[i]);
    }

    return 0;
}