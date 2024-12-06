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
    int parsed_input_dim[9]);
