#include <stdbool.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <cblas.h>
#include <utils.h>

// Logical function:
double *ckalman_filter_test(
    // n: the total number of observations
    int n,
    // m: the dimension of the state vector
    int m,
    // d: the dimension of observations
    int d,
    // Arrays / matrices, are we incrementing on these arrays / matrices?
    double *a0,
    double *P0,
    double *dt, int incdt,
    double *ct, int incct,
    double *Tt, int incTt,
    double *Zt, int incZt,
    double *HHt, int incHHt,
    double *GGt, int incGGt,
    double *yt)
{

    // Create a result tuple with two arrays to return
    double **results = (double **)malloc(2 * sizeof(double *));
    results[0] = a0;
    results[1] = yt;

    return results; // Return the two arrays

    // return yt;
}
