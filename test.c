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

    // Create an array of addresses of output arrays:
    double **results = (double **)malloc(8 * sizeof(double *));
    results[0] = a0;
    results[1] = yt;
    results[2] = P0;
    results[3] = dt;
    results[4] = ct;
    results[5] = Tt;
    results[6] = Zt;
    results[7] = HHt;

    return results; // Return the two arrays
}
