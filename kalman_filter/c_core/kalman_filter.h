#include <Python.h>

void ckalman_filter(
    // n: the total number of measurements
    int n,
    // m: the dimension of the state vector
    int m,
    // d: the dimension of measurements
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
    double *yt,
    // Outputs:
    double *loglik);
