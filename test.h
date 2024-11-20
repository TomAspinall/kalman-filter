#include <Python.h>

// Logical function:
void ckalman_filter_test(
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
    double *yt,
    // Outputs:
    double *loglik,
    double *att_output,
    double *Ptt_output,
    double *at_output,
    double *Pt_output,
    double *Ft_inv_output,
    double *vt_output,
    double *Kt_output);