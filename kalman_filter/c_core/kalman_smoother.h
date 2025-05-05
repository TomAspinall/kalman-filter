#include <stdbool.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <cblas.h>
#include <utils.h>

/************************************************************************************/
/* ---------- --------- Kalman Smoothing Through Sequential Processing --------- ---*/
/************************************************************************************/
// This function performs Kalman smoothing. It iterates backwards through t.

void ckalman_smoother(
    /* Dimesions*/
    // n: the total number of observations
    int n,
    // m: the dimension of the state vector
    int m,
    // d: the dimension of observations
    int d,

    // Inputs:
    double *Zt, int incZt,
    double *yt,
    double *vt,
    double *Tt, int incTt,
    double *Kt,
    double *Ft_inv,
    double *att,
    double *Ptt,
    // Outputs:
    double *xhatt_output,
    double *Vt_output);