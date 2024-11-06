#include <stdbool.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <cblas.h>
#include <utils.h>

// Logical function:
double **ckalman_filter_test(
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
    int *loglik)
{

    // Coerce dtypes:
    blasint blas_n = (blasint)n;
    blasint blas_m = (blasint)m;
    blasint blas_d = (blasint)d;

    // Utilised array dimensions:
    blasint m_x_m = m * m;
    blasint m_x_d = m * d;

    // integers and double precisions used in dcopy and dgemm
    blasint intone = 1;
    blasint intminusone = -1;
    double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;

    // Sequential Processing variables:
    int N_obs = 0;
    int na_sum;
    double V;
    double Ft;
    double tmpFtinv;

    // Time-series iterator:
    int t = 0;

    /* NA detection */
    int *NAindices = malloc(sizeof(int) * d);
    int *positions = malloc(sizeof(int) * d);

    /* Reduced arrays when NA's at time t */
    double *yt_temp = malloc(sizeof(double) * (d - 1));
    double *ct_temp = malloc(sizeof(double) * (d - 1));
    double *Zt_temp = malloc(sizeof(double) * (d - 1) * m);
    double *GGt_temp = malloc(sizeof(double) * (d - 1));

    double *Zt_t = malloc(sizeof(double) * (d * m));
    double *Zt_tSP = malloc(sizeof(double) * m);

    double *at = malloc(sizeof(double) * m);
    double *Pt = malloc(sizeof(double) * m * m);
    double *Kt = malloc(sizeof(double) * m);

    double *tmpmxSP = (double *)calloc(m, sizeof(double));
    double *tmpmxm = (double *)calloc(m_x_m, sizeof(double));

    /* at = a0 */
    cblas_dcopy(blas_m, a0, intone, at, intone);

    /* Pt = P0 */
    cblas_dcopy(m_x_m, P0, intone, Pt, intone);

    /**************************************************************/
    /* ---------- ---------- end recursions ---------- ---------- */
    /**************************************************************/

    // Update the final Log-Likelihood Score:
    *loglik -= 0.5 * N_obs * log(2 * M_PI);

    // Memory clean - vectors / matrices:
    free(tmpmxSP);
    free(tmpmxm);
    free(positions);
    free(yt_temp);
    free(ct_temp);
    free(Zt_temp);
    free(GGt_temp);
    free(Zt_t);
    free(Zt_tSP);
    free(NAindices);
    free(Kt);
    free(at);
    free(Pt);

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

    return results;
}
