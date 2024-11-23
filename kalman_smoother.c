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
    double *Ptt)
{

    // integers and double precisions used in dcopy and dgemm
    blasint intone = 1;
    blasint intminusone = -1;
    double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;

    // Coerce dtypes:
    blasint blas_n = (blasint)n;
    blasint blas_m = (blasint)m;
    blasint blas_d = (blasint)d;

    // Utilised array dimensions:
    blasint m_x_m = m * m;
    blasint m_x_d = m * d;

    /* temporary arrays */
    double *tmpmxm = (double *)calloc(m_x_m, sizeof(double));
    double *tmpPt = (double *)calloc(m_x_m, sizeof(double));
    double *tmpN = (double *)calloc(m_x_m, sizeof(double));
    double *tmpr = (double *)calloc(m, sizeof(double));

    /* NA detection */
    int na_sum;
    int *NAindices = malloc(sizeof(int) * d);
    int *positions = malloc(sizeof(int) * d);

    /* create reduced arrays for SP and when NULL's are present */
    double *Zt_t = malloc(sizeof(double) * (d * m));
    double *Zt_temp = malloc(sizeof(double) * m);
    double *Zt_NA = malloc(sizeof(double) * (d - 1) * m);

    // Used during smoothing:
    double tmp_scalar;

    // Smoothing Recursion parameters:
    double *N = (double *)calloc(m_x_m, sizeof(double));
    double *r = (double *)calloc(m, sizeof(double));
    double *L = (double *)calloc(m_x_m, sizeof(double));

    // Develop Identity matrix (for L):
    double *identity_matrix = (double *)calloc(m_x_m, sizeof(double));
    for (int i = 0; i < m; i++)
    {
        identity_matrix[i * m + i] = 1.0;
    }
    // Kalman smoothing iterates backwards:
    int t = n - 1;

    // Iterate over all time steps:
    while (t > -1)
    {

        /* ahat_t = P_t %*% r_t-1 + a_t */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    blas_m, intone, blas_m,
                    dblone, &Ptt[m_x_m * t], blas_m,
                    r, blas_m,
                    dblone, &att[m * t], blas_m);

        /* V_t = P_t - P_t %*% N_t-1 %*% P_t */
        // Step 1: tmpmxm = P_t %*% N_t-1:
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    blas_m, blas_m, blas_m,
                    dblone, &Ptt[m_x_m * t], blas_m,
                    N, blas_m,
                    dblzero, tmpmxm, blas_m);

        /* Pt[,,i] = Pt[,,i] - tmpmxm%*% Pt[,,i] */
        cblas_dcopy(m_x_m, &Ptt[m_x_m * t], intone, tmpPt, intone);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    blas_m, blas_m, blas_m,
                    dblminusone, tmpmxm, blas_m,
                    tmpPt, blas_m,
                    dblone, &Ptt[m_x_m * t], blas_m);

        // Move from r_t,0 to r_(t-1),pt:
        // r_(t-1),p_t = t(T_t-1) %*% r_t,0:
        cblas_dcopy(blas_m, r, intone, tmpr, intone);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    blas_m, intone, blas_m,
                    dblone, &Tt[m_x_m * t * incTt], blas_m,
                    tmpr, blas_m,
                    dblzero, r, blas_m);

        // N_(t-1,p_t )= t(T_t-1) N_(t,0) T_(t-1)

        // Step 1 - tmpmxm = t(T_t-1) %*% N
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    blas_m, blas_m, blas_m,
                    dblone, &Tt[m_x_m * t * incTt], blas_m,
                    N, blas_m,
                    dblzero, tmpmxm, blas_m);

        // Step 2 - N = tmpmxm %*% T_(t-1)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    blas_m, blas_m, blas_m,
                    dblone, tmpmxm, blas_m,
                    &Tt[m_x_m * t * incTt], blas_m,
                    dblzero, N, blas_m);

        /************************/
        /* check for NA's in observation yt[,t] */
        /************************/
        na_sum = numberofNA(&yt[d * t], NAindices, positions, d);

        /*********************************************************************************/
        /* ---------- ---------- ---------- smoothing step ---------- ---------- -------- */
        /*********************************************************************************/

        // Case 1: No NA's:
        if (na_sum == 0)
        {
            // Create Zt for time t
            cblas_dcopy(m_x_d, &Zt[m_x_d * t * incZt], intone, Zt_t, intone);

            // Sequential Processing - Univariate Treatment of the Multivariate Series:
            for (int SP = d - 1; SP > -1; SP--)
            {

                // Get the specific values of Z for SP:
                for (int j = 0; j < m; j++)
                {
                    Zt_temp[j] = Zt_t[SP + j * d];
                }

                /* L_(t,i) = I_m - K_(t,i) %*% Z_(t,i) %*% F_(t,i)^-1 */

                // Step 1: L = I_m
                cblas_dcopy(m_x_m, identity_matrix, intone, L, intone);

                // Step 2: L_(t,i) = - K_(t,i) %*% Z_(t,i) + L_(t,i):
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            blas_m, blas_m, intone,
                            dblminusone, &Kt[m_x_d * t + (m * SP)], blas_m,
                            Zt_temp, blas_m,
                            dblone, L, blas_m);

                /* N_t,i-1 = t(Z_t,i) %*% F^-1 %*% Z_t,i + t(L) %*% N_t,i %*% L */
                tmp_scalar = Ft_inv[(d * t) + SP];
                // Step 1: tmpmxm = t(Z_t) %*% F^-1 %*% Z_t
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            blas_m, blas_m, intone,
                            tmp_scalar, Zt_temp, blas_m,
                            Zt_temp, blas_m,
                            dblzero, tmpmxm, blas_m);

                // Step 2: tmpN = t(L) %*% N_t,i
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            blas_m, blas_m, blas_m,
                            dblone, L, blas_m,
                            N, blas_m,
                            dblzero, tmpN, blas_m);

                // Step 3: N = tmpN %*% L
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            blas_m, blas_m, blas_m,
                            dblone, tmpN, blas_m,
                            L, blas_m,
                            dblzero, N, blas_m);

                // Step 4: N = N + tmpmxm
                cblas_daxpy(m_x_m, dblone, tmpmxm, intone, N, intone);

                /* r_t,i-1 = t(Z_t,i) %*% f_t,i^-1 %*% v_t,i + t(L_t,i) %*% r_t,i */

                // Step 1: f_t,i^-1 * v_t,i (scalar * scalar)
                tmp_scalar *= vt[(d * t) + SP];

                // Step 2: r = t(L_t,i) %*% r_t,i
                cblas_dcopy(blas_m, r, intone, tmpr, intone);
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            blas_m, intone, blas_m,
                            dblone, L, blas_m,
                            tmpr, blas_m,
                            dblzero, r, blas_m);

                // Step 3: r_t,i-1 = Zt_tmp + r:
                cblas_daxpy(blas_m, tmp_scalar, Zt_temp, intone, r, intone);

                // Rprintf("SP: %f\n", SP);
            }
        }

        // Iterate backwards through time:
        t--;
    }

    // Memory clean - vectors / matrices:
    free(tmpmxm);
    free(tmpPt);
    free(tmpN);
    free(tmpr);
    free(NAindices);
    free(positions);
    free(Zt_t);
    free(Zt_temp);
    free(Zt_NA);
    free(N);
    free(r);
    free(L);
    free(identity_matrix);
}