#include <stdbool.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <cblas.h>
#include <utils.h>

void ckalman_filter(
    // n: the total number of measurements
    int n,
    // m: the dimension of the state vector
    int m,
    // d: the dimension of measurements
    int d,
    // Filter State Estimate:
    double *x,
    // Covariance Matrix:
    double *P,
    double *dt, int incdt,
    double *ct, int incct,
    double *Tt, int incTt,
    double *Zt, int incZt,
    // Measurement Uncertainty / Noise:
    double *HHt, int incHHt,
    // Measurement Function:
    double *GGt, int incGGt,
    double *yt,
    // Outputs:
    double *loglik)
{
        // Instantiate:
        *loglik = 0;

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

        double *tmpmxSP = (double *)calloc(m, sizeof(double));
        double *tmpmxm = (double *)calloc(m_x_m, sizeof(double));

        // kalman_filter specific definitions:
        double V;
        double Ft;
        double tmpFtinv;
        double *Kt = malloc(sizeof(double) * m);

        /* at = x */
        cblas_dcopy(blas_m, x, intone, at, intone);

        /* Pt = P */
        cblas_dcopy(m_x_m, P, intone, Pt, intone);

        /*********************************************************************************/
        /* ---------- ----------------- Begin Kalman Filter ----------------- ---------- */
        /*********************************************************************************/

        // Iterate over all time steps:
        while (t < n)
        {

                // How many NA's at time t?
                na_sum = numberofNA(&yt[d * t], NAindices, positions, d);
#ifdef DEBUGMODE
                printf("\nNumber of NAs in iter %i: %i\n", t, na_sum);
#endif

                /*********************************************************************************/
                /* ---------- ---------- ---------- filter step ---------- ---------- ---------- */
                /*********************************************************************************/

                /*****************************************/
                /* ---------- case 1: no NA's:---------- */
                /*****************************************/
                if (na_sum == 0)
                {

                        // Create Zt for time t
                        cblas_dcopy(m_x_d, &Zt[m_x_d * t * incZt], intone, Zt_t, intone);
                        // Increment number of measurements:
                        N_obs += d;

                        // Sequential Processing - Univariate Treatment of the Multivariate Series:
                        for (int SP = 0; SP < d; SP++)
                        {
#ifdef DEBUGMODE
                                printf("SP = %i", SP);
#endif

                                // Get the specific values of Z for SP:
                                cblas_dcopy(m, &Zt_t[SP], d, Zt_tSP, 1);

                                // Step 1 - Measurement Error:
                                // Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] + Zt[SP,,t * incZt] %*% at[SP,t]
                                V = yt[SP + d * t] - ct[SP + d * t * incct];

#ifdef DEBUGMODE
                                print_array(Zt_tSP, 1, m, "Zt_tSP");
#endif

                                // vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            intone, intone, blas_m,
                                            dblminusone, Zt_tSP, intone,
                                            at, blas_m,
                                            dblone, &V, intone);

                                // Step 2 - Function of Covariance Matrix:
                                // Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]

                                // First, Let us calculate:
                                // tmpmxSP = Pt %*% t(Zt[SP,,t * incZt])
                                // because we use this result twice
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                            blas_m, intone, blas_m,
                                            dblone, Pt, blas_m,
                                            Zt_tSP, intone,
                                            dblzero, tmpmxSP, blas_m);

                                // Ft = GGt[SP]
                                Ft = GGt[SP + (d * t * incGGt)];

                                // Ft = Zt[SP,,t*incZt] %*% tmpmxSP + Ft
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            intone, intone, blas_m,
                                            dblone, Zt_tSP, intone,
                                            tmpmxSP, blas_m,
                                            dblone, &Ft, intone);

                                // Step 3 - Calculate the Kalman Gain:
                                // Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)

                                // Inv Ft:
                                tmpFtinv = 1 / Ft;

                                // Kt is an m x 1 matrix

                                // We already have tmpSPxm:
                                // Kt = tmpmxSP %*% tmpFtinv
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            blas_m, intone, intone,
                                            dblone, tmpmxSP, blas_m,
                                            &tmpFtinv, intone,
                                            dblzero, Kt, blas_m);

                                // Step 4 - Correct State Vector mean and Covariance:

                                // Correction to att based upon prediction error:
                                // att = Kt %*% V + att
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            blas_m, intone, intone,
                                            dblone, Kt, blas_m,
                                            &V, intone,
                                            dblone, at, blas_m);

                                // Correction to covariance based upon Kalman Gain:
                                // ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
                                // ptt = ptt - tempmxSP %*% t(Ktt)
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                            blas_m, blas_m, intone,
                                            dblminusone, tmpmxSP, blas_m,
                                            Kt, blas_m,
                                            dblone, Pt, blas_m);

                                // Step 5 - Update Log-Likelihood Score:
                                *loglik -= 0.5 * (log(Ft) + (V * V * tmpFtinv));

#ifdef DEBUGMODE
                                printf("\n Log-Likelihood: %f \n", *loglik);
#endif
                        }
                }
                /*******************************************/
                /* ---------- case 2: some NA's ---------- */
                /*******************************************/
                else
                {
                        // Total measurements this time period:
                        int d_reduced = d - na_sum;
                        N_obs += d_reduced;

                        // Temporary, reduced arrays:
                        reduce_array(&yt[d * t], d, 1, yt_temp, positions, d_reduced);
                        reduce_array(&ct[d * t * incct], d, 1, ct_temp, positions, d_reduced);
                        reduce_array(&Zt[m_x_d * t * incZt], d, m, Zt_temp, positions, d_reduced);
                        reduce_array(&GGt[d * t * incGGt], d, 1, GGt_temp, positions, d_reduced);

                        // Sequential Processing - Univariate Treatment of the Multivariate Series:
                        for (int SP = 0; SP < d_reduced; SP++)
                        {
                                // Get the specific values of Z for SP:
                                cblas_dcopy(m, &Zt_t[SP], d_reduced, Zt_tSP, 1);

                                // Step 1 - Measurement Error:
                                // Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] - Zt[SP,,t * incZt] %*% at[SP,t]
                                V = yt_temp[SP] - ct_temp[SP];
                                // vt[SP, t] = vt[SP, t] - Zt[SP, , t * incZt] % *% at[, t]
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            intone, intone, blas_m,
                                            dblminusone, Zt_tSP, intone,
                                            at, blas_m,
                                            dblone, &V, intone);

                                // Step 2 - Function of Covariance Matrix:
                                // Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]
                                // Where GGt is already input within this algorithm as the diagonals of the variance / covariance matrix.

                                // First, Let us calculate:
                                // tmpmxSP = Pt %*% t(Zt[SP,,t * incZt])
                                // because we use this result twice
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                            blas_m, intone, blas_m,
                                            dblone, Pt, blas_m,
                                            Zt_tSP, intone,
                                            dblzero, tmpmxSP, blas_m);

                                // Ft = GGt[SP]
                                Ft = GGt_temp[SP];

                                // Ft = Zt[SP,,t*incZt] %*% tmpmxSP + Ft
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            intone, intone, blas_m,
                                            dblone, Zt_tSP, intone,
                                            tmpmxSP, blas_m,
                                            dblone, &Ft, intone);

                                // Inv Ft:
                                tmpFtinv = 1 / Ft;

                                // Kt is an m x 1 matrix

                                // We already have tmpSPxm:
                                // Kt = tmpmxSP %*% tmpFtinv
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            blas_m, intone, intone,
                                            dblone, tmpmxSP, blas_m,
                                            &tmpFtinv, intone,
                                            dblzero, Kt, blas_m);

                                // Step 4 - Correct State Vector mean and Covariance:

                                // Correction to att based upon prediction error:
                                // att = Kt %*% V + att
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            blas_m, intone, intone,
                                            dblone, Kt, blas_m,
                                            &V, intone,
                                            dblone, at, blas_m);

                                // Correction to covariance based upon Kalman Gain:
                                // ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
                                // ptt = ptt - tempmxSP %*% t(Ktt)
                                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                            blas_m, blas_m, intone,
                                            dblminusone, tmpmxSP, blas_m,
                                            Kt, blas_m,
                                            dblone, Pt, blas_m);

                                // Step 5 - Update Log-Likelihood Score:
                                *loglik -= 0.5 * (log(Ft) + (V * V * tmpFtinv));
                        }
                }

                /*********************************************************************************/
                /*  ---------- ---------- ------- prediction step -------- ---------- ---------- */
                /*********************************************************************************/

                /* ---------------------------------------------------------------------- */
                /* at[,t + 1] = dt[,t * incdt] + Tt[,,t * incTt] %*% att[,t]              */
                /* ---------------------------------------------------------------------- */

#ifdef DEBUGMODE
                print_array(at, 1, m, "at:");
#endif

                // tmpmxSP = Tt[,,i * incTt] %*% att[,i]
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            blas_m, intone, blas_m,
                            dblone, &Tt[m_x_m * t * incTt], blas_m,
                            at, blas_m,
                            dblzero, tmpmxSP, blas_m);

                /* at[,t + 1] = dt[,t] + at[,t] */
                cblas_dcopy(blas_m, &dt[m * t * incdt], intone, at, intone);
                cblas_daxpy(blas_m, dblone, tmpmxSP, intone, at, intone);

#ifdef DEBUGMODE
                print_array(at, 1, m, "atp1:");
                print_array(Pt, m, m, "Pt:");
#endif

                /* ------------------------------------------------------------------------------------- */
                /* Pt[,,t + 1] = Tt[,,t * incTt] %*% Ptt[,,t] %*% t(Tt[,,t * incTt]) + HHt[,,t * incHHt] */
                /* ------------------------------------------------------------------------------------- */

                /* tmpmxm = Ptt[,,i] %*% t(Tt[,,i * incTt]) */
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            blas_m, blas_m, blas_m,
                            dblone, Pt, blas_m,
                            &Tt[m_x_m * t * incTt], blas_m,
                            dblzero, tmpmxm, blas_m);

                /* Pt[,,i + 1] = HHt[,,i * incHHt] */
                cblas_dcopy(m_x_m, &HHt[m_x_m * t * incHHt], intone, Pt, intone);

#ifdef DEBUGMODE
                print_array(&HHt[m_x_m * t * incHHt], m, m, "HHt:");
#endif

                /* Pt[,,i + 1] = Tt[,,i * incTt] %*% tmpmxm + Pt[,,i + 1] */
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            blas_m, blas_m, blas_m,
                            dblone, &Tt[m_x_m * t * incTt], blas_m,
                            tmpmxm, blas_m,
                            dblone, Pt, blas_m);

#ifdef DEBUGMODE
                print_array(at, 1, m, "at:");
                print_array(Pt, m, m, "Pt:");
                printf("\n---------- iteration %i ----------\n", t + 1);
#endif

                // Iterate:
                t++;
        }

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
}