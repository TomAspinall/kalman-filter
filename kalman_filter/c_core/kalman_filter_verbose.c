#include <stdbool.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include <cblas.h>
#include <utils.h>

void ckalman_filter_verbose(
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
    double *loglik,
    double *att_output,
    double *Ptt_output,
    double *at_output,
    double *Pt_output,
    double *Ft_inv_output,
    double *vt_output,
    double *Kt_output)
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

        // Total Measurements (subtracted for each encountered N/A):
        int N_obs = n * d;
        double Ft;

        // Time-series iterator:
        int t = 0;

        double *Zt_t = malloc(sizeof(double) * (d * m));
        double *Zt_tSP = malloc(sizeof(double) * m);

        double *at = malloc(sizeof(double) * m);
        double *Pt = malloc(sizeof(double) * m * m);

        double *tmpmxSP = (double *)calloc(m, sizeof(double));
        double *tmpmxm = (double *)calloc(m_x_m, sizeof(double));

        /* at = x */
        cblas_dcopy(blas_m, x, intone, at, intone);

        /* Pt = P */
        cblas_dcopy(m_x_m, P, intone, Pt, intone);

        // initialise att:
        cblas_dcopy(blas_m, at, intone, &at_output[m * t + m], intone);
        // initialise Ptt:
        cblas_dcopy(blas_m, Pt, intone, &Pt_output[m * t + m], intone);

        /*Initial State Outputs:*/
        cblas_dcopy(blas_m, at, intone, &at_output[0], intone);
        cblas_dcopy(m_x_m, Pt, intone, &Pt_output[0], intone);

        // Iterate over all time steps:
        while (t < n)
        {

                /*********************************************************************************/
                /* ---------- ---------- ---------- filter step ---------- ---------- ---------- */
                /*********************************************************************************/

                // Create Zt for time t
                cblas_dcopy(m_x_d, &Zt[m_x_d * t * incZt], intone, Zt_t, intone);
                // Increment number of measurements:
                N_obs += d;

                // Sequential Processing - Univariate Treatment of the Multivariate Series:
                for (int SP = 0; SP < d; SP++)
                {
#ifdef DEBUGMODE
                        printf("SP = %i\n", SP);
#endif

                        // Missing measurements are skipped:
                        if (npy_isnan(yt[SP + d * t]))
                        {
                                // Total observations impacts final log-likelihood calculation:
                                N_obs -= 1;
                                continue;
                        }

                        // Get the specific values of Z for SP:
                        cblas_dcopy(m, &Zt_t[SP], d, Zt_tSP, 1);

#ifdef DEBUGMODE
                        print_array(Zt_tSP, m, 1, "Zt_tSP");
#endif

                        // Step 1 - Measurement Error:
                        // Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] + Zt[SP,,t * incZt] %*% at[SP,t]
                        vt_output[SP + d * t] = yt[SP + d * t] - ct[SP + d * t * incct];

#ifdef DEBUGMODE
                        printf("V = %f", vt_output[SP + d * t]);
#endif

                        // vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    intone, intone, blas_m,
                                    dblminusone, Zt_tSP, intone,
                                    at, blas_m,
                                    dblone, &vt_output[SP + d * t], intone);

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
                        Ft_inv_output[SP + d * t] = 1 / Ft;

                        // Kt is an m x 1 matrix

                        // We already have tmpSPxm:
                        // Kt = tmpmxSP %*% tmpFtinv
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    blas_m, intone, intone,
                                    dblone, tmpmxSP, blas_m,
                                    &Ft_inv_output[SP + d * t], intone,
                                    dblzero, &Kt_output[m_x_d * t + (m * SP)], blas_m);

                        // Step 4 - Correct State Vector mean and Covariance:

                        // Correction to att based upon prediction error:
                        // att = Kt %*% V + att
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    blas_m, intone, intone,
                                    dblone, &Kt_output[m_x_d * t + (m * SP)], blas_m,
                                    &vt_output[SP + d * t], intone,
                                    dblone, at, blas_m);

                        // Correction to covariance based upon Kalman Gain:
                        // ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
                        // ptt = ptt - tempmxSP %*% t(Ktt)
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    blas_m, blas_m, intone,
                                    dblminusone, tmpmxSP, blas_m,
                                    &Kt_output[m_x_d * t + (m * SP)], blas_m,
                                    dblone, Pt, blas_m);

                        // Step 5 - Update Log-Likelihood Score:
                        *loglik -= 0.5 * (log(Ft) + (vt_output[SP + d * t] * vt_output[SP + d * t] * Ft_inv_output[SP + d * t]));

#ifdef DEBUGMODE
                        printf("\n Log-Likelihood: %f \n", *loglik);
#endif
                }

                /*********************************************************************************/
                /*  ---------- ---------- ------- prediction step -------- ---------- ---------- */
                /*********************************************************************************/

                /* ---------------------------------------------------------------------- */
                /* at[,t + 1] = dt[,t * incdt] + Tt[,,t * incTt] %*% att[,t]              */
                /* ---------------------------------------------------------------------- */

#ifdef DEBUGMODE
                print_array(at, m, 1, "at:");
#endif

                // tmpmxSP = Tt[,,i * incTt] %*% att[,i]
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            blas_m, intone, blas_m,
                            dblone, &Tt[m_x_m * t * incTt], blas_m,
                            at, blas_m,
                            dblzero, tmpmxSP, blas_m);

                // save att:
                cblas_dcopy(blas_m, at, intone, &att_output[m * t], intone);
                // save ptt:
                cblas_dcopy(m_x_m, Pt, intone, &Ptt_output[m_x_m * t], intone);

                /* at[,t + 1] = dt[,t] + at[,t] */
                cblas_dcopy(blas_m, &dt[m * t * incdt], intone, at, intone);
                cblas_daxpy(blas_m, dblone, tmpmxSP, intone, at, intone);

#ifdef DEBUGMODE
                print_array(at, m, 1, "atp1:");
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
                print_array(at, m, 1, "at:");
                print_array(Pt, m, m, "Pt:");
                printf("\n---------- iteration %i ----------\n", t + 1);
#endif

                // save at:
                cblas_dcopy(blas_m, at, intone, &at_output[m * t + m], intone);
                // save pt:
                cblas_dcopy(m_x_m, Pt, intone, &Pt_output[m_x_m * t + m_x_m], intone);

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
        free(Zt_t);
        free(Zt_tSP);
        free(at);
}