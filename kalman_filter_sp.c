#include <stdbool.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <cblas.h>
#include <utils.h>

// Logical function:
double ckalman_filter_sequential(
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

    blasint blas_n = (blasint)n;
    blasint blas_m = (blasint)m;
    blasint blas_d = (blasint)d;

    // Utilised array dimensions:
    blasint m_x_m = m * m;
    blasint m_x_d = m * d;

    // blasint blas_m_x_m = (blasint)m_x_m;
    // blasint blas_m_x_d = (blasint)m_x_d;

    // integers and double precisions used in dcopy and dgemm
    blasint intone = 1;
    blasint intminusone = -1;
    double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;
    // To transpose or not transpose matrix
    char *transpose = "T", *dont_transpose = "N";

    // Sequential Processing variables:
    int N_obs = 0;
    int na_sum;
    double V;
    double Ft;
    double tmpFtinv;

    // Function output:
    double loglik = 0;

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

    // Recursion across all time steps:
    while (t < n)
    {

        // How many NA's at time t?
        na_sum = numberofNA(&yt[d * t], NAindices, positions, d);
        // printf("\nNumber of NAs in iter %i: %i\n", t, na_sum);

        /*****************************************/
        /* ---------- case 1: no NA's:---------- */
        /*****************************************/
        if (na_sum == 0)
        {
            // Create Zt for time t
            cblas_dcopy(m_x_d, &Zt[m_x_d * t * incZt], intone, Zt_t, intone);
            // Increment number of observations:
            N_obs += d;

            // Sequential Processing - Univariate Treatment of the Multivariate Series:
            for (int SP = 0; SP < d; SP++)
            {
                // Get the specific values of Z for SP:
                for (int j = 0; j < m; j++)
                {
                    Zt_tSP[j] = Zt_t[SP + j * d];
                }
                // Step 1 - Measurement Error:
                // Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] - Zt[SP,,t * incZt] %*% at[SP,t]

                // vt[SP,t] = yt[SP,t] - ct[SP,t * incct]
                V = yt[SP + d * t] - ct[SP + d * t * incct];

                // vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    intone,
                    intone,
                    blas_m,
                    dblminusone,
                    Zt_tSP,
                    intone,
                    at,
                    blas_m,
                    dblone,
                    &V,
                    intone);

                // Step 2 - Function of Covariance Matrix:
                // Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]

                // First, Let us calculate:
                // Pt %*% t(Zt[SP,,t * incZt])
                // because we use this result twice
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    blas_m,
                    intone,
                    blas_m,
                    dblone,
                    Pt,
                    blas_m,
                    Zt_tSP,
                    intone,
                    dblzero,
                    tmpmxSP,
                    blas_m);

                // Ft = GGt[SP]
                Ft = GGt[SP + (d * t * incGGt)];

                // Ft = Zt[SP,,t*incZt] %*% tmpmxSP + Ft
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    intone,
                    intone,
                    blas_m,
                    dblone,
                    Zt_tSP,
                    intone,
                    tmpmxSP,
                    blas_m,
                    dblone,
                    &Ft,
                    intone);

                // Step 3 - Calculate the Kalman Gain:
                // Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)

                // Inv Ft:
                tmpFtinv = 1 / Ft;

                // Kt is an m x 1 matrix

                // We already have tmpSPxm:
                // Kt = tmpmxSP %*% tmpFtinv
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    blas_m,
                    intone,
                    intone,
                    dblone,
                    tmpmxSP,
                    blas_m,
                    &tmpFtinv,
                    intone,
                    dblzero,
                    Kt,
                    blas_m);

                // Step 4 - Correct State Vector mean and Covariance:

                // Correction to att based upon prediction error:
                // att = Kt %*% V + att
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    blas_m,
                    intone,
                    intone,
                    dblone,
                    Kt,
                    blas_m,
                    &V,
                    intone,
                    dblone,
                    at,
                    blas_m);

                // Correction to covariance based upon Kalman Gain:
                // ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
                // ptt = ptt - tempmxSP %*% t(Ktt)
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    blas_m,
                    blas_m,
                    intone,
                    dblminusone,
                    tmpmxSP,
                    blas_m,
                    Kt,
                    blas_m,
                    dblone,
                    Pt,
                    blas_m);
            }
        }
        /*******************************************/
        /* ---------- case 2: some NA's ---------- */
        /*******************************************/
        else
        {
            int d_reduced = d - na_sum;
            // Increment number of observations for the Log-likelihood at the end:
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
                for (int j = 0; j < m; j++)
                {
                    Zt_tSP[j] = Zt_temp[SP + j * d_reduced];
                }

                // Step 1 - Measurement Error:
                // Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] + Zt[SP,,t * incZt] %*% at[SP,t]

                // vt[SP,t] = yt[SP,t] - ct[SP,t * incct]
                V = yt_temp[SP] - ct_temp[SP];

                // vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    intone,
                    intone,
                    blas_m,
                    dblminusone,
                    Zt_tSP,
                    intone,
                    at,
                    blas_m,
                    dblone,
                    &V,
                    intone);

                // Step 2 - Function of Covariance Matrix:
                // Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]

                // First, Let us calculate:
                // Pt %*% t(Zt[SP,,t * incZt])
                // because we use this result twice

                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    blas_m,
                    intone,
                    blas_m,
                    dblone,
                    Pt,
                    blas_m,
                    Zt_tSP,
                    intone,
                    dblzero,
                    tmpmxSP,
                    blas_m);

                // Ft = GGt[SP]
                Ft = GGt_temp[SP];

                // Ft = Zt[SP,,i*incZt] %*% tmpmxSP + Ft
                // Ft = Zt[SP,,t*incZt] %*% tmpmxSP + Ft
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    intone,
                    intone,
                    blas_m,
                    dblone,
                    Zt_tSP,
                    intone,
                    tmpmxSP,
                    blas_m,
                    dblone,
                    &Ft,
                    intone);

                // Step 3 - Calculate the Kalman Gain:
                // Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)

                // Inv Ft:
                tmpFtinv = 1 / Ft;

                // Kt is an m x 1 matrix

                // We already have tmpSPxm:
                // Kt = tmpmxSP %*% tmpFtinv
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    blas_m,
                    intone,
                    intone,
                    dblone,
                    tmpmxSP,
                    blas_m,
                    &tmpFtinv,
                    intone,
                    dblzero,
                    Kt,
                    blas_m);

                // Step 4 - Correct State Vector mean and Covariance:

                // Correction to att based upon prediction error:
                // att = Kt %*% V + att
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    blas_m,
                    intone,
                    intone,
                    dblone,
                    Kt,
                    blas_m,
                    &V,
                    intone,
                    dblone,
                    at,
                    blas_m);

                // Correction to covariance based upon Kalman Gain:
                // ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
                // ptt = ptt - tempmxSP %*% t(Ktt)
                cblas_dgemm(
                    CblasColMajor,
                    CblasNoTrans,
                    CblasTrans,
                    blas_m,
                    blas_m,
                    intone,
                    dblminusone,
                    tmpmxSP,
                    blas_m,
                    Kt,
                    blas_m,
                    dblone,
                    Pt,
                    blas_m);
            }
            // Step 5 - Update Log-Likelihood Score:
            loglik -= 0.5 * (log(Ft) + (V * V * tmpFtinv));
        }

        /*********************************************************************************/
        /*  ---------- ---------- ------- prediction step -------- ---------- ---------- */
        /*********************************************************************************/

        /* ---------------------------------------------------------------------- */
        /* at[,t + 1] = dt[,t * incdt] + Tt[,,t * incTt] %*% att[,t]              */
        /* ---------------------------------------------------------------------- */

        // tmpmxm = Tt[,,i * incTt] %*% att[,i]
        cblas_dgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            blas_m,
            intone,
            blas_m,
            dblone,
            &Tt[m_x_m * t * incTt],
            blas_m,
            at,
            blas_m,
            dblzero,
            tmpmxSP,
            blas_m);

        /* at[,t + 1] = dt[,t] */
        cblas_dcopy(blas_m, &dt[m * t * incdt], intone, at, intone);

        cblas_daxpy(blas_m, dblone, tmpmxSP, intone, at, intone);

        /* ------------------------------------------------------------------------------------- */
        /* Pt[,,t + 1] = Tt[,,t * incTt] %*% Ptt[,,t] %*% t(Tt[,,t * incTt]) + HHt[,,t * incHHt] */
        /* ------------------------------------------------------------------------------------- */

        /* tmpmxm = Ptt[,,i] %*% t(Tt[,,i * incTt]) */
        cblas_dgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasTrans,
            blas_m,
            blas_m,
            blas_m,
            dblone,
            Pt,
            blas_m,
            &Tt[m_x_m * t * incTt],
            blas_m,
            dblzero,
            tmpmxm,
            blas_m);

        /* Pt[,,i + 1] = HHt[,,i * incHHt] */
        cblas_dcopy(m_x_m, &HHt[m_x_m * t * incHHt], intone, Pt, intone);

        /* Pt[,,i + 1] = Tt[,,i * incTt] %*% tmpmxm + Pt[,,i + 1] */
        cblas_dgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            blas_m,
            blas_m,
            blas_m,
            dblone,
            &Tt[m_x_m * t * incTt],
            blas_m,
            tmpmxm,
            blas_m,
            dblone,
            Pt,
            blas_m);

        // Iterate:
        t++;
    }
    /**************************************************************/
    /* ---------- ---------- end recursions ---------- ---------- */
    /**************************************************************/

    // Update the final Log-Likelihood Score:
    loglik -= 0.5 * N_obs * log(2 * M_PI);

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

    // PyArray_malloc();
    return loglik;
}

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter(PyObject *self, PyObject *args)
{

    // Initialise variable arguments:
    PyObject *input_a0, *input_P0, *input_dt, *input_ct, *input_Tt, *input_Zt, *input_HHt, *input_GGt, *input_yt;

    /* Parse the input tuple, expecting 8 NumPy ndarrays */
    if (!PyArg_ParseTuple(args, "OOOOOOOOO", &input_a0, &input_P0, &input_dt, &input_ct, &input_Tt, &input_Zt, &input_HHt, &input_GGt, &input_yt))
    {
        return NULL;
    }

    // Ensure all objects are NumPy arrays:
    if (!PyArray_Check(input_a0))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'a0' is not of type numpy.array");
        return NULL;
    }
    if (!PyArray_Check(input_P0))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'P0' is not of type numpy.ndarray");
        return NULL;
    }
    if (!PyArray_Check(input_dt))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'dt' is not of type numpy.ndarray");
        return NULL;
    }
    if (!PyArray_Check(input_ct))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'ct' is not of type numpy.ndarray");
        return NULL;
    }
    if (!PyArray_Check(input_Tt))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'Tt' is not of type numpy.ndarray");
        return NULL;
    }
    if (!PyArray_Check(input_Zt))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'Zt' is not of type numpy.ndarray");
        return NULL;
    }
    if (!PyArray_Check(input_HHt))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'HHt' is not of type numpy.ndarray");
        return NULL;
    }
    if (!PyArray_Check(input_GGt))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'GGt' is not of type numpy.ndarray");
        return NULL;
    }
    if (!PyArray_Check(input_yt))
    {
        PyErr_SetString(PyExc_TypeError, "argument 'yt' is not of type numpy.ndarray");
        return NULL;
    }

    // Get the NumPy array data and dimensions
    PyArrayObject *a0_arr = (PyArrayObject *)input_a0;
    PyArrayObject *P0_arr = (PyArrayObject *)input_P0;
    PyArrayObject *dt_arr = (PyArrayObject *)input_dt;
    PyArrayObject *ct_arr = (PyArrayObject *)input_ct;
    PyArrayObject *Tt_arr = (PyArrayObject *)input_Tt;
    PyArrayObject *Zt_arr = (PyArrayObject *)input_Zt;
    PyArrayObject *HHt_arr = (PyArrayObject *)input_HHt;
    PyArrayObject *GGt_arr = (PyArrayObject *)input_GGt;
    PyArrayObject *yt_arr = (PyArrayObject *)input_yt;

    // Check that array shapes are consistent:
    if (PyArray_NDIM(a0_arr) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "'a0' is not 1-dimensional");
        return NULL;
    }
    if (PyArray_NDIM(P0_arr) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'P0' is not 2-dimensional");
        return NULL;
    }
    if (PyArray_NDIM(yt_arr) > 2)
    {
        PyErr_SetString(PyExc_ValueError, "'yt' is not 1 or 2-dimensional");
        return NULL;
    }

    // Fetch array dimension sizes:
    npy_intp *a0_dims = PyArray_DIMS(a0_arr);
    npy_intp *P0_dims = PyArray_DIMS(P0_arr);
    npy_intp *dt_dims = PyArray_DIMS(dt_arr);
    npy_intp *ct_dims = PyArray_DIMS(ct_arr);
    npy_intp *Tt_dims = PyArray_DIMS(Tt_arr);
    npy_intp *Zt_dims = PyArray_DIMS(Zt_arr);
    npy_intp *HHt_dims = PyArray_DIMS(HHt_arr);
    npy_intp *GGt_dims = PyArray_DIMS(GGt_arr);
    npy_intp *yt_dims = PyArray_DIMS(yt_arr);

    // Max observations per time point:
    npy_intp d = yt_dims[0];
    // Total observations:
    npy_intp n = yt_dims[1];
    // Number of state variables:
    npy_intp m = a0_dims[0];
    // npy_intp one = 1;

    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);

    // Check for consistency in array shapes:

    // Number of state variables (m):
    if (P0_dims[0] != m || P0_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions of square matrix 'Pt' do not match length of state vector 'a0'");
        return NULL;
    }
    if (dt_dims[0] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of matrix 'dt' does not match length of state vector 'a0'");
        return NULL;
    }
    if (Zt_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of matrix 'Zt' does not match length of state vector 'a0'");
        return NULL;
    }
    if (Tt_dims[0] != m || Tt_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 1 or 2 of matrix 'Tt' does not match length of state vector 'a0'");
        return NULL;
    }
    if (HHt_dims[0] != m || HHt_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 1 or 2 of matrix 'HHt' does not match length of state vector 'a0'");
        return NULL;
    }

    // Total observations (n):
    if (dt_dims[1] != n && dt_dims[1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (ct_dims[1] != n && ct_dims[1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'ct' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (PyArray_NDIM(Tt_arr) > 2 && Tt_dims[2] != n && Tt_dims[2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Tt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (PyArray_NDIM(Zt_arr) > 2 && Zt_dims[2] != n && Zt_dims[2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Zt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (PyArray_NDIM(HHt_arr) > 2 && HHt_dims[2] != n && HHt_dims[2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'HHt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (GGt_dims[1] != n && GGt_dims[1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'GGt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }

    // Max observations per time point (d):
    if (ct_dims[0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'ct' does not equal dimension 0 of 'yt'");
        return NULL;
    }
    if (Zt_dims[0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'Zt' does not equal dimension 0 of 'yt'");
        return NULL;
    }
    if (GGt_dims[0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'GGt' does not equal dimension 0 of 'yt'");
        return NULL;
    }

    // Fetch increment logic:
    int incdt = dt_dims[1] == n;
    int incct = ct_dims[1] == n;
    int incTt = Tt_dims[2] == n;
    int incZt = Zt_dims[2] == n;
    int incHHt = HHt_dims[2] == n;
    int incGGt = GGt_dims[1] == n;

    // Fetch data pointers:
    double *a0 = (double *)PyArray_DATA(a0_arr);
    double *P0 = (double *)PyArray_DATA(P0_arr);
    double *dt = (double *)PyArray_DATA(dt_arr);
    double *ct = (double *)PyArray_DATA(ct_arr);
    double *Tt = (double *)PyArray_DATA(Tt_arr);
    double *Zt = (double *)PyArray_DATA(Zt_arr);
    double *HHt = (double *)PyArray_DATA(HHt_arr);
    double *GGt = (double *)PyArray_DATA(GGt_arr);
    double *yt = (double *)PyArray_DATA(yt_arr);

    // Call the C function
    // Py_RETURN_NONE; // No return value
    /* Construct the result: a Python integer object */
    return Py_BuildValue("d", ckalman_filter_sequential(
                                  (int)n,
                                  (int)m,
                                  (int)d,
                                  a0,
                                  P0,
                                  dt, incdt,
                                  ct, incct,
                                  Tt, incTt,
                                  Zt, incZt,
                                  HHt, incHHt,
                                  GGt, incGGt,
                                  yt));
}

/* Define functions in module */
static PyMethodDef KalmanFilterMethods[] = {
    {"kalman_filter", kalman_filter, METH_VARARGS, "Perform the Kalman filter algorithm through Sequential Processing"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Create PyModuleDef structure */
static struct PyModuleDef KalmanFilterStruct = {
    PyModuleDef_HEAD_INIT,
    "kalman_filter",     // name of module
    "Kalman filtering ", // module documentation
    -1,                  // size of per-interpreter state of the module
    KalmanFilterMethods,
    NULL,
    NULL,
    NULL,
    NULL};

/* Module initialization */
PyObject *PyInit_kalman_filter(void)
{

    import_array(); // Inititalise NumPy C API
    if (PyErr_Occurred())
    {
        printf("Failed to import numpy Python module(s).");
        return NULL;
    }

    return PyModule_Create(&KalmanFilterStruct);
}
