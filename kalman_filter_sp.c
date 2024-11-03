#include <stdbool.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <cblas.h>
#include <math.h>

/* Macro to transform an index of a 2-dimensional array into an index of a vector */
#define IDX(i, j, dim0) (i) + (j) * (dim0)

/* Locate NA's in observations at time t*/
void locateNA(double *vec, int *NAindices, int *positions, int len)
{
    int j = 0;
    for (int i = 0; i < len; i++)
    {
        if (isnan(vec[i]))
            NAindices[i] = 1;
        else
        {
            NAindices[i] = 0;
            positions[j] = i;
            j++;
        }
    }
}

/* Number of NA's in observations at time t*/
int numberofNA(double *vec, int *NAindices, int *positions, int len)
{
    locateNA(vec, NAindices, positions, len);
    int sum = 0;
    for (int i = 0; i < len; i++)
        sum += NAindices[i];
    return sum;
}

/* Temporary reduced arrays when missing obverations are present */
void reduce_array(double *array_full, int dim0, int dim1,
                  double *array_reduced, int *pos, int len)
{
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < dim1; j++)
            array_reduced[IDX(i, j, len)] = array_full[IDX(pos[i], j, dim0)];
    }
}

// Logical function:
int ckalman_filter_sequential(
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
    int m_x_m = m * m;
    int m_x_d = m * d;

    blasint blas_m_x_m = (blasint)m_x_m;
    blasint blas_m_x_d = (blasint)m_x_d;

    // integers and double precisions used in dcopy and dgemm
    blasint intone = 1;
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
    cblas_dcopy(blas_m_x_m, P0, intone, Pt, intone);

    // Recursion across all time steps:
    while (t < n)
    {

        // How many NA's at time t?
        na_sum = numberofNA(&yt[d * t], NAindices, positions, d);
        printf("\nNumber of NAs in iter %i: %i\n", t, na_sum);

        /*****************************************/
        /* ---------- case 1: no NA's:---------- */
        /*****************************************/
        if (na_sum == 0)
        {
            // Create Zt for time t
            cblas_dcopy(blas_m_x_d, &Zt[m_x_d * t * incZt], intone, Zt_t, intone);
            // Increment number of observations:
            N_obs += d;
        }

        t++;
    }

    // for (npy_intp i = 0; i < n; i++)
    // {
    //     a0[i] += P0[i];
    //     printf("Debug: Pt is: (%f)\n", Pt[i]);
    // }

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
    if (n < m)
        return n;
    else
        return m;
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
    int d = yt_dims[0];
    // Total observations:
    int n = yt_dims[1];
    // Number of state variables:
    int m = a0_dims[0];
    // npy_intp one = 1;

    printf("Debug: n is: (%d)\n", n);
    printf("Debug: m is: (%d)\n", m);
    printf("Debug: d is: (%d)\n", d);

    // Check for consistency in array shapes:

    // Number of state variables (m):
    if (P0_dims[0] != m || P0_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions of square matrix 'Pt' do not match length of state vector 'a0'");
        return NULL;
    }
    if (dt_dims[0] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension [0] of matrix 'dt' does not match length of state vector 'a0'");
        return NULL;
    }
    if (Zt_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension [1] of matrix 'Zt' does not match length of state vector 'a0'");
        return NULL;
    }
    if (Tt_dims[0] != m || Tt_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 0 or 1 of matrix 'Tt' does not match length of state vector 'a0'");
        return NULL;
    }
    if (HHt_dims[0] != m || HHt_dims[1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 0 or 1 of matrix 'HHt' does not match length of state vector 'a0'");
        return NULL;
    }

    // Total observations (n):
    // if (dt_dims[1] != n && dt_dims[1] != one)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (ct_dims[1] != n && ct_dims[1] != one)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (Tt_dims[2] != n && Tt_dims[2] != one)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (Zt_dims[2] != n && Zt_dims[2] != one)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (HHt_dims[2] != n && HHt_dims[2] != one)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (GGt_dims[1] != n && GGt_dims[1] != one)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }

    // Max observations per time point (d):
    if (ct_dims[0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 0 of ndarray 'ct' does not equal dimension 0 of 'yt'");
        return NULL;
    }
    if (Zt_dims[0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 0 of ndarray 'Zt' does not equal dimension 0 of 'yt'");
        return NULL;
    }
    if (GGt_dims[0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 0 of ndarray 'GGt' does not equal dimension 0 of 'yt'");
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
    return Py_BuildValue("i", ckalman_filter_sequential(
                                  n,
                                  m,
                                  d,
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
