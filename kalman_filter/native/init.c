#include <Python.h>
#include "numpy/arrayobject.h"
#include "kalman_filter_verbose.h"
#include "kalman_filter.h"
#include "kalman_smoother.h"
#include "utils.h"

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter(PyObject *self, PyObject *args)
{

    // Valid dictionary input:
    PyObject *kwargs;
    if (!PyArg_ParseTuple(args, "O", &kwargs))
    {
        return NULL;
    }
    if (!PyDict_Check(kwargs))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    // Input dictionary keys:
    const char *keys[] = {
        "x", "P", "yt", "dt", "ct", "GGt", "Tt", "Zt", "HHt"};
    const int total_keys = 9;
    // ndarrays:
    PyArrayObject *ndarrays[9];
    // ndarray dims:
    npy_intp *array_dims[9];
    // ndarray number of dimensions:
    int array_ndims[9];

    // Extract NumPy arrays from input dictionary:
    for (int i = 0; i < total_keys; i++)
    {
        PyObject *item = PyDict_GetItemString(kwargs, keys[i]);
        if (item == NULL)
        {
            PyErr_Format(PyExc_KeyError, "Dictionary is missing '%s' key", keys[i]);
            return NULL;
        }
        if (!PyArray_Check(item))
        {
            PyErr_Format(PyExc_TypeError, "'%s' is not a valid numpy.ndarray object", keys[i]);
            return NULL;
        }
        // Fetch ndarray object:
        ndarrays[i] = (PyArrayObject *)item;
        // Fetch dimension sizes:
        array_dims[i] = PyArray_DIMS(ndarrays[i]);
        // Fetch dimension size:
        array_ndims[i] = PyArray_NDIM(ndarrays[i]);
    }

    // Check that array shapes are consistent:
    if (array_ndims[0] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "'x' is not 1-dimensional");
        return NULL;
    }
    if (array_ndims[1] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'P' is not 2-dimensional");
        return NULL;
    }
    if (array_ndims[2] != 1 && array_ndims[2] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'yt' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[3] != 1 && array_ndims[3] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'dt' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[4] != 1 && array_ndims[4] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'ct' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[5] != 1 && array_ndims[5] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'GGt' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[6] != 2 && array_ndims[6] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Tt' is not 2 or 3-dimensional");
        return NULL;
    }
    if (array_ndims[7] != 2 && array_ndims[7] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Zt' is not 2 or 3-dimensional");
        return NULL;
    }
    if (array_ndims[8] != 2 && array_ndims[8] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'HHt' is not 2 or 3-dimensional");
        return NULL;
    }

    // Number of state variables - x dim[0]:
    npy_intp m = array_dims[0][0];
    int int_m = (int)m;

    // Allow 1- or 2-dimensional values for y:
    npy_intp n;
    npy_intp d;
    int int_d;
    int int_n;
    if (array_ndims[2] == 2)
    {
        // Max observations per time point - yt dim[0]:
        d = array_dims[2][0];
        int_d = (int)d;
        n = array_dims[2][1];
        int_n = (int)n;
    }
    else
    {
        int_d = 1;
        d = (npy_intp)int_d;
        n = array_dims[2][0];
        int_n = (int)n;
    }

    // Check for consistency in array shapes:

    // Number of state variables (m):
    if (array_dims[1][0] != m || array_dims[1][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions of square matrix 'Pt' do not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[3][0] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of matrix 'dt' does not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[6][0] != m || array_dims[6][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 1 or 2 of matrix 'Tt' does not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[7][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of matrix 'Zt' does not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[8][0] != m || array_dims[8][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 1 or 2 of matrix 'HHt' does not match length of state vector 'x'");
        return NULL;
    }

    // Total observations (n):
    if (array_dims[3][1] != n && array_dims[3][1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_dims[4][1] != n && array_dims[4][1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'ct' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_dims[5][1] != n && array_dims[5][1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'GGt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_ndims[6] > 2 && array_dims[6][2] != n && array_dims[6][2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Tt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_ndims[7] > 2 && array_dims[7][2] != n && array_dims[7][2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Zt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_ndims[8] > 2 && array_dims[8][2] != n && array_dims[8][2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'HHt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }

    // Max observations per time point (d):
    if (array_dims[4][0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'ct' does not equal dimension 0 of 'yt'");
        return NULL;
    }
    if (array_dims[7][0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'Zt' does not equal dimension 0 of 'yt'");
        return NULL;
    }

    // Support for either 1- or 2-dimensional:
    int incGGt = 0;
    if (array_ndims[5] > 1)
    {
        if (array_dims[5][1] == 1)
        {
            incGGt = 0;
        }
        else if (
            array_dims[5][1] == int_n)
        {
            incGGt = 1;
        }
        else
        {
            PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'GGt' does not equal the number of observations of 'yt'");
            return NULL;
        }
    }

    // Support for either 2- or 3-dimensional:
    int incHHt = 0;
    if (array_ndims[8] > 2)
    {
        if (array_dims[8][2] == 1)
        {
            incHHt = 0;
        }
        else if (
            array_dims[8][2] == int_n)
        {
            incHHt = 1;
        }
        else
        {
            PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'HHt' does not equal the number of observations of 'yt'");
            return NULL;
        }
    }

#ifdef DEBUGMODE
    // Print algorithm dimensions:
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);

    // Print input dimensions:
    print_npy_intp_array(x_dims, 1, 1, "x_dims");
    print_npy_intp_array(P_dims, 1, 2, "P_dims");
    print_npy_intp_array(dt_dims, 1, 2, "dt_dims");
    print_npy_intp_array(ct_dims, 1, 2, "ct_dims");
    print_npy_intp_array(Tt_dims, 1, 3, "Tt_dims");
    print_npy_intp_array(Zt_dims, 1, 3, "Zt_dims");
    print_npy_intp_array(HHt_dims, 1, 3, "HHt_dims");
    print_npy_intp_array(GGt_dims, 1, 2, "GGt_dims");
    print_npy_intp_array(yt_dims, 1, 2, "yt_dims");

#endif

    // Fetch increment logic:

    int incdt = array_dims[3][1] == n;
    int incct = array_dims[4][1] == n;
    int incTt = array_dims[5][2] == n;
    int incZt = array_dims[6][2] == n;

#ifdef DEBUGMODE
    // Print time variant increments:
    printf("incdt: %d\n", incdt);
    printf("incct: %d\n", incct);
    printf("incTt: %d\n", incTt);
    printf("incZt: %d\n", incZt);
    printf("incHHt: %d\n", incHHt);
    printf("incGGt: %d\n", incGGt);
#endif

    // Fetch input data pointers:
    double *x = (double *)PyArray_DATA(ndarrays[0]);
    double *P = (double *)PyArray_DATA(ndarrays[1]);
    double *yt = (double *)PyArray_DATA(ndarrays[2]);
    double *dt = (double *)PyArray_DATA(ndarrays[3]);
    double *ct = (double *)PyArray_DATA(ndarrays[4]);
    double *GGt = (double *)PyArray_DATA(ndarrays[5]);
    double *Tt = (double *)PyArray_DATA(ndarrays[6]);
    double *Zt = (double *)PyArray_DATA(ndarrays[7]);
    double *HHt = (double *)PyArray_DATA(ndarrays[8]);

#ifdef DEBUGMODE
    // Print arrays:
    print_array(x, 1, int_m, "x");
    print_array(P, int_m, int_m, "P");
    print_array(dt, int_m, 1, "dt");
    print_array(ct, int_d, 1, "ct");
    print_array_3D(Tt, int_m, int_m, 1, "Tt");
    print_array_3D(Zt, int_d, int_m, 1, "Zt");
    print_array(yt, int_d, int_n, "yt");
#endif

    // Generate output data pointers:
    double *loglik = (double *)malloc(sizeof(double));

    // Call the C function
    ckalman_filter(
        // Inputs:
        int_n,
        int_m,
        int_d,
        x,
        P,
        dt, incdt,
        ct, incct,
        Tt, incTt,
        Zt, incZt,
        HHt, incHHt,
        GGt, incGGt,
        yt,
        // Outputs:
        loglik);

    // Return log likelihood output:
    return PyFloat_FromDouble(*loglik);
}

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter_verbose(PyObject *self, PyObject *args)
{

    // Valid dictionary input:
    PyObject *kwargs;
    if (!PyArg_ParseTuple(args, "O", &kwargs))
    {
        return NULL;
    }
    if (!PyDict_Check(kwargs))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    // Input dictionary keys:
    const char *keys[] = {
        "x", "P", "yt", "dt", "ct", "GGt", "Tt", "Zt", "HHt"};
    const int total_keys = 9;
    // ndarrays:
    PyArrayObject *ndarrays[9];
    // ndarray dims:
    npy_intp *array_dims[9];
    // ndarray number of dimensions:
    int array_ndims[9];

    // Extract NumPy arrays from input dictionary:
    for (int i = 0; i < total_keys; i++)
    {
        PyObject *item = PyDict_GetItemString(kwargs, keys[i]);
        if (item == NULL)
        {
            PyErr_Format(PyExc_KeyError, "Dictionary is missing '%s' key", keys[i]);
            return NULL;
        }
        if (!PyArray_Check(item))
        {
            PyErr_Format(PyExc_TypeError, "'%s' is not a valid numpy.ndarray object", keys[i]);
            return NULL;
        }
        // Fetch ndarray object:
        ndarrays[i] = (PyArrayObject *)item;
        // Fetch dimension sizes:
        array_dims[i] = PyArray_DIMS(ndarrays[i]);
        // Fetch dimension size:
        array_ndims[i] = PyArray_NDIM(ndarrays[i]);
    }

    // Check that array shapes are consistent:
    if (array_ndims[0] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "'x' is not 1-dimensional");
        return NULL;
    }
    if (array_ndims[1] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'P' is not 2-dimensional");
        return NULL;
    }
    if (array_ndims[2] != 1 && array_ndims[2] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'yt' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[3] != 1 && array_ndims[3] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'dt' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[4] != 1 && array_ndims[4] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'ct' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[5] != 1 && array_ndims[5] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'GGt' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[6] != 2 && array_ndims[6] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Tt' is not 2 or 3-dimensional");
        return NULL;
    }
    if (array_ndims[7] != 2 && array_ndims[7] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Zt' is not 2 or 3-dimensional");
        return NULL;
    }
    if (array_ndims[8] != 2 && array_ndims[8] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'HHt' is not 2 or 3-dimensional");
        return NULL;
    }

    // Number of state variables - x dim[0]:
    npy_intp m = array_dims[0][0];
    int int_m = (int)m;

    // Allow 1- or 2-dimensional values for y:
    npy_intp n;
    npy_intp d;
    int int_d;
    int int_n;
    if (array_ndims[2] == 2)
    {
        // Max observations per time point - yt dim[0]:
        d = array_dims[2][0];
        int_d = (int)d;
        n = array_dims[2][1];
        int_n = (int)n;
    }
    else
    {
        int_d = 1;
        d = (npy_intp)int_d;
        n = array_dims[2][0];
        int_n = (int)n;
    }

    // Check for consistency in array shapes:

    // Number of state variables (m):
    if (array_dims[1][0] != m || array_dims[1][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions of square matrix 'Pt' do not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[3][0] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of matrix 'dt' does not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[6][0] != m || array_dims[6][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 1 or 2 of matrix 'Tt' does not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[7][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of matrix 'Zt' does not match length of state vector 'x'");
        return NULL;
    }
    if (array_dims[8][0] != m || array_dims[8][1] != m)
    {
        PyErr_SetString(PyExc_ValueError, "dimensions 1 or 2 of matrix 'HHt' does not match length of state vector 'x'");
        return NULL;
    }

    // Total observations (n):
    if (array_dims[3][1] != n && array_dims[3][1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'dt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_dims[4][1] != n && array_dims[4][1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'ct' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_dims[5][1] != n && array_dims[5][1] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'GGt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_ndims[6] > 2 && array_dims[6][2] != n && array_dims[6][2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Tt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_ndims[7] > 2 && array_dims[7][2] != n && array_dims[7][2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Zt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }
    if (array_ndims[8] > 2 && array_dims[8][2] != n && array_dims[8][2] != 1)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'HHt' does not match either 1 or number of observations/columns of 'yt'");
        return NULL;
    }

    // Max observations per time point (d):
    if (array_dims[4][0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'ct' does not equal dimension 0 of 'yt'");
        return NULL;
    }
    if (array_dims[7][0] != d)
    {
        PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'Zt' does not equal dimension 0 of 'yt'");
        return NULL;
    }

    // Support for either 1- or 2-dimensional:
    int incGGt = 0;
    if (array_ndims[5] > 1)
    {
        if (array_dims[5][1] == 1)
        {
            incGGt = 0;
        }
        else if (
            array_dims[5][1] == int_n)
        {
            incGGt = 1;
        }
        else
        {
            PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'GGt' does not equal the number of observations of 'yt'");
            return NULL;
        }
    }

    // Support for either 2- or 3-dimensional:
    int incHHt = 0;
    if (array_ndims[8] > 2)
    {
        if (array_dims[8][2] == 1)
        {
            incHHt = 0;
        }
        else if (
            array_dims[8][2] == int_n)
        {
            incHHt = 1;
        }
        else
        {
            PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'HHt' does not equal the number of observations of 'yt'");
            return NULL;
        }
    }

#ifdef DEBUGMODE
    // Print algorithm dimensions:
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);

    // Print input dimensions:
    print_npy_intp_array(x_dims, 1, 1, "x_dims");
    print_npy_intp_array(P_dims, 1, 2, "P_dims");
    print_npy_intp_array(dt_dims, 1, 2, "dt_dims");
    print_npy_intp_array(ct_dims, 1, 2, "ct_dims");
    print_npy_intp_array(Tt_dims, 1, 3, "Tt_dims");
    print_npy_intp_array(Zt_dims, 1, 3, "Zt_dims");
    print_npy_intp_array(HHt_dims, 1, 3, "HHt_dims");
    print_npy_intp_array(GGt_dims, 1, 2, "GGt_dims");
    print_npy_intp_array(yt_dims, 1, 2, "yt_dims");

#endif

    // Fetch increment logic:

    int incdt = array_dims[3][1] == n;
    int incct = array_dims[4][1] == n;
    int incTt = array_dims[5][2] == n;
    int incZt = array_dims[6][2] == n;

#ifdef DEBUGMODE
    // Print time variant increments:
    printf("incdt: %d\n", incdt);
    printf("incct: %d\n", incct);
    printf("incTt: %d\n", incTt);
    printf("incZt: %d\n", incZt);
    printf("incHHt: %d\n", incHHt);
    printf("incGGt: %d\n", incGGt);
#endif

    // Fetch input data pointers:
    double *x = (double *)PyArray_DATA(ndarrays[0]);
    double *P = (double *)PyArray_DATA(ndarrays[1]);
    double *yt = (double *)PyArray_DATA(ndarrays[2]);
    double *dt = (double *)PyArray_DATA(ndarrays[3]);
    double *ct = (double *)PyArray_DATA(ndarrays[4]);
    double *GGt = (double *)PyArray_DATA(ndarrays[5]);
    double *Tt = (double *)PyArray_DATA(ndarrays[6]);
    double *Zt = (double *)PyArray_DATA(ndarrays[7]);
    double *HHt = (double *)PyArray_DATA(ndarrays[8]);

#ifdef DEBUGMODE
    // Print arrays:
    print_array(x, 1, int_m, "x");
    print_array(P, int_m, int_m, "P");
    print_array(dt, int_m, 1, "dt");
    print_array(ct, int_d, 1, "ct");
    print_array_3D(Tt, int_m, int_m, 1, "Tt");
    print_array_3D(Zt, int_d, int_m, 1, "Zt");
    print_array(yt, int_d, int_n, "yt");
#endif

    // Output dimensions:
    npy_intp xtt_dims[2] = {m, n};
    npy_intp Ptt_dims[3] = {m, m, n};
    npy_intp at_dims[2] = {m, n + 1};
    npy_intp Pt_dims[3] = {m, m, n + 1};
    npy_intp Ft_inv_dims[2] = {d, n};
    npy_intp vt_dims[2] = {d, n};
    npy_intp Kt_dims[3] = {m, d, n};

    // Total output sizes:
    int xtt_size = int_m * int_n * sizeof(double);
    int Ptt_size = int_m * int_m * int_n * sizeof(double);
    int at_size = int_m * (int_n + 1) * sizeof(double);
    int Pt_size = int_m * int_m * (int_n + 1) * sizeof(double);
    int Ft_inv_size = int_d * int_n * sizeof(double);
    int vt_size = int_d * int_n * sizeof(double);
    int Kt_size = int_m * int_d * int_n * sizeof(double);

    // Generate output data pointers:
    double *loglik = (double *)malloc(sizeof(double));
    double *xtt_output = (double *)malloc(xtt_size);
    double *Ptt_output = (double *)malloc(Ptt_size);
    double *at_output = (double *)malloc(at_size);
    double *Pt_output = (double *)malloc(Pt_size);
    double *Ft_inv_output = (double *)malloc(Ft_inv_size);
    double *vt_output = (double *)malloc(vt_size);
    double *Kt_output = (double *)malloc(Kt_size);

    // Call the C function
    ckalman_filter_verbose(
        // Inputs:
        int_n,
        int_m,
        int_d,
        x,
        P,
        dt, incdt,
        ct, incct,
        Tt, incTt,
        Zt, incZt,
        HHt, incHHt,
        GGt, incGGt,
        yt,
        // Outputs:
        loglik,
        xtt_output,
        Ptt_output,
        at_output,
        Pt_output,
        Ft_inv_output,
        vt_output,
        Kt_output);

    // Create NumPy arrays from the results:
    PyArrayObject *xtt = (PyArrayObject *)PyArray_SimpleNew(2, xtt_dims, NPY_DOUBLE);
    PyArrayObject *Ptt = (PyArrayObject *)PyArray_SimpleNew(3, Ptt_dims, NPY_DOUBLE);
    PyArrayObject *at = (PyArrayObject *)PyArray_SimpleNew(2, at_dims, NPY_DOUBLE);
    PyArrayObject *Pt = (PyArrayObject *)PyArray_SimpleNew(3, Pt_dims, NPY_DOUBLE);
    PyArrayObject *Ft_inv = (PyArrayObject *)PyArray_SimpleNew(2, Ft_inv_dims, NPY_DOUBLE);
    PyArrayObject *vt = (PyArrayObject *)PyArray_SimpleNew(2, vt_dims, NPY_DOUBLE);
    PyArrayObject *Kt = (PyArrayObject *)PyArray_SimpleNew(3, Kt_dims, NPY_DOUBLE);

    // Copy arrays into numpy objects:
    memcpy(PyArray_DATA(xtt), xtt_output, xtt_size);
    memcpy(PyArray_DATA(Ptt), Ptt_output, Ptt_size);
    memcpy(PyArray_DATA(at), at_output, at_size);
    memcpy(PyArray_DATA(Pt), Pt_output, Pt_size);
    memcpy(PyArray_DATA(Ft_inv), Ft_inv_output, Ft_inv_size);
    memcpy(PyArray_DATA(vt), vt_output, vt_size);
    memcpy(PyArray_DATA(Kt), Kt_output, Kt_size);

    // Create a new dictionary:
    PyObject *result_dict = PyDict_New();

    // Add arrays to the dictionary with keys
    PyDict_SetItemString(result_dict, "log_likelihood", PyFloat_FromDouble(*loglik));
    PyDict_SetItemString(result_dict, "xtt", (PyObject *)xtt);
    PyDict_SetItemString(result_dict, "Ptt", (PyObject *)Ptt);
    PyDict_SetItemString(result_dict, "at", (PyObject *)at);
    PyDict_SetItemString(result_dict, "Pt", (PyObject *)Pt);
    PyDict_SetItemString(result_dict, "Ft_inv", (PyObject *)Ft_inv);
    PyDict_SetItemString(result_dict, "vt", (PyObject *)vt);
    PyDict_SetItemString(result_dict, "Kt", (PyObject *)Kt);

    // Free dynamic memory:
    free(xtt_output);
    free(Ptt_output);
    free(at_output);
    free(Pt_output);
    free(Ft_inv_output);
    free(vt_output);
    free(Kt_output);
    free(loglik);

    return result_dict;
}

/* Wrapped ckalman_filter function */
static PyObject *kalman_smoother(PyObject *self, PyObject *args)
{

    // Valid dictionary input:
    PyObject *kwargs;
    if (!PyArg_ParseTuple(args, "O", &kwargs))
    {
        return NULL;
    }
    if (!PyDict_Check(kwargs))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    // Input dictionary keys:
    const char *keys[] = {
        "yt", "xtt", "Ptt", "Ft_inv", "Kt", "Tt", "Zt", "vt"};
    // len(keys):
    const int total_keys = 8;
    // ndarrays:
    PyArrayObject *ndarrays[8];
    // ndarray dims:
    npy_intp *array_dims[8];
    // ndarray number of dimensions:
    int array_ndims[8];

    // Extract NumPy arrays from input dictionary:
    for (int i = 0; i < total_keys; i++)
    {
        PyObject *item = PyDict_GetItemString(kwargs, keys[i]);
        if (item == NULL)
        {
            PyErr_Format(PyExc_KeyError, "Dictionary is missing '%s' key", keys[i]);
            return NULL;
        }
        if (!PyArray_Check(item))
        {
            PyErr_Format(PyExc_TypeError, "'%s' is not a valid numpy.ndarray object", keys[i]);
            return NULL;
        }
        // Fetch ndarray object:
        ndarrays[i] = (PyArrayObject *)item;
        // Fetch dimension sizes:
        array_dims[i] = PyArray_DIMS(ndarrays[i]);
        // Fetch dimension size:
        array_ndims[i] = PyArray_NDIM(ndarrays[i]);
    }

    // Number of state variables - x:
    npy_intp m = array_dims[1][0];
    int int_m = (int)m;

    // Max observations per time point (yt[0] or yt[1])
    // Allow 1- or 2-dimensional values for y:
    npy_intp n;
    npy_intp d;
    int int_n;
    int int_d;
    if (array_ndims[0] == 2)
    {
        d = array_dims[2][0];
        int_d = (int)d;
        n = array_dims[2][1];
        int_n = (int)n;
    }
    else
    {
        int_d = 1;
        d = (npy_intp)int_d;
        n = array_dims[2][0];
        int_n = (int)n;
    }

    // Ensure dimension input size consistency:
    if (array_ndims[0] > 2)
    {
        PyErr_SetString(PyExc_ValueError, "'yt' is not 1 or 2-dimensional");
        return NULL;
    }
    if (array_ndims[1] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'xt' is not 2-dimensional");
        return NULL;
    }
    if (array_ndims[2] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Pt' is not 3-dimensional");
        return NULL;
    }
    if (array_ndims[3] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'Ft_inv' is not 2-dimensional");
        return NULL;
    }
    if (array_ndims[4] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Kt' is not 3-dimensional");
        return NULL;
    }
    if (array_ndims[5] != 2 && array_ndims[5] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Tt' is not 2- or 3-dimensional");
        return NULL;
    }
    if (array_ndims[6] != 2 && array_ndims[6] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "'Zt' is not 2- or 3-dimensional");
        return NULL;
    }
    if (array_ndims[7] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "'vt' is not 2-dimensional");
        return NULL;
    }

#ifdef DEBUGMODE
    // Print algorithm dimensions:
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);

    // Print input dimensions:
    print_npy_intp_array(array_dims[0], 1, 2, "yt_dims");
    print_npy_intp_array(array_dims[1], 1, 2, "xtt_dims");
    print_npy_intp_array(array_dims[2], 1, 3, "Ptt_dims");
    print_npy_intp_array(array_dims[3], 1, 2, "Ft_inv_dims");
    print_npy_intp_array(array_dims[5], 1, 3, "Tt_dims");
    print_npy_intp_array(array_dims[6], 1, 3, "Zt_dims");
    print_npy_intp_array(array_dims[7], 1, 2, "vt_dims");

#endif

    int incTt = array_dims[5][2] == n;
    int incZt = array_dims[6][2] == n;

#ifdef DEBUGMODE
    // Print time variant increments:
    printf("incTt: %d\n", incTt);
    printf("incZt: %d\n", incZt);
#endif

    // Fetch double input data pointers:
    double *yt = (double *)PyArray_DATA(ndarrays[0]);
    double *xtt = (double *)PyArray_DATA(ndarrays[1]);
    double *Ptt = (double *)PyArray_DATA(ndarrays[2]);
    double *Ft_inv = (double *)PyArray_DATA(ndarrays[3]);
    double *Kt = (double *)PyArray_DATA(ndarrays[4]);
    double *Tt = (double *)PyArray_DATA(ndarrays[5]);
    double *Zt = (double *)PyArray_DATA(ndarrays[6]);
    double *vt = (double *)PyArray_DATA(ndarrays[7]);

#ifdef DEBUGMODE
    // Print arrays:
    print_array(xtt, int_m, int_n, "xtt");
    print_array(vt, int_m, int_n, "vt");
    print_array_3D(Ptt, int_m, int_m, int_n, "Ptt");
    print_array(Ft_inv, int_d, int_n, "Ft_inv");
    print_array_3D(Tt, int_m, int_m, 1, "Tt");
    print_array_3D(Kt, int_m, int_m, int_n, "Kt");
    print_array_3D(Zt, int_d, int_m, 1, "Zt");
    print_array(yt, int_d, int_n, "yt");
#endif

    // Call the Kalman Smoother algorithm:
    ckalman_smoother(
        /* Dimesions*/
        int_n, int_m, int_d,
        // Inputs:
        Zt, incZt,
        yt,
        vt,
        Tt, incTt,
        Kt,
        Ft_inv,
        xtt,
        Ptt);

    // Create NumPy arrays from the results:
    PyArrayObject *xhatt = (PyArrayObject *)PyArray_SimpleNew(2, array_dims[1], NPY_DOUBLE);
    PyArrayObject *Vt = (PyArrayObject *)PyArray_SimpleNew(3, array_dims[2], NPY_DOUBLE);

    // Copy arrays into numpy objects:
    memcpy(PyArray_DATA(xhatt), xtt, int_m * int_n * sizeof(double));
    memcpy(PyArray_DATA(Vt), Ptt, int_m * int_m * int_n * sizeof(double));

    // Create a new dictionary:
    PyObject *result_dict = PyDict_New();

    // Add arrays to the dictionary with keys
    PyDict_SetItemString(result_dict, "xhatt", (PyObject *)xhatt);
    PyDict_SetItemString(result_dict, "Vt", (PyObject *)Vt);

    return result_dict;
}

//  Module function definitions:
static PyMethodDef KalmanFilterMethods[] = {
    {"kalman_filter", (PyCFunction)kalman_filter, METH_VARARGS, "Perform the Kalman filter algorithm through Sequential Processing, return log-likelihood"},
    {"kalman_smoother", (PyCFunction)kalman_smoother, METH_VARARGS, "Perform the Kalman smoother algorithm through Sequential Processing, return smoothed values"},
    {"kalman_filter_verbose", (PyCFunction)kalman_filter_verbose, METH_VARARGS, "Perform the Kalman filter algorithm through Sequential Processing, return log-likelihood and filtered states"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Create PyModuleDef structure */
static struct PyModuleDef KalmanFilterStruct = {
    PyModuleDef_HEAD_INIT,
    "kalman_filter",                                                                                             // name of module
    "Kalman filtering & smoothing through compile C code. Closely coupled with the kalman_filter python module", // Documentation
    -1,
    KalmanFilterMethods,
    NULL,
    NULL,
    NULL,
    NULL};

/* Module initialization */
PyObject *PyInit_kalman_filter(void)
{

    import_array();
    if (PyErr_Occurred())
    {
        printf("Failed to Inititalise NumPy C API.");
        return NULL;
    }

    return PyModule_Create(&KalmanFilterStruct);
}
