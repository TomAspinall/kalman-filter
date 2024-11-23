#include <Python.h>
#include "numpy/arrayobject.h"
#include "kalman_filter_verbose.h"
#include "kalman_filter.h"
#include "kalman_smoother.h"
#include "utils.h"

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter(PyObject *self, PyObject *args)
{

    PyObject *kwargs; // We'll accept a dictionary (kwargs)
    if (!PyArg_ParseTuple(args, "O", &kwargs))
    {
        return NULL; // Error parsing tuple (we only expect a single dictionary)
    }

    if (!PyDict_Check(kwargs))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    // if (input_validation(kwargs, kwargs))
    // {
    //     return NULL; // return error;
    // };

    // Extract the NumPy array from the dictionary
    PyObject *input_a0 = PyDict_GetItemString(kwargs, "a0");
    if (input_a0 == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'a0' key, or 'a0' is not a numpy.ndarray object.");
        return NULL;
    }
    if (!PyArray_Check(input_a0))
    {
        PyErr_SetString(PyExc_KeyError, "'a0' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_P0 = PyDict_GetItemString(kwargs, "P0");
    if (input_P0 == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'P0' key");
        return NULL;
    }
    if (!PyArray_Check(input_P0))
    {
        PyErr_SetString(PyExc_KeyError, "'P0' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_dt = PyDict_GetItemString(kwargs, "dt");
    if (input_dt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'dt' key");
        return NULL;
    }
    if (!PyArray_Check(input_dt))
    {
        PyErr_SetString(PyExc_KeyError, "'dt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_ct = PyDict_GetItemString(kwargs, "ct");
    if (input_ct == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Ct' key");
        return NULL;
    }
    if (!PyArray_Check(input_ct))
    {
        PyErr_SetString(PyExc_KeyError, "'ct' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Tt = PyDict_GetItemString(kwargs, "Tt");
    if (input_Tt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Tt' key");
        return NULL;
    }
    if (!PyArray_Check(input_Tt))
    {
        PyErr_SetString(PyExc_KeyError, "'Tt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Zt = PyDict_GetItemString(kwargs, "Zt");
    if (input_Zt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Zt' key");
        return NULL;
    }
    if (!PyArray_Check(input_Zt))
    {
        PyErr_SetString(PyExc_KeyError, "'zt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_HHt = PyDict_GetItemString(kwargs, "HHt");
    if (input_HHt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'HHt' key");
        return NULL;
    }
    if (!PyArray_Check(input_HHt))
    {
        PyErr_SetString(PyExc_KeyError, "'HHt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_GGt = PyDict_GetItemString(kwargs, "GGt");
    if (input_GGt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'GGt' key");
        return NULL;
    }
    if (!PyArray_Check(input_GGt))
    {
        PyErr_SetString(PyExc_KeyError, "'GGt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_yt = PyDict_GetItemString(kwargs, "yt");
    if (input_yt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'yt' key");
        return NULL;
    }
    if (!PyArray_Check(input_yt))
    {
        PyErr_SetString(PyExc_KeyError, "'yt' is not a valid numpy.ndarray object.");
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
    int int_d = (int)d;
    // Total observations:
    npy_intp n = yt_dims[1];
    int int_n = (int)n;
    // Number of state variables:
    npy_intp m = a0_dims[0];
    int int_m = (int)m;

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

#ifdef DEBUGMODE
    // Print algorithm dimensions:
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);

    // Print input dimensions:
    print_npy_intp_array(a0_dims, 1, 1, "a0_dims");
    print_npy_intp_array(P0_dims, 1, 2, "P0_dims");
    print_npy_intp_array(dt_dims, 1, 2, "dt_dims");
    print_npy_intp_array(ct_dims, 1, 2, "ct_dims");
    print_npy_intp_array(Tt_dims, 1, 3, "Tt_dims");
    print_npy_intp_array(Zt_dims, 1, 3, "Zt_dims");
    print_npy_intp_array(HHt_dims, 1, 3, "HHt_dims");
    print_npy_intp_array(GGt_dims, 1, 2, "GGt_dims");
    print_npy_intp_array(yt_dims, 1, 2, "yt_dims");

#endif

    // Fetch increment logic:
    int incdt = dt_dims[1] == n;
    int incct = ct_dims[1] == n;
    int incTt = Tt_dims[2] == n;
    int incZt = Zt_dims[2] == n;
    int incHHt = HHt_dims[2] == n;
    int incGGt = GGt_dims[1] == n;

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
    double *a0 = (double *)PyArray_DATA(a0_arr);
    double *P0 = (double *)PyArray_DATA(P0_arr);
    double *dt = (double *)PyArray_DATA(dt_arr);
    double *ct = (double *)PyArray_DATA(ct_arr);
    double *Tt = (double *)PyArray_DATA(Tt_arr);
    double *Zt = (double *)PyArray_DATA(Zt_arr);
    double *HHt = (double *)PyArray_DATA(HHt_arr);
    double *GGt = (double *)PyArray_DATA(GGt_arr);
    double *yt = (double *)PyArray_DATA(yt_arr);

#ifdef DEBUGMODE
    // Print arrays:
    print_array(a0, 1, int_m, "a0");
    print_array(P0, int_m, int_m, "P0");
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
        a0,
        P0,
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

    PyObject *kwargs; // We'll accept a dictionary (kwargs)
    if (!PyArg_ParseTuple(args, "O", &kwargs))
    {
        return NULL; // Error parsing tuple (we only expect a single dictionary)
    }

    if (!PyDict_Check(kwargs))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    // if (input_validation(kwargs, kwargs))
    // {
    //     return NULL; // return error;
    // };

    // Extract the NumPy array from the dictionary
    PyObject *input_a0 = PyDict_GetItemString(kwargs, "a0");
    if (input_a0 == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'a0' key, or 'a0' is not a numpy.ndarray object.");
        return NULL;
    }
    if (!PyArray_Check(input_a0))
    {
        PyErr_SetString(PyExc_KeyError, "'a0' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_P0 = PyDict_GetItemString(kwargs, "P0");
    if (input_P0 == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'P0' key");
        return NULL;
    }
    if (!PyArray_Check(input_P0))
    {
        PyErr_SetString(PyExc_KeyError, "'P0' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_dt = PyDict_GetItemString(kwargs, "dt");
    if (input_dt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'dt' key");
        return NULL;
    }
    if (!PyArray_Check(input_dt))
    {
        PyErr_SetString(PyExc_KeyError, "'dt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_ct = PyDict_GetItemString(kwargs, "ct");
    if (input_ct == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Ct' key");
        return NULL;
    }
    if (!PyArray_Check(input_ct))
    {
        PyErr_SetString(PyExc_KeyError, "'ct' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Tt = PyDict_GetItemString(kwargs, "Tt");
    if (input_Tt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Tt' key");
        return NULL;
    }
    if (!PyArray_Check(input_Tt))
    {
        PyErr_SetString(PyExc_KeyError, "'Tt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Zt = PyDict_GetItemString(kwargs, "Zt");
    if (input_Zt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Zt' key");
        return NULL;
    }
    if (!PyArray_Check(input_Zt))
    {
        PyErr_SetString(PyExc_KeyError, "'zt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_HHt = PyDict_GetItemString(kwargs, "HHt");
    if (input_HHt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'HHt' key");
        return NULL;
    }
    if (!PyArray_Check(input_HHt))
    {
        PyErr_SetString(PyExc_KeyError, "'HHt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_GGt = PyDict_GetItemString(kwargs, "GGt");
    if (input_GGt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'GGt' key");
        return NULL;
    }
    if (!PyArray_Check(input_GGt))
    {
        PyErr_SetString(PyExc_KeyError, "'GGt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_yt = PyDict_GetItemString(kwargs, "yt");
    if (input_yt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'yt' key");
        return NULL;
    }
    if (!PyArray_Check(input_yt))
    {
        PyErr_SetString(PyExc_KeyError, "'yt' is not a valid numpy.ndarray object.");
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
    int int_d = (int)d;
    // Total observations:
    npy_intp n = yt_dims[1];
    int int_n = (int)n;
    // Number of state variables:
    npy_intp m = a0_dims[0];
    int int_m = (int)m;

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

#ifdef DEBUGMODE
    // Print algorithm dimensions:
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);

    // Print input dimensions:
    print_npy_intp_array(a0_dims, 1, 1, "a0_dims");
    print_npy_intp_array(P0_dims, 1, 2, "P0_dims");
    print_npy_intp_array(dt_dims, 1, 2, "dt_dims");
    print_npy_intp_array(ct_dims, 1, 2, "ct_dims");
    print_npy_intp_array(Tt_dims, 1, 3, "Tt_dims");
    print_npy_intp_array(Zt_dims, 1, 3, "Zt_dims");
    print_npy_intp_array(HHt_dims, 1, 3, "HHt_dims");
    print_npy_intp_array(GGt_dims, 1, 2, "GGt_dims");
    print_npy_intp_array(yt_dims, 1, 2, "yt_dims");

#endif

    // Fetch increment logic:
    int incdt = dt_dims[1] == n;
    int incct = ct_dims[1] == n;
    int incTt = Tt_dims[2] == n;
    int incZt = Zt_dims[2] == n;
    int incHHt = HHt_dims[2] == n;
    int incGGt = GGt_dims[1] == n;

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
    double *a0 = (double *)PyArray_DATA(a0_arr);
    double *P0 = (double *)PyArray_DATA(P0_arr);
    double *dt = (double *)PyArray_DATA(dt_arr);
    double *ct = (double *)PyArray_DATA(ct_arr);
    double *Tt = (double *)PyArray_DATA(Tt_arr);
    double *Zt = (double *)PyArray_DATA(Zt_arr);
    double *HHt = (double *)PyArray_DATA(HHt_arr);
    double *GGt = (double *)PyArray_DATA(GGt_arr);
    double *yt = (double *)PyArray_DATA(yt_arr);

#ifdef DEBUGMODE
    // Print arrays:
    print_array(a0, 1, int_m, "a0");
    print_array(P0, int_m, int_m, "P0");
    print_array(dt, int_m, 1, "dt");
    print_array(ct, int_d, 1, "ct");
    print_array_3D(Tt, int_m, int_m, 1, "Tt");
    print_array_3D(Zt, int_d, int_m, 1, "Zt");
    print_array(yt, int_d, int_n, "yt");
#endif

    // Output dimensions:
    npy_intp att_dims[2] = {m, n};
    npy_intp Ptt_dims[3] = {m, m, n};
    npy_intp at_dims[2] = {m, n + 1};
    npy_intp Pt_dims[3] = {m, m, n + 1};
    npy_intp Ft_inv_dims[2] = {d, n};
    npy_intp vt_dims[2] = {d, n};
    npy_intp Kt_dims[3] = {m, d, n};

    // Total output sizes:
    int att_size = int_m * int_n * sizeof(double);
    int Ptt_size = int_m * int_m * int_n * sizeof(double);
    int at_size = int_m * (int_n + 1) * sizeof(double);
    int Pt_size = int_m * int_m * (int_n + 1) * sizeof(double);
    int Ft_inv_size = int_d * int_n * sizeof(double);
    int vt_size = int_d * int_n * sizeof(double);
    int Kt_size = int_m * int_d * int_n * sizeof(double);

    // Generate output data pointers:
    double *loglik = (double *)malloc(sizeof(double));
    double *att_output = (double *)malloc(att_size);
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
        a0,
        P0,
        dt, incdt,
        ct, incct,
        Tt, incTt,
        Zt, incZt,
        HHt, incHHt,
        GGt, incGGt,
        yt,
        // Outputs:
        loglik,
        att_output,
        Ptt_output,
        at_output,
        Pt_output,
        Ft_inv_output,
        vt_output,
        Kt_output);

    // Create NumPy arrays from the results:
    PyArrayObject *att = (PyArrayObject *)PyArray_SimpleNew(2, att_dims, NPY_DOUBLE);
    PyArrayObject *Ptt = (PyArrayObject *)PyArray_SimpleNew(3, Ptt_dims, NPY_DOUBLE);
    PyArrayObject *at = (PyArrayObject *)PyArray_SimpleNew(2, at_dims, NPY_DOUBLE);
    PyArrayObject *Pt = (PyArrayObject *)PyArray_SimpleNew(3, Pt_dims, NPY_DOUBLE);
    PyArrayObject *Ft_inv = (PyArrayObject *)PyArray_SimpleNew(2, Ft_inv_dims, NPY_DOUBLE);
    PyArrayObject *vt = (PyArrayObject *)PyArray_SimpleNew(2, vt_dims, NPY_DOUBLE);
    PyArrayObject *Kt = (PyArrayObject *)PyArray_SimpleNew(3, Kt_dims, NPY_DOUBLE);

    // Copy arrays into numpy objects:
    memcpy(PyArray_DATA(att), att_output, att_size);
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
    PyDict_SetItemString(result_dict, "att", (PyObject *)att);
    PyDict_SetItemString(result_dict, "Ptt", (PyObject *)Ptt);
    PyDict_SetItemString(result_dict, "at", (PyObject *)at);
    PyDict_SetItemString(result_dict, "Pt", (PyObject *)Pt);
    PyDict_SetItemString(result_dict, "Ft_inv", (PyObject *)Ft_inv);
    PyDict_SetItemString(result_dict, "vt", (PyObject *)vt);
    PyDict_SetItemString(result_dict, "Kt", (PyObject *)Kt);

    // Free dynamic memory:
    free(att_output);
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

    PyObject *kwargs; // We'll accept a dictionary (kwargs)
    if (!PyArg_ParseTuple(args, "O", &kwargs))
    {
        return NULL; // Error parsing tuple (we only expect a single dictionary)
    }

    if (!PyDict_Check(kwargs))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    // if (input_validation(kwargs, kwargs))
    // {
    //     return NULL; // return error;
    // };

    // Extract the NumPy array from the dictionary
    PyObject *input_att = PyDict_GetItemString(kwargs, "att");
    if (input_att == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'att' key.");
        return NULL;
    }
    if (!PyArray_Check(input_att))
    {
        PyErr_SetString(PyExc_KeyError, "'a0' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Ptt = PyDict_GetItemString(kwargs, "Ptt");
    if (input_Ptt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Ptt' key");
        return NULL;
    }
    if (!PyArray_Check(input_Ptt))
    {
        PyErr_SetString(PyExc_KeyError, "'Ptt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Ft_inv = PyDict_GetItemString(kwargs, "Ft_inv");
    if (input_Ft_inv == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Ft_inv' key");
        return NULL;
    }
    if (!PyArray_Check(input_Ft_inv))
    {
        PyErr_SetString(PyExc_KeyError, "'Ft_inv' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Kt = PyDict_GetItemString(kwargs, "Kt");
    if (input_Kt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Ct' key");
        return NULL;
    }
    if (!PyArray_Check(input_Kt))
    {
        PyErr_SetString(PyExc_KeyError, "'Kt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Tt = PyDict_GetItemString(kwargs, "Tt");
    if (input_Tt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Tt' key");
        return NULL;
    }
    if (!PyArray_Check(input_Tt))
    {
        PyErr_SetString(PyExc_KeyError, "'Tt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_Zt = PyDict_GetItemString(kwargs, "Zt");
    if (input_Zt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Zt' key");
        return NULL;
    }
    if (!PyArray_Check(input_Zt))
    {
        PyErr_SetString(PyExc_KeyError, "'zt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_yt = PyDict_GetItemString(kwargs, "yt");
    if (input_yt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'yt' key");
        return NULL;
    }
    if (!PyArray_Check(input_yt))
    {
        PyErr_SetString(PyExc_KeyError, "'yt' is not a valid numpy.ndarray object.");
        return NULL;
    }
    PyObject *input_vt = PyDict_GetItemString(kwargs, "vt");
    if (input_vt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'vt' key");
        return NULL;
    }
    if (!PyArray_Check(input_vt))
    {
        PyErr_SetString(PyExc_KeyError, "'vt' is not a valid numpy.ndarray object.");
        return NULL;
    }

    // Get the NumPy array data and dimensions
    PyArrayObject *att_arr = (PyArrayObject *)input_att;
    PyArrayObject *Ptt_arr = (PyArrayObject *)input_Ptt;
    PyArrayObject *Ft_inv_arr = (PyArrayObject *)input_Ft_inv;
    PyArrayObject *Kt_arr = (PyArrayObject *)input_Kt;
    PyArrayObject *Tt_arr = (PyArrayObject *)input_Tt;
    PyArrayObject *Zt_arr = (PyArrayObject *)input_Zt;
    PyArrayObject *yt_arr = (PyArrayObject *)input_yt;
    PyArrayObject *vt_arr = (PyArrayObject *)input_vt;

    // Check that array shapes are consistent:
    // if (PyArray_NDIM(a0_arr) != 1)
    // {
    //     PyErr_SetString(PyExc_ValueError, "'a0' is not 1-dimensional");
    //     return NULL;
    // }
    // if (PyArray_NDIM(Ptt_arr) != 2)
    // {
    //     PyErr_SetString(PyExc_ValueError, "'Ptt' is not 2-dimensional");
    //     return NULL;
    // }
    // if (PyArray_NDIM(yt_arr) > 2)
    // {
    //     PyErr_SetString(PyExc_ValueError, "'yt' is not 1 or 2-dimensional");
    //     return NULL;
    // }

    // Fetch array dimension sizes:
    npy_intp *att_dims = PyArray_DIMS(att_arr);
    npy_intp *Ptt_dims = PyArray_DIMS(Ptt_arr);
    npy_intp *Ft_inv_dims = PyArray_DIMS(Ft_inv_arr);
    npy_intp *Tt_dims = PyArray_DIMS(Tt_arr);
    npy_intp *Zt_dims = PyArray_DIMS(Zt_arr);
    npy_intp *vt_dims = PyArray_DIMS(vt_arr);
    npy_intp *yt_dims = PyArray_DIMS(yt_arr);

    // Max observations per time point:
    npy_intp d = yt_dims[0];
    int int_d = (int)d;
    // Total observations:
    npy_intp n = yt_dims[1];
    int int_n = (int)n;
    // Number of state variables:
    npy_intp m = att_dims[0];
    int int_m = (int)m;

    // Check for consistency in array shapes:

    // Number of state variables (m):
    // if (Ptt_dims[0] != m || Ptt_dims[1] != m)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimensions of square matrix 'Pt' do not match length of state vector 'a0'");
    //     return NULL;
    // }
    // if (Ft_inv_dims[0] != m)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of matrix 'Ft_inv' does not match length of state vector 'a0'");
    //     return NULL;
    // }
    // if (Zt_dims[1] != m)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 2 of matrix 'Zt' does not match length of state vector 'a0'");
    //     return NULL;
    // }
    // if (Tt_dims[0] != m || Tt_dims[1] != m)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimensions 1 or 2 of matrix 'Tt' does not match length of state vector 'a0'");
    //     return NULL;
    // }

    // Total observations (n):
    // if (Ft_inv_dims[1] != n && Ft_inv_dims[1] != 1)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'Ft_inv' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (vt_dims[1] != n && vt_dims[1] != 1)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 2 of ndarray 'ct' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (PyArray_NDIM(Tt_arr) > 2 && Tt_dims[2] != n && Tt_dims[2] != 1)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Tt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }
    // if (PyArray_NDIM(Zt_arr) > 2 && Zt_dims[2] != n && Zt_dims[2] != 1)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 3 of ndarray 'Zt' does not match either 1 or number of observations/columns of 'yt'");
    //     return NULL;
    // }

    // // Max observations per time point (d):
    // if (vt_dims[0] != d)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'ct' does not equal dimension 0 of 'yt'");
    //     return NULL;
    // }
    // if (Zt_dims[0] != d)
    // {
    //     PyErr_SetString(PyExc_ValueError, "dimension 1 of ndarray 'Zt' does not equal dimension 0 of 'yt'");
    //     return NULL;
    // }

#ifdef DEBUGMODE
    // Print algorithm dimensions:
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);

    // Print input dimensions:
    print_npy_intp_array(att_dims, 1, 2, "att_dims");
    print_npy_intp_array(Ptt_dims, 1, 3, "Ptt_dims");
    print_npy_intp_array(Ft_inv_dims, 1, 2, "Ft_inv_dims");
    print_npy_intp_array(Tt_dims, 1, 3, "Tt_dims");
    print_npy_intp_array(Zt_dims, 1, 3, "Zt_dims");
    print_npy_intp_array(vt_dims, 1, 2, "vt_dims");
    print_npy_intp_array(yt_dims, 1, 2, "yt_dims");

#endif

    // Fetch increment logic:
    int incTt = Tt_dims[2] == n;
    int incZt = Zt_dims[2] == n;

#ifdef DEBUGMODE
    // Print time variant increments:
    printf("incTt: %d\n", incTt);
    printf("incZt: %d\n", incZt);
#endif

    // Fetch input data pointers:
    double *att = (double *)PyArray_DATA(att_arr);
    double *Ptt = (double *)PyArray_DATA(Ptt_arr);
    double *Ft_inv = (double *)PyArray_DATA(Ft_inv_arr);
    double *Tt = (double *)PyArray_DATA(Tt_arr);
    double *Zt = (double *)PyArray_DATA(Zt_arr);
    double *vt = (double *)PyArray_DATA(vt_arr);
    double *Kt = (double *)PyArray_DATA(Kt_arr);
    double *yt = (double *)PyArray_DATA(yt_arr);

#ifdef DEBUGMODE
    // Print arrays:
    print_array(att, int_m, int_n, "att");
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
        att,
        Ptt);

    // Create NumPy arrays from the results:
    PyArrayObject *ahatt = (PyArrayObject *)PyArray_SimpleNew(2, att_dims, NPY_DOUBLE);
    PyArrayObject *Vt = (PyArrayObject *)PyArray_SimpleNew(3, Ptt_dims, NPY_DOUBLE);

    // Copy arrays into numpy objects:
    memcpy(PyArray_DATA(ahatt), att, int_m * int_n * sizeof(double));
    memcpy(PyArray_DATA(Vt), Ptt, int_m * int_m * int_n * sizeof(double));

    // Create a new dictionary:
    PyObject *result_dict = PyDict_New();

    // Add arrays to the dictionary with keys
    PyDict_SetItemString(result_dict, "ahatt", (PyObject *)ahatt);
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
    "kalman_filter",    // name of module
    "Kalman filtering", // Documentation
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
