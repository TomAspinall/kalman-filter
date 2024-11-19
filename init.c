#include <Python.h>
#include "numpy/arrayobject.h"
#include "kalman_filter_sp.h"
#include "kalman_filter_verbose.h"
#include "test.h"
#include "utils.h"

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

#ifdef DEBUGMODE
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);
#endif

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

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter_verbose(PyObject *self, PyObject *args)
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

#ifdef DEBUGMODE
    printf("Debug: n is: (%Id)\n", n);
    printf("Debug: m is: (%Id)\n", m);
    printf("Debug: d is: (%Id)\n", d);
#endif

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
    /* Construct the result: a tuple of numpy arrays */
    PyObject *kalman_filter_output = ckalman_filter_verbose(
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
        yt);

    return kalman_filter_output;
}

/* Wrapped ckalman_filter function */
static PyObject *kalman_filter_test(PyObject *self, PyObject *args)
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
        PyErr_SetString(PyExc_KeyError, "Missing 'a0' key");
        return NULL;
    }
    PyObject *input_P0 = PyDict_GetItemString(kwargs, "P0");
    if (input_P0 == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'P0' key");
        return NULL;
    }
    PyObject *input_dt = PyDict_GetItemString(kwargs, "dt");
    if (input_dt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'dt' key");
        return NULL;
    }
    PyObject *input_ct = PyDict_GetItemString(kwargs, "ct");
    if (input_ct == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Ct' key");
        return NULL;
    }
    PyObject *input_Tt = PyDict_GetItemString(kwargs, "Tt");
    if (input_Tt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Tt' key");
        return NULL;
    }
    PyObject *input_Zt = PyDict_GetItemString(kwargs, "Zt");
    if (input_Zt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'Zt' key");
        return NULL;
    }
    PyObject *input_HHt = PyDict_GetItemString(kwargs, "HHt");
    if (input_HHt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'HHt' key");
        return NULL;
    }
    PyObject *input_GGt = PyDict_GetItemString(kwargs, "GGt");
    if (input_GGt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'GGt' key");
        return NULL;
    }
    PyObject *input_yt = PyDict_GetItemString(kwargs, "yt");
    if (input_yt == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing 'yt' key");
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

#ifdef DEBUGMODE
    // Print arrays:
    print_array(a0, (int)m, 1, "a0");
    print_array(P0, (int)m, (int)m, "P0");
    print_array(dt, (int)m, 1, "dt");
    print_array(ct, (int)d, 1, "ct");
    print_array_3D(Tt, (int)m, (int)m, 1, "Tt");
    print_array_3D(Zt, (int)d, (int)m, 1, "Zt");
    print_array(yt, (int)d, (int)n, "yt");
#endif

    // Dynamically allocate memory for a double
    double *loglik = (double *)malloc(sizeof(double));

    // Call the C function
    /* Construct the result: a tuple of numpy arrays */
    double **kalman_filter_output = ckalman_filter_test(
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
        yt,
        loglik);

    // Get the two results from the output (a0 and yt)
    double *a0_result = kalman_filter_output[0];
    double *yt_result = kalman_filter_output[1];
    double *P0_result = kalman_filter_output[2];
    double *dt_result = kalman_filter_output[3];
    double *ct_result = kalman_filter_output[4];
    double *Tt_result = kalman_filter_output[5];
    double *Zt_result = kalman_filter_output[6];
    double *HHt_result = kalman_filter_output[7];

    // Create NumPy arrays from the results:
    PyArrayObject *a0_output = (PyArrayObject *)PyArray_SimpleNew(1, a0_dims, NPY_DOUBLE);
    PyArrayObject *yt_output = (PyArrayObject *)PyArray_SimpleNew(2, yt_dims, NPY_DOUBLE);
    PyArrayObject *P0_output = (PyArrayObject *)PyArray_SimpleNew(2, P0_dims, NPY_DOUBLE);
    PyArrayObject *dt_output = (PyArrayObject *)PyArray_SimpleNew(2, dt_dims, NPY_DOUBLE);
    PyArrayObject *ct_output = (PyArrayObject *)PyArray_SimpleNew(2, ct_dims, NPY_DOUBLE);
    PyArrayObject *Tt_output = (PyArrayObject *)PyArray_SimpleNew(3, Tt_dims, NPY_DOUBLE);
    PyArrayObject *Zt_output = (PyArrayObject *)PyArray_SimpleNew(3, Zt_dims, NPY_DOUBLE);
    PyArrayObject *HHt_output = (PyArrayObject *)PyArray_SimpleNew(3, HHt_dims, NPY_DOUBLE);

    // Copy arrays into numpy objects:
    memcpy(PyArray_DATA(a0_output), a0_result, m * 1 * sizeof(double));
    memcpy(PyArray_DATA(yt_output), yt_result, n * d * sizeof(double));
    memcpy(PyArray_DATA(P0_output), P0_result, m * m * sizeof(double));
    memcpy(PyArray_DATA(dt_output), dt_result, m * 1 * sizeof(double));
    memcpy(PyArray_DATA(ct_output), ct_result, d * 1 * sizeof(double));
    memcpy(PyArray_DATA(Tt_output), Tt_result, m * m * 1 * sizeof(double));
    memcpy(PyArray_DATA(Zt_output), Zt_result, d * m * 1 * sizeof(double));
    memcpy(PyArray_DATA(HHt_output), HHt_result, m * m * 1 * sizeof(double));

    // npy_intp att_dims[2] = {m, n};
    // npy_intp Ptt_dims[3] = {m, n, n};
    // npy_intp at_dims[2] = {m, n + 1};
    // npy_intp Pt_dims[3] = {m, m, n + 1};
    // npy_intp Ft_inv_dims[2] = {d, n};
    // npy_intp vt_dims[2] = {d, n};
    // npy_intp Kt_dims[3] = {m, d, n};

    // PyArrayObject *att = (PyArrayObject *)PyArray_SimpleNew(2, att_dims, NPY_DOUBLE);
    // PyArrayObject *Ptt = (PyArrayObject *)PyArray_SimpleNew(3, Ptt_dims, NPY_DOUBLE);
    // PyArrayObject *at = (PyArrayObject *)PyArray_SimpleNew(2, at_dims, NPY_DOUBLE);
    // PyArrayObject *Pt = (PyArrayObject *)PyArray_SimpleNew(3, Pt_dims, NPY_DOUBLE);
    // PyArrayObject *Ft_inv = (PyArrayObject *)PyArray_SimpleNew(2, Ft_inv_dims, NPY_DOUBLE);
    // PyArrayObject *vt = (PyArrayObject *)PyArray_SimpleNew(2, vt_dims, NPY_DOUBLE);
    // PyArrayObject *Kt = (PyArrayObject *)PyArray_SimpleNew(3, Kt_dims, NPY_DOUBLE);

    // memcpy(PyArray_DATA(att),    att_result,    m * n           * sizeof(double));
    // memcpy(PyArray_DATA(Ptt),    Ptt_result,    m * m * n       * sizeof(double));
    // memcpy(PyArray_DATA(at),     at_result,     m * (n + 1)     * sizeof(double));
    // memcpy(PyArray_DATA(Pt),     Pt_result,     m * m * (n + 1) * sizeof(double));
    // memcpy(PyArray_DATA(Ft_inv), Ft_inv_result, d * n           * sizeof(double));
    // memcpy(PyArray_DATA(vt),     vt_result,     d * n           * sizeof(double));
    // memcpy(PyArray_DATA(Kt),     Kt_result,     m * d * n       * sizeof(double));

    // Create a new dictionary:
    PyObject *result_dict = PyDict_New();

    // Add arrays to the dictionary with keys
    PyDict_SetItemString(result_dict, "a0", (PyObject *)a0_output);
    PyDict_SetItemString(result_dict, "yt", (PyObject *)yt_output);
    PyDict_SetItemString(result_dict, "P0", (PyObject *)P0_output);
    PyDict_SetItemString(result_dict, "dt", (PyObject *)dt_output);
    PyDict_SetItemString(result_dict, "ct", (PyObject *)ct_output);
    PyDict_SetItemString(result_dict, "Tt", (PyObject *)Tt_output);
    PyDict_SetItemString(result_dict, "Zt", (PyObject *)Zt_output);
    PyDict_SetItemString(result_dict, "HHt", (PyObject *)HHt_output);
    // Pointer to value:
    PyDict_SetItemString(result_dict, "log_likelihood", PyFloat_FromDouble(*loglik));

    free(a0_result);
    free(yt_result);
    free(P0_result);
    free(dt_result);
    free(ct_result);
    free(Tt_result);
    free(Zt_result);
    free(HHt_result);
    free(loglik);

    // Clean up the memory for the raw data
    free(kalman_filter_output);

    return result_dict;
}

/* Define functions in module */
static PyMethodDef KalmanFilterMethods[] = {
    {"kalman_filter", (PyCFunction)kalman_filter, METH_VARARGS, "Perform the Kalman filter algorithm through Sequential Processing, return log-likelihood"},
    {"kalman_filter_verbose", (PyCFunction)kalman_filter_verbose, METH_VARARGS, "Perform the Kalman filter algorithm through Sequential Processing, return log-likelihood and filtered states"},
    {"kalman_filter_test", (PyCFunction)kalman_filter_test, METH_VARARGS, "Perform the Kalman filter algorithm through Sequential Processing, return log-likelihood and filtered states"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Create PyModuleDef structure */
static struct PyModuleDef KalmanFilterStruct = {
    PyModuleDef_HEAD_INIT,
    "kalman_filter",    // name of module
    "Kalman filtering", // module documentation
    -1,                 // size of per-interpreter state of the module
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
