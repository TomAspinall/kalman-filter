#include "numpy/arrayobject.h"

// Debug printing:
// #define DEBUGMODE

#define M_PI 3.14159265358979323846
#define IDX(i, j, ncols) ((i) * (ncols) + (j))
#define IDX_3D(i, j, k, ncols, ndepth) ((i) * (ncols) * (ndepth) + (j) * (ndepth) + (k))

/* Print arrays */
void print_array_3D(double *data, int i, int j, int k, const char *lab);
void print_array(double *data, int i, int j, const char *lab);
void print_npy_intp_array(npy_intp *data, int i, int j, const char *lab);

/* Locate NA's in observations at time t*/
void locateNA(double *vec, int *NAindices, int *positions, int len);

/* Number of NA's in observations at time t*/
int numberofNA(double *vec, int *NAindices, int *positions, int len);

/* Temporary reduced arrays when missing obverations are present */
void reduce_array(double *array_full, int dim0, int dim1,
                  double *array_reduced, int *pos, int len);