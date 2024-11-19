#include <Python.h>
#include "numpy/arrayobject.h"

/* Macro to transform an index of a 2-dimensional array into an index of a vector */
#define IDX(i, j, ncols) ((i) * (ncols) + (j))
/* Macro to transform an index of a 3-dimensional array into an index of a vector */
#define IDX_3D(i, j, k, ncols, ndepth) ((i) * (ncols) * (ndepth) + (j) * (ndepth) + (k))

/* Print arrays */
/* Print a 3D array */
void print_array_3D(double *data, int i, int j, int k, const char *lab)
{
    printf("\n'%s':\n", lab);

    // Loop through the 3D matrix
    for (int icnt = 0; icnt < i; icnt++) // Loop over rows
    {
        for (int jcnt = 0; jcnt < j; jcnt++) // Loop over columns
        {
            for (int kcnt = 0; kcnt < k; kcnt++) // Loop over depth
            {
                // Print the current element in the 3D matrix
                printf("%3.6f   ", data[IDX_3D(icnt, jcnt, kcnt, j, k)]);
            }
            printf("\n"); // Print a newline after each row
        }
        printf("\n"); // Add an extra newline between different rows
    }
}

void print_array(double *data, int i, int j, const char *lab)
{
    printf("\n'%s':\n", lab);
    for (int icnt = 0; icnt < i; icnt++)
    {
        for (int jcnt = 0; jcnt < j; jcnt++)
        {
            printf("%3.6f   ", data[IDX(icnt, jcnt, j)]);
        }
        printf("\n");
    }
}

void print_npy_intp_array(npy_intp *data, int i, int j, const char *lab)
{
    printf("\n'%s':\n", lab);
    for (int icnt = 0; icnt < i; icnt++)
    {
        for (int jcnt = 0; jcnt < j; jcnt++)
        {
            printf("%Id   ", data[IDX(icnt, jcnt, j)]);
        }
        printf("\n");
    }
}

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
