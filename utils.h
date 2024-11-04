
#define M_PI 3.14159265358979323846

/* Print arrays */
void print_array(double *data, int i, int j, const char *lab);

/* Locate NA's in observations at time t*/
void locateNA(double *vec, int *NAindices, int *positions, int len);

/* Number of NA's in observations at time t*/
int numberofNA(double *vec, int *NAindices, int *positions, int len);

/* Temporary reduced arrays when missing obverations are present */
void reduce_array(double *array_full, int dim0, int dim1,
                  double *array_reduced, int *pos, int len);