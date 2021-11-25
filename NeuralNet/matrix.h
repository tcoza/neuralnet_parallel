#ifndef MATRIX_H
#define MATRIX_H

#include <cuda.h>
#include <cuda_runtime.h>

typedef struct
{
    int height;     // Height of rectangular array
    int width;      // Width of rectangular array
    size_t size;       // Size of memory occupied by one element of the matrix.
    void *array;    // Actual array
} RectangularArray;

__host__ __device__ RectangularArray *rectarr_new(int height, int width, size_t size);
__host__ __device__ void *rectarr_get(RectangularArray *that, int i, int j);
//void *rectarr_set(struct RectangularArray *that, int i, int j, void *data);
__host__ __device__ void rectarr_foreach(RectangularArray *that, void (*action)(void *, int, int));
__host__ __device__ RectangularArray *rectarr_clone(RectangularArray *that);
__host__ __device__ void rectarr_free(RectangularArray *that);

typedef RectangularArray Matrix;

__host__ __device__ Matrix *matrix_new(int height, int width);
__host__ __device__ double matrix_get(Matrix *that, int i, int j);
__host__ __device__ void matrix_set(Matrix *that, int i, int j, double v);
void matrix_foreach(Matrix *that, void (*action)(double, int, int));
Matrix *matrix_print(Matrix *m);
void printDouble(double v);

// The following functions modify and return m1.
__host__ __device__ Matrix *matrix_add(Matrix *m1, Matrix *m2);
__host__ __device__ Matrix *matrix_subtract(Matrix *m1, Matrix *m2);
__host__ __device__ Matrix *matrix_multiply_scalar(Matrix *m, double v);
double matrix_toScalar(Matrix *m);

// The following function returns a new Matrix.
__host__ __device__ Matrix *matrix_multiply(Matrix *m1, Matrix *m2);
__host__ __device__ Matrix *matrix_transpose(Matrix *m);

// Copy to and fro Device
__host__ RectangularArray* rectarr_toDevice(RectangularArray* that);
__host__ RectangularArray* rectarr_fromDevice(RectangularArray* that);
#endif
