#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
    int height;     // Height of rectangular array
    int width;      // Width of rectangular array
    int size;       // Size of memory occupied by one element of the matrix.
    void *array;    // Actual array
} RectangularArray;

RectangularArray *rectarr_new(int height, int width, int size);
void *rectarr_get(RectangularArray *that, int i, int j);
//void *rectarr_set(struct RectangularArray *that, int i, int j, void *data);
void rectarr_foreach(RectangularArray *that, void (*action)(void *, int, int));
RectangularArray *rectarr_clone(RectangularArray *that);
void rectarr_free(RectangularArray *that);

typedef RectangularArray Matrix;

Matrix *matrix_new(int height, int width);
double matrix_get(Matrix *that, int i, int j);
void matrix_set(Matrix *that, int i, int j, double v);
void matrix_foreach(Matrix *that, void (*action)(double, int, int));
Matrix *matrix_print(Matrix *m);
void printDouble(double v);

// The following functions modify and return m1.
Matrix *matrix_add(Matrix *m1, Matrix *m2);
Matrix *matrix_subtract(Matrix *m1, Matrix *m2);
Matrix *matrix_multiply_scalar(Matrix *m, double v);
double matrix_toScalar(Matrix *m);

// The following function returns a new Matrix.
Matrix *matrix_multiply(Matrix *m1, Matrix *m2);
Matrix *matrix_transpose(Matrix *m);
#endif
