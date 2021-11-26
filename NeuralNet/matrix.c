#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "mallocu.c"

__host__ __device__ RectangularArray *rectarr_new(int height, int width, size_t size)
{
	RectangularArray *that = (RectangularArray *)mallocu(sizeof(RectangularArray));
	that->size = size;
	that->height = height;
	that->width = width;
	int product = height * width;
	if (product == 0)
		printf("product is zero\n");
	int totalSize = (size_t)product * size;
	if (totalSize == 0)
		printf("totalSize is zero\n");
	that->array = mallocu(totalSize);
	if (that->array == NULL)
	{
		printf("Tried allocating array: %d x %d (product: %d, totalSize: %ld)\n", height, width, product, totalSize);
	}
	return that;
}

__host__ __device__ void *rectarr_get(RectangularArray *that, int i, int j)
{
	if (i > that->height || j > that->width) return NULL;
	return (void *)((char *)(that->array) + (i * that->width + j) * that->size);
}

/* It is hereby encouraged to use rectarr_get directly.
void *rectarr_set(struct RectangularArray *that, int i, int j, void *data)
{
if (i > that->height || j > that->width) return NULL;
return memcpy(rectarr_get(that, i, j), data, that->size);
}*/

__host__ __device__ void rectarr_foreach(RectangularArray *that, void (*action)(void *, int, int))
{
	for (int i = 0; i < that->height; i++)
		for (int j = 0; j < that->width; j++)
			action(rectarr_get(that, i, j), i, j);
}

__host__ __device__ RectangularArray *rectarr_clone(RectangularArray *that)
{
	RectangularArray *clone = rectarr_new(that->height, that->width, that->size);
	memcpy(clone->array, that->array, that->height * that->width * that->size);
	return clone;
}

__host__ RectangularArray* rectarr_toDevice(RectangularArray* that)
{
	RectangularArray* thus;
	cudaMalloc(&thus, sizeof(RectangularArray));
	cudaMemcpy(thus, that, sizeof(RectangularArray), cudaMemcpyHostToDevice);
	void* thus_array;
	cudaMalloc(&thus_array, that->height * that->width * that->size);
	cudaMemcpy(thus_array, that->array, that->height * that->width * that->size, cudaMemcpyHostToDevice);
	cudaMemcpy(&thus->array, &thus_array, sizeof(void*), cudaMemcpyHostToDevice);
	return thus;
}

__host__ RectangularArray* rectarr_fromDevice(RectangularArray* thus)
{
	RectangularArray* that = (RectangularArray*)malloc(sizeof(RectangularArray));
	cudaMemcpy(that, thus, sizeof(RectangularArray), cudaMemcpyDeviceToHost);
	void* thus_array = that->array;
	that->array = malloc(that->height * that->width * that->size);
	cudaMemcpy(that->array, thus_array, that->height * that->width * that->size, cudaMemcpyDeviceToHost);
	cudaFree(thus_array);
	cudaFree(thus);
	return that;
}

__host__ __device__ void rectarr_free(RectangularArray *that)
{
	freeu(that->array);
	freeu(that);
}

__host__ __device__ Matrix *matrix_new(int height, int width)
{
	return rectarr_new(height, width, sizeof(double));
}

__host__ __device__ double matrix_get(Matrix *that, int i, int j)
{
	return *((double *)rectarr_get(that, i, j));
}

__host__ __device__ void matrix_set(Matrix *that, int i, int j, double v)
{
	*((double *)rectarr_get(that, i, j)) = v;
}

void matrix_foreach(Matrix *that, void (*action)(double, int, int))
{
	for (int i = 0; i < that->height; i++)
		for (int j = 0; j < that->width; j++)
			action(matrix_get(that, i, j), i, j);
}

extern int stdoutIsTTY;
Matrix *matrix_print(Matrix *m)
{
	if (m->height == 0)
		return m;

	if (!stdoutIsTTY)
	{
		for (int i = 0; i < m->height; i++)
			for (int j = 0; j < m->width; j++)
			{
				printDouble(matrix_get(m, i, j));
				putchar((j == m->width-1) ? '\n' : ' ');
			}
		return m;
	}

	// If stdout is a tty
	int *widths = (int *)malloc(sizeof(int) * m->width);		// Width of each column
	for (int j = 0; j < m->width; j++)
		for (int i = 0; i < m->height; i++)
		{
			int currWidth = snprintf(NULL, 0, "%lf", matrix_get(m, i, j));
			if (i == 0) widths[j] = currWidth;
			else if (currWidth > widths[j])
				widths[j] = currWidth;
		}

	char **formats = (char **)malloc(sizeof(char *) * m->width);	// Format strings for each column
	for (int j = 0; j < m->width; j++)
	{
		char *format_format = "%%%dlf ";
		int len = snprintf(NULL, 0, format_format, widths[j]);
		char *format = (char *)malloc(sizeof(char) * (len + 1));
		sprintf(format, format_format, widths[j]);
		formats[j] = format;
	}

	for (int i = 0; i < m->height; i++)
	{
		printf("[ ");
		for (int j = 0; j < m->width; j++)
			printf(formats[j], matrix_get(m, i, j));
		printf("]\n");
	}

	for (int j = 0; j < m->width; j++)
		free(formats[j]);
	free(formats);
	free(widths);
	return m;
}

void printDouble(double v)
{
	const char *fullDoubleFormat = "%.20g";
	const char *ttyDoubleFormat = "%lf";

	if (stdoutIsTTY) printf(ttyDoubleFormat, v);
	else printf(fullDoubleFormat, v);
}

__host__ __device__ Matrix *matrix_add(Matrix *m1, Matrix *m2)
{
	if (m1->height != m2->height || m1->width != m2->width)
		return NULL;
	for (int i = 0; i < m1->height; i++)
		for (int j = 0; j < m1->width; j++)
			matrix_set(m1, i, j, matrix_get(m1, i, j) + matrix_get(m2, i, j));
	return m1;
}

__host__ __device__ Matrix *matrix_subtract(Matrix *m1, Matrix *m2)
{
	if (m1->height != m2->height || m1->width != m2->width)
		return NULL;
	for (int i = 0; i < m1->height; i++)
		for (int j = 0; j < m1->width; j++)
			matrix_set(m1, i, j, matrix_get(m1, i, j) - matrix_get(m2, i, j));
	return m1;
}

__host__ __device__ Matrix *matrix_multiply_scalar(Matrix *m, double v)
{
	for (int i = 0; i < m->height; i++)
		for (int j = 0; j < m->width; j++)
			matrix_set(m, i, j, matrix_get(m, i, j) * v);
	return m;
}

double matrix_toScalar(Matrix *m)
{
	if (m->height == 1 && m->width == 1)
		return matrix_get(m, 0, 0);
	return 0;
}

__host__ __device__ Matrix *matrix_multiply(Matrix *m1, Matrix *m2)
{
	if (m1->width != m2->height)
		return NULL;
	Matrix *ret = matrix_new(m1->height, m2->width);

//	for (int i = 0; i < ret->height; i++)
//		for (int j = 0; j < ret->width; j++)
//		{
//			double sum = 0;
//			for (int x = 0; x < m1->width; x++)
//				sum += matrix_get(m1, i, x) * matrix_get(m2, x, j);
//			matrix_set(ret, i, j, sum);
//		}

//	return ret;

	memset(ret->array, 0, ret->size * ret->height * ret->width);
	double *m1_a = (double *)m1->array;
	double *m2_a = (double *)m2->array;
	double *ret_a = (double *)ret->array;

	for (int i = 0; i < ret->height; i++)
		for (int j = 0; j < ret->width; j++)
			for (int x = 0; x < m1->width; x++)
				ret_a[i*ret->width + j] += m1_a[i*m1->width + x] * m2_a[x*m2->width + j];		// Optimized?

	return ret;

}

__host__ __device__ Matrix *matrix_transpose(Matrix *m)
{
	Matrix *ret = matrix_new(m->width, m->height);
	for (int i = 0; i < ret->height; i++)
		for (int j = 0; j < ret->width; j++)
			matrix_set(ret, i, j, matrix_get(m, j, i));
	return ret;
}
