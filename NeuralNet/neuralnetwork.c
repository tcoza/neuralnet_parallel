#include <stdlib.h>
#include <math.h>
#include "../msgpack.h"
#include "../Random/random.h"
#include "matrix.h"
#include "neuralnetwork.h"
#include "msgpack_reader.h"
#include "activationf.c"
#include "../_ret1free2.c"
#include <time.h>
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "mallocu.c"

NeuralNetwork *neuralnetwork_new(int inputSize, int numberOfLayers, int layerSizes[], enum AF_TYPE af_type)
{
	NeuralNetwork *that = (NeuralNetwork *)malloc(sizeof(NeuralNetwork) + sizeof(Layer) * numberOfLayers);
	that->inputSize = inputSize;
	that->numberOfLayers = numberOfLayers;
	that->activation = activationFunctions[af_type];
	for (int i = 0; i < that->numberOfLayers; i++)
	{
		that->layers[i].weights = matrix_new(layerSizes[i], i > 0 ? layerSizes[i-1] : inputSize);
		that->layers[i].biases = matrix_new(layerSizes[i], 1);
		that->activation.initLayer(&that->layers[i]);
	}
	return that;
}

NeuralNetwork *neuralnetwork_clone(NeuralNetwork *that)
{
	NeuralNetwork *clone = (NeuralNetwork *)malloc(sizeof(NeuralNetwork) + sizeof(Layer) * that->numberOfLayers);
	clone->inputSize = that->inputSize;
	clone->numberOfLayers = that->numberOfLayers;
	clone->activation = that->activation;
	for (int i = 0; i < clone->numberOfLayers; i++)
	{
		clone->layers[i].weights = rectarr_clone(that->layers[i].weights);
		clone->layers[i].biases = rectarr_clone(that->layers[i].biases);
	}
	return clone;
}

__host__ Layer* layer_toDevice(Layer* that)
{
	Layer* thus;
	cudaMalloc(&thus, sizeof(Layer));
	Matrix* weightsCuda, * biasesCuda;
	weightsCuda = rectarr_toDevice(that->weights);
	biasesCuda = rectarr_toDevice(that->biases);
	cudaMemcpy(&thus->weights, &weightsCuda, sizeof(Matrix*), cudaMemcpyHostToDevice);
	cudaMemcpy(&thus->biases, &biasesCuda, sizeof(Matrix*), cudaMemcpyHostToDevice);
	return thus;
}

__host__ Layer* layer_fromDevice(Layer* thus)
{
	Layer* that = (Layer*)malloc(sizeof(Layer));
	cudaMemcpy(that, thus, sizeof(Layer), cudaMemcpyDeviceToHost);
	that->weights = rectarr_fromDevice(that->weights);
	that->biases = rectarr_fromDevice(that->biases);
	return that;
}

__host__ NeuralNetwork* neuralnetwork_toDevice(NeuralNetwork* that)
{
	NeuralNetwork* thus;
	cudaMalloc(&thus, sizeof(NeuralNetwork) + sizeof(Layer) * that->numberOfLayers);
	cudaMemcpy(thus, that, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);
	for (int i = 0; i < that->numberOfLayers; i++)
	{
		Matrix* weightsCuda, * biasesCuda;
		weightsCuda = rectarr_toDevice(that->layers[i].weights);
		biasesCuda = rectarr_toDevice(that->layers[i].biases);
		cudaMemcpy(&thus->layers[i].weights, &weightsCuda, sizeof(Matrix*), cudaMemcpyHostToDevice);
		cudaMemcpy(&thus->layers[i].biases, &biasesCuda, sizeof(Matrix*), cudaMemcpyHostToDevice);
	}
	return thus;
}

// Returns 1 on success, 0 on failure.
int neuralnetwork_serialize(NeuralNetwork *that, char *file)
{
	FILE *fbuf = fopen(file, "wb");
	if (!fbuf) return 0;

	msgpack_packer packer;
	msgpack_packer_init(&packer, fbuf, msgpack_fbuffer_write);

	msgpack_pack_int(&packer, that->inputSize);
	msgpack_pack_int(&packer, that->activation.type);
	msgpack_pack_int(&packer, that->numberOfLayers);
	for (int L = 0; L < that->numberOfLayers; L++)
		msgpack_pack_int(&packer, that->layers[L].biases->height);

	for (int L = 0; L < that->numberOfLayers; L++)
	{
		for (int i = 0; i < that->layers[L].weights->height; i++)
			for (int j = 0; j < that->layers[L].weights->width; j++)
				msgpack_pack_double(&packer, matrix_get(that->layers[L].weights, i, j));
		for (int i = 0; i < that->layers[L].biases->height; i++)
			msgpack_pack_double(&packer, matrix_get(that->layers[L].biases, i, 0));
	}

	fclose(fbuf);
	return 1;
}

NeuralNetwork *neuralnetwork_deserialize(char *file)
{
	msgpack_reader *reader = msgpack_reader_new(file, 64);
	if (!reader) return NULL;

	msgpack_object_type types[] =
	{
		MSGPACK_OBJECT_POSITIVE_INTEGER,
		MSGPACK_OBJECT_POSITIVE_INTEGER,
		MSGPACK_OBJECT_POSITIVE_INTEGER
	};

	uint64_t inputSize, af_type, numberOfLayers;
	if (msgpack_reader_read(reader, types, 3, &inputSize, &af_type, &numberOfLayers) != 3)
		{ msgpack_reader_free(reader); return NULL; }
	if (af_type < 0 || af_type >= AF_TYPE_COUNT)
		{ msgpack_reader_free(reader); return NULL; }

	int *layerSizes = (int *)malloc(sizeof(int) * numberOfLayers);
	for (int L = 0; L < numberOfLayers; L++)
	{
		uint64_t temp;
		if (msgpack_reader_read(reader, types, 1, &temp) != 1)			// Can reuse types array
			{ msgpack_reader_free(reader); free(layerSizes); return NULL; }
		layerSizes[L] = (int)temp;
	}

	NeuralNetwork *that = neuralnetwork_new((int)inputSize, (int)numberOfLayers, layerSizes, (enum AF_TYPE)af_type);
	free(layerSizes);

	types[0] = MSGPACK_OBJECT_FLOAT64;		// For use in loop
	for (int L = 0; L < that->numberOfLayers; L++)
	{
		for (int i = 0; i < that->layers[L].weights->height; i++)
			for (int j = 0; j < that->layers[L].weights->width; j++)
				if (msgpack_reader_read(reader, types, 1, rectarr_get(that->layers[L].weights, i, j)) != 1)
					goto read_error;

		for (int i = 0; i < that->layers[L].biases->height; i++)
			if (msgpack_reader_read(reader, types, 1, rectarr_get(that->layers[L].biases, i, 0)) != 1)
				goto read_error;

		continue;
	read_error:
		msgpack_reader_free(reader);
		neuralnetwork_free(that);
		return NULL;
	}

	if (msgpack_reader_next(reader))
		{ msgpack_reader_free(reader); neuralnetwork_free(that); return NULL; }
	msgpack_reader_free(reader);

	return that;
}

/** Behaviour:
*  run input through the layer's weights and biases with the activationfunction
*  if that is NULL, simply apply the activationfunction to input and return that same matrix.
*  if af is null, simply return the output of the layer without the activationfunction
*/
__host__ __device__ Matrix *layer_output(Layer *that, Matrix *input, ActivationFunction *af)
{
#ifndef __CUDA_ARCH__
#define F(x) af->f(x)
#else
#define F(x) af->f_d(x)
#endif // !__CUDA_ARCH__
	if (that)
		input = matrix_add(matrix_multiply(that->weights, input), that->biases);
	if (af)
		for (int i = 0; i < input->height; i++)
			matrix_set(input, i, 0, f(matrix_get(input, i, 0)));
	return input;
}

Matrix *neuralnetwork_output(NeuralNetwork *that, Matrix *input)
{
	for (int l = 0; l < that->numberOfLayers; l++)
		input = (Matrix *)_ret1free2f(layer_output(&that->layers[l], input, &that->activation), l == 0 ? NULL : input, (void (*)(void *))rectarr_free);
	return input;
}

__host__ __device__ static double getCost(Matrix *exOutput, Matrix *output);
__host__ __device__ static Layer *neuralnetwork_getCostGradient(NeuralNetwork *that, TrainingExample *example);
__global__ void neuralnetwork_getCostGradient_parallel(NeuralNetwork* that, TrainingExample* examples, Layer** gradientParts, int numberOfExamples);
static double costH;			// Increments by the cost function after a call to neuralnetwork_getCostGradient
static __device__ double costD;			// Increments by the cost function after a call to neuralnetwork_getCostGradient

double neuralnetwork_train(NeuralNetwork *that, TrainingExample examples[], int numberOfExamples, double step)
{
	if (numberOfExamples == 0) return NAN;		// Nothing to do here

	if (step == 0)		// Then just calculate cost
	{
		double cost = 0;
		for (int x = 0; x < numberOfExamples; x++)
		{
			Matrix *output = neuralnetwork_output(that, examples[x].input);
			cost += getCost(output, examples[x].output);
			rectarr_free(output);
		}
		return cost / numberOfExamples;
	}

	Layer *gradient = (Layer *)malloc(sizeof(Layer) * that->numberOfLayers);
	// Initialize gradient
	for (int L = 0; L < that->numberOfLayers; L++)
		gradient[L].weights = matrix_new(that->layers[L].weights->height, that->layers[L].weights->width),
		gradient[L].biases = matrix_new(that->layers[L].biases->height, 1);
	// Gradient values will be uninitialized at that point. Careful.

	costH = 0;


	Layer** gradientParts = (Layer**)malloc(numberOfExamples * sizeof(Layer*));
	// Get gradient

	if (false)
	{
		for (int x = 0; x < numberOfExamples; x++)
			gradientParts[x] = neuralnetwork_getCostGradient(that, &examples[x]);
	}
	else
	{
		NeuralNetwork* thatCuda = neuralnetwork_toDevice(that);
		TrainingExample* examplesCuda;
		Layer** gradientPartsCuda;
		// That to device
		cudaMalloc(&examplesCuda, numberOfExamples * sizeof(TrainingExample));
		cudaMalloc(&gradientPartsCuda, numberOfExamples * sizeof(Layer*));
		for (int x = 0; x < numberOfExamples; x++)
		{
			Matrix* inputCuda = rectarr_toDevice(examples[x].input);
			Matrix* outputCuda = rectarr_toDevice(examples[x].output);
			cudaMemcpy(&examplesCuda[x].input, &inputCuda, sizeof(Matrix*), cudaMemcpyHostToDevice);
			cudaMemcpy(&examplesCuda[x].output, &outputCuda, sizeof(Matrix*), cudaMemcpyHostToDevice);
		}

#define BLOCKDIM_MAX 1024
		neuralnetwork_getCostGradient_parallel<<<(numberOfExamples-1)/BLOCK_DIM_MAX+1,BLOCKDIM_MAX>>>
				(thatCuda, examplesCuda, gradientPartsCuda, numberOfExamples);
		cudaDeviceSynchronize();

		cudaMemcpy(gradientParts, gradientPartsCuda, numberOfExamples * sizeof(Layer*), cudaMemcpyDeviceToHost);
		for (int x = 0; x < numberOfExamples; x++)
			gradientParts[x] = layer_fromDevice(gradientParts[x]);

		return 0;
	}

	for (int x = 0; x < numberOfExamples; x++)
	{
		Layer* gradientPart = gradientParts[x];
		for (int L = 0; L < that->numberOfLayers; L++)
		{
			for (int i = 0; i < gradient[L].weights->height; i++)
			{
				// Weights
				for (int j = 0; j < gradient[L].weights->width; j++)
					matrix_set(gradient[L].weights, i, j,
							matrix_toScalar(*(Matrix **)rectarr_get(gradientPart[L].weights, i, j))
							+ ((x == 0) ? 0 : matrix_get(gradient[L].weights, i, j))),
					rectarr_free(*(Matrix **)rectarr_get(gradientPart[L].weights, i, j));

				// Biases
				matrix_set(gradient[L].biases, i, 0,
						matrix_toScalar(*(Matrix **)rectarr_get(gradientPart[L].biases, i, 0))
						+ ((x == 0) ? 0 : matrix_get(gradient[L].biases, i, 0)));
				rectarr_free(*(Matrix **)rectarr_get(gradientPart[L].biases, i, 0));
			}
			rectarr_free(gradientPart[L].weights);
			rectarr_free(gradientPart[L].biases);
		}
		free(gradientPart);
	}
	free(gradientParts);

	// Apply gradient to neural network
	for (int L = 0; L < that->numberOfLayers; L++)
	{
		matrix_multiply_scalar(gradient[L].weights, step / numberOfExamples);
		matrix_multiply_scalar(gradient[L].biases, step / numberOfExamples);
		matrix_subtract(that->layers[L].weights, gradient[L].weights);
		matrix_subtract(that->layers[L].biases, gradient[L].biases);
		rectarr_free(gradient[L].weights);
		rectarr_free(gradient[L].biases);
	}
	free(gradient);

	return costH / numberOfExamples;
}

__global__ void neuralnetwork_getCostGradient_parallel(NeuralNetwork* that, TrainingExample* examples, Layer** gradientParts, int numberOfExamples)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= numberOfExamples) return;
	printf("Here\n");
	gradientParts[x] = neuralnetwork_getCostGradient(that, &examples[x]);
	printf("There\n");
}

static Matrix* layerMultiplierH;
__device__ static Matrix* layerMultiplierD;
__host__ __device__ static void multiply_prev(void* e, int i, int j)		// Very useful
{
#ifndef __CUDA_ARCH__
#define layerMultiplier layerMultiplierH
#else
#define layerMultiplier layerMultiplierD
#endif // !__CUDA_ARCH__
	Matrix* v = *((Matrix**)e);
	*((Matrix**)e) = matrix_multiply(layerMultiplier, v);
	rectarr_free(v);
}
__host__ __device__ static Matrix* createStandardBasisVector(int size, int component, double magnitude)
{
	Matrix* v = matrix_new(size, 1);
	for (int i = 0; i < size; i++)
		matrix_set(v, i, 0, i == component ? magnitude : 0);
	return v;
}

// Returns a gradient stored in an array of layers.
// CAREFUL: The weights and biases in the layers will not be matrices, but RectangularArrays of Matrices of size 1 by 1
static __host__ __device__ Layer *neuralnetwork_getCostGradient(NeuralNetwork *that, TrainingExample *example)
{
#ifndef __CUDA_ARCH__
#define layerMultiplier layerMultiplierH
#define cost costH
#define DF(x) (that->activation.df(x))
#else
#define layerMultiplier layerMultiplierD
#define cost costD
#define DF(x) (that->activation.df_d(x))
#endif // !__CUDA_ARCH__

	Layer *gradient = (Layer *)mallocu(sizeof(Layer) * that->numberOfLayers);

	// Moved multiply_prev out

	Matrix *input = example->input;
	for (int L = 0; L < that->numberOfLayers; L++)
	{
		Matrix *rawOutput = layer_output(&that->layers[L], input, NULL);
		layerMultiplier = rectarr_clone(that->layers[L].weights);

		// Initialize gradient at current layer.
		gradient[L].weights = rectarr_new(that->layers[L].weights->height, that->layers[L].weights->width, sizeof(Matrix *));
		gradient[L].biases = rectarr_new(that->layers[L].biases->height, 1, sizeof(Matrix *));

		// Moved createStandardBasisVector out
		for (int i = 0; i < gradient[L].weights->height; i++)
		{
			double df = DF(matrix_get(rawOutput, i, 0));
			for (int j = 0; j < gradient[L].weights->width; j++)
			{
				*(Matrix **)rectarr_get(gradient[L].weights, i, j) =
					createStandardBasisVector(that->layers[L].biases->height, i, matrix_get(input, j, 0) * df);

					// Use the occasion to build layerMultiplier (optimization)
				matrix_set(layerMultiplier, i, j, matrix_get(layerMultiplier, i, j) * df);
			}

			*(Matrix **)rectarr_get(gradient[L].biases, i, 0) =
				createStandardBasisVector(that->layers[L].biases->height, i, df);
		}

		// Take care of previous layers
		for (int l = 0; l < L; l++)
		{
			rectarr_foreach(gradient[l].weights, multiply_prev);
			rectarr_foreach(gradient[l].biases, multiply_prev);
		}

		rectarr_free(layerMultiplier);
		if (L > 0) rectarr_free(input);
		input = layer_output(NULL, rawOutput, &that->activation);
	}

	// Update cost
	cost += getCost(input, example->output);

	// Now apply cost function multiplier
	layerMultiplier = matrix_transpose(matrix_multiply_scalar(matrix_subtract(input, example->output), 2));
	for (int L = 0; L < that->numberOfLayers; L++)
	{
		rectarr_foreach(gradient[L].weights, multiply_prev);
		rectarr_foreach(gradient[L].biases, multiply_prev);
	}

	rectarr_free(layerMultiplier);
	if (that->numberOfLayers > 0) rectarr_free(input);

	return gradient;
}

__host__ __device__ static double getCost(Matrix *exOutput, Matrix *output)
{
	double c = 0;
	for (int i = 0; i < output->height; i++)
	{
		double x = matrix_get(output, i, 0) - matrix_get(exOutput, i, 0);
		c += x * x;		// x^2
	}
	return c;
}

void neuralnetwork_free(NeuralNetwork *that)
{
	for (int l = 0; l < that->numberOfLayers; l++)
		rectarr_free(that->layers[l].weights),
		rectarr_free(that->layers[l].biases);
	free(that);
}
