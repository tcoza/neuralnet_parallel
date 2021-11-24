#include "neuralnetwork.h"
#include "../Random/random.h"

__host__ __device__ static double logistic_f(double x) { return 1 / (exp(-x) + 1); }
__host__ __device__ static double logistic_df(double x) { double t = logistic_f(x); return t * (1 - t); }
static void logistic_initLayer(Layer *layer)
{
	for (int i = 0; i < layer->weights->height; i++)
	{
		for (int j = 0; j < layer->weights->width; j++)
			matrix_set(layer->weights, i, j, 3 * uniform(-1,+1) / layer->weights->width);
		matrix_set(layer->biases, i, 0, gaussian(0, 2));
	}
}

__host__ __device__ static double relu01_df(double x) { return (x > 0) ? 1 : 0.01; }
__host__ __device__ static double relu01_f(double x) { return x * relu01_df(x); }
static void relu01_initLayer(Layer *layer)
{
	for (int i = 0; i < layer->weights->height; i++)
	{
		for (int j = 0; j < layer->weights->width; j++)
			matrix_set(layer->weights, i, j, 2 * random01() / layer->weights->width);
		matrix_set(layer->biases, i, 0, gaussian(0, 1));
	}
}

static ActivationFunction activationFunctions[] =
{
	{ LOGISTIC },
	{ RELU_01 }
};

__global__ void initActivationFunctionsDevice(ActivationFunction *activationFunctions);

__host__ void initActivationFunctions()
{
	activationFunctions[0].f = logistic_f;
	activationFunctions[0].df = logistic_df;
	activationFunctions[0].initLayer = logistic_initLayer;
	activationFunctions[1].f = relu01_f;
	activationFunctions[1].df = relu01_df;
	activationFunctions[1].initLayer = relu01_initLayer;
	ActivationFunction *activationFunctionsCuda;
	cudaMalloc(&activationFunctionsCuda, sizeof(activationFunctions));
	cudaMemcpy(activationFunctionsCuda, activationFunctions, sizeof(activationFunctions), cudaMemcpyHostToDevice);
	initActivationFunctionsDevice<<<1,1>>>(activationFunctionsCuda);
	cudaDeviceSynchronize();
	cudaMemcpy(activationFunctions, activationFunctionsCuda, sizeof(activationFunctions), cudaMemcpyDeviceToHost);
}

__global__ void initActivationFunctionsDevice(ActivationFunction *activationFunctions)
{
	activationFunctions[0].f_d = logistic_f;
	activationFunctions[0].df_d = logistic_df;
	activationFunctions[1].f_d = relu01_f;
	activationFunctions[1].df_d = relu01_df;
}
