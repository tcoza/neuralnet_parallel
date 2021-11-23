#include <stdlib.h>
#include <math.h>
#include "../msgpack.h"
#include "../Random/random.h"
#include "matrix.h"
#include "neuralnetwork.h"
#include "msgpack_reader.h"
#include "activationf.c"
#include "../_ret1free2.c"

NeuralNetwork *neuralnetwork_new(int inputSize, int numberOfLayers, int layerSizes[], enum AF_TYPE af_type)
{
	NeuralNetwork *this = (NeuralNetwork *)malloc(sizeof(NeuralNetwork) + sizeof(Layer) * numberOfLayers);
	this->inputSize = inputSize;
	this->numberOfLayers = numberOfLayers;
	this->activation = activationFunctions[af_type];
	for (int i = 0; i < this->numberOfLayers; i++)
	{
		this->layers[i].weights = matrix_new(layerSizes[i], i > 0 ? layerSizes[i-1] : inputSize);
		this->layers[i].biases = matrix_new(layerSizes[i], 1);
		this->activation.initLayer(&this->layers[i]);
	}
	return this;
}

NeuralNetwork *neuralnetwork_clone(NeuralNetwork *this)
{
	NeuralNetwork *clone = (NeuralNetwork *)malloc(sizeof(NeuralNetwork) + sizeof(Layer) * this->numberOfLayers);
	clone->inputSize = this->inputSize;
	clone->numberOfLayers = this->numberOfLayers;
	clone->activation = this->activation;
	for (int i = 0; i < clone->numberOfLayers; i++)
	{
		clone->layers[i].weights = rectarr_clone(this->layers[i].weights);
		clone->layers[i].biases = rectarr_clone(this->layers[i].biases);
	}
	return clone;
}

// Returns 1 on success, 0 on failure.
int neuralnetwork_serialize(NeuralNetwork *this, char *file)
{
	FILE *fbuf = fopen(file, "wb");
	if (!fbuf) return 0;

	msgpack_packer packer;
	msgpack_packer_init(&packer, fbuf, msgpack_fbuffer_write);

	msgpack_pack_int(&packer, this->inputSize);
	msgpack_pack_int(&packer, this->activation.type);
	msgpack_pack_int(&packer, this->numberOfLayers);
	for (int L = 0; L < this->numberOfLayers; L++)
		msgpack_pack_int(&packer, this->layers[L].biases->height);

	for (int L = 0; L < this->numberOfLayers; L++)
	{
		for (int i = 0; i < this->layers[L].weights->height; i++)
			for (int j = 0; j < this->layers[L].weights->width; j++)
				msgpack_pack_double(&packer, matrix_get(this->layers[L].weights, i, j));
		for (int i = 0; i < this->layers[L].biases->height; i++)
			msgpack_pack_double(&packer, matrix_get(this->layers[L].biases, i, 0));
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

	NeuralNetwork *this = neuralnetwork_new((int)inputSize, (int)numberOfLayers, layerSizes, (enum AF_TYPE)af_type);
	free(layerSizes);

	types[0] = MSGPACK_OBJECT_FLOAT64;		// For use in loop
	for (int L = 0; L < this->numberOfLayers; L++)
	{
		for (int i = 0; i < this->layers[L].weights->height; i++)
			for (int j = 0; j < this->layers[L].weights->width; j++)
				if (msgpack_reader_read(reader, types, 1, rectarr_get(this->layers[L].weights, i, j)) != 1)
					goto read_error;

		for (int i = 0; i < this->layers[L].biases->height; i++)
			if (msgpack_reader_read(reader, types, 1, rectarr_get(this->layers[L].biases, i, 0)) != 1)
				goto read_error;

		continue;
	read_error:
		msgpack_reader_free(reader);
		neuralnetwork_free(this);
		return NULL;
	}

	if (msgpack_reader_next(reader))
		{ msgpack_reader_free(reader); neuralnetwork_free(this); return NULL; }
	msgpack_reader_free(reader);

	return this;
}

/** Behaviour:
*  run input through the layer's weights and biases with the activationfunction
*  if this is NULL, simply apply the activationfunction to input and return that same matrix.
*  if af is null, simply return the output of the layer without the activationfunction
*/
Matrix *layer_output(Layer *this, Matrix *input, ActivationFunction *af)
{
	if (this)
		input = matrix_add(matrix_multiply(this->weights, input), this->biases);
	if (af)
		for (int i = 0; i < input->height; i++)
			matrix_set(input, i, 0, af->f(matrix_get(input, i, 0)));
	return input;
}

Matrix *neuralnetwork_output(NeuralNetwork *this, Matrix *input)
{
	for (int l = 0; l < this->numberOfLayers; l++)
		input = _ret1free2f(layer_output(&this->layers[l], input, &this->activation), l == 0 ? NULL : input, (void (*)(void *))rectarr_free);
	return input;
}

static double getCost(Matrix *exOutput, Matrix *output);
static Layer *neuralnetwork_getCostGradient(NeuralNetwork *this, TrainingExample *example);
static double cost;			// Increments by the cost function after a call to neuralnetwork_getCostGradient

double neuralnetwork_train(NeuralNetwork *this, TrainingExample examples[], int numberOfExamples, double step)
{
	if (numberOfExamples == 0) return NAN;		// Nothing to do here

	if (step == 0)		// Then just calculate cost
	{
		cost = 0;
		for (int x = 0; x < numberOfExamples; x++)
		{
			Matrix *output = neuralnetwork_output(this, examples[x].input);
			cost += getCost(output, examples[x].output);
			rectarr_free(output);
		}
		return cost / numberOfExamples;
	}

	Layer *gradient = (Layer *)malloc(sizeof(Layer) * this->numberOfLayers);
	// Initialize gradient
	for (int L = 0; L < this->numberOfLayers; L++)
		gradient[L].weights = matrix_new(this->layers[L].weights->height, this->layers[L].weights->width),
		gradient[L].biases = matrix_new(this->layers[L].biases->height, 1);
	// Gradient values will be uninitialized at this point. Careful.

	cost = 0;
	// Get gradient
	for (int x = 0; x < numberOfExamples; x++)
	{
		Layer *gradientPart = neuralnetwork_getCostGradient(this, &examples[x]);

		for (int L = 0; L < this->numberOfLayers; L++)
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

	// Apply gradient to neural network
	for (int L = 0; L < this->numberOfLayers; L++)
	{
		matrix_multiply_scalar(gradient[L].weights, step / numberOfExamples);
		matrix_multiply_scalar(gradient[L].biases, step / numberOfExamples);
		matrix_subtract(this->layers[L].weights, gradient[L].weights);
		matrix_subtract(this->layers[L].biases, gradient[L].biases);
		rectarr_free(gradient[L].weights);
		rectarr_free(gradient[L].biases);
	}
	free(gradient);

	return cost / numberOfExamples;
}


static Matrix* layerMultiplier;
static void multiply_prev(void* e, int i, int j)		// Very useful
{
	Matrix* v = *((Matrix**)e);
	*((Matrix**)e) = matrix_multiply(layerMultiplier, v);
	rectarr_free(v);
}
static Matrix* createStandardBasisVector(int size, int component, double magnitude)
{
	Matrix* v = matrix_new(size, 1);
	for (int i = 0; i < size; i++)
		matrix_set(v, i, 0, i == component ? magnitude : 0);
	return v;
}

// Returns a gradient stored in an array of layers.
// CAREFUL: The weights and biases in the layers will not be matrices, but RectangularArrays of Matrices of size 1 by 1
static Layer *neuralnetwork_getCostGradient(NeuralNetwork *this, TrainingExample *example)
{
	Layer *gradient = (Layer *)malloc(sizeof(Layer) * this->numberOfLayers);

	// Moved multiply_prev out

	Matrix *input = example->input;
	for (int L = 0; L < this->numberOfLayers; L++)
	{
		Matrix *rawOutput = layer_output(&this->layers[L], input, NULL);
		layerMultiplier = rectarr_clone(this->layers[L].weights);

		// Initialize gradient at current layer.
		gradient[L].weights = rectarr_new(this->layers[L].weights->height, this->layers[L].weights->width, sizeof(Matrix *));
		gradient[L].biases = rectarr_new(this->layers[L].biases->height, 1, sizeof(Matrix *));

		// Moved createStandardBasisVector out
		
		for (int i = 0; i < gradient[L].weights->height; i++)
		{
			double df = this->activation.df(matrix_get(rawOutput, i, 0));
			for (int j = 0; j < gradient[L].weights->width; j++)
			{
				*(Matrix **)rectarr_get(gradient[L].weights, i, j) =
					createStandardBasisVector(this->layers[L].biases->height, i, matrix_get(input, j, 0) * df);

					// Use the occasion to build layerMultiplier (optimization)
				matrix_set(layerMultiplier, i, j, matrix_get(layerMultiplier, i, j) * df);
			}

			*(Matrix **)rectarr_get(gradient[L].biases, i, 0) =
				createStandardBasisVector(this->layers[L].biases->height, i, df);
		}

		// Take care of previous layers
		for (int l = 0; l < L; l++)
		{
			rectarr_foreach(gradient[l].weights, multiply_prev);
			rectarr_foreach(gradient[l].biases, multiply_prev);
		}

		rectarr_free(layerMultiplier);
		if (L > 0) rectarr_free(input);
		input = layer_output(NULL, rawOutput, &this->activation);
	}

	// Update cost
	cost += getCost(input, example->output);

	// Now apply cost function multiplier
	layerMultiplier = matrix_transpose(matrix_multiply_scalar(matrix_subtract(input, example->output), 2));
	for (int L = 0; L < this->numberOfLayers; L++)
	{
		rectarr_foreach(gradient[L].weights, multiply_prev);
		rectarr_foreach(gradient[L].biases, multiply_prev);
	}

	rectarr_free(layerMultiplier);
	if (this->numberOfLayers > 0) rectarr_free(input);

	return gradient;
}

static double getCost(Matrix *exOutput, Matrix *output)
{
	double c = 0;
	for (int i = 0; i < output->height; i++)
	{
		double x = matrix_get(output, i, 0) - matrix_get(exOutput, i, 0);
		c += x * x;		// x^2
	}
	return c;
}

void neuralnetwork_free(NeuralNetwork *this)
{
	for (int l = 0; l < this->numberOfLayers; l++)
		rectarr_free(this->layers[l].weights),
		rectarr_free(this->layers[l].biases);
	free(this);
}
