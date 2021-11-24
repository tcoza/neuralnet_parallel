#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "matrix.h"

typedef struct
{
	Matrix *weights;
	Matrix *biases;
} Layer;

#define AF_TYPE_COUNT 2
enum AF_TYPE { LOGISTIC, RELU_01 };
typedef struct
{
	enum AF_TYPE type;
	double (*f)(double);
	double (*df)(double);
	double (*f_d)(double);
	double (*df_d)(double);
	void (*initLayer)(Layer *);
} ActivationFunction;

#ifdef MAIN
char *AF_TYPE_STRINGS[] = { "LOGISTIC", "RELU.01" };
#else
extern char *AF_TYPE_STRINGS[];
#endif

typedef struct
{
	int inputSize;
	int numberOfLayers;
	ActivationFunction activation;
	Layer layers[];
} NeuralNetwork;

typedef struct
{
	Matrix *input;
	Matrix *output;
} TrainingExample;

NeuralNetwork *neuralnetwork_new(int inputSize, int numberOfLayers, int layerSizes[], enum AF_TYPE af);
NeuralNetwork *neuralnetwork_clone(NeuralNetwork *that);
Matrix *neuralnetwork_output(NeuralNetwork *that, Matrix *input);
double neuralnetwork_train(NeuralNetwork *that, TrainingExample examples[], int numberOfExamples, double step, int parallel);
int neuralnetwork_serialize(NeuralNetwork *that, char *file);
NeuralNetwork *neuralnetwork_deserialize(char *file);
void neuralnetwork_free(NeuralNetwork *that);

// To and fro device
__host__ Layer* layer_toDevice(Layer*);
__host__ Layer* layer_fromDevice(Layer*);
__host__ NeuralNetwork* neuralnetwork_toDevice(NeuralNetwork*);

__host__ void initActivationFunctions();

#endif
