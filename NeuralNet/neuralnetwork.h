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
NeuralNetwork *neuralnetwork_clone(NeuralNetwork *this);
Matrix *neuralnetwork_output(NeuralNetwork *this, Matrix *input);
double neuralnetwork_train(NeuralNetwork *this, TrainingExample examples[], int numberOfExamples, double step);
int neuralnetwork_serialize(NeuralNetwork *this, char *file);
NeuralNetwork *neuralnetwork_deserialize(char *file);
void neuralnetwork_free(NeuralNetwork *this);

#endif
