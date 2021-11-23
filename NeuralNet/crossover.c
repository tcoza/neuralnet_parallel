#include <stdlib.h>
#include "../ErrorF/errorf.h"
#include "../strmat.c"
#include "commands.h"
#include "matrix.h"

void crossover_avg(int argc, char *argv[]);
void crossover_mix(int argc, char *argv[]);
static int isSameTopology(NeuralNetwork *net1, NeuralNetwork *net2);

void crossover(int argc, char *argv[])
{
	const char *usage = "%s {avg|mix} parent_indices...";

	if (argc < 3)
		{ printUsage(usage, argv[0]); return; }

	switch (strmat(argv[1], "avg", "mix", NULL))
	{
	case 1: crossover_avg(argc, argv); break;
	case 2: crossover_mix(argc, argv); break;
	default:
		errPut(INVALID_OPTION, argv[1]);
		return;
	}
}

void crossover_avg(int argc, char *argv[])
{
	const int numParents = argc-2;
	int *indices = (int *)malloc(sizeof(int) * numParents);
	for (int i = 0; i < numParents; i++)
	{
		if (sscanf(argv[i+2], "%d", &indices[i]) != 1 || indices[i] < 1 || indices[i] > llist_count(loadednets))
			{ errPut(INVALID_VALUE_FOR, argv[i+2], "parent_index"); free(indices); return; }
		if (i > 0 && !isSameTopology(getLoadedNet(indices[0])->net, getLoadedNet(indices[i])->net))
			{ errPut(TOPOLOGY_MISMATCH); free(indices); return; }
	}

	LoadedNet *loadednet = (LoadedNet *)malloc(sizeof(LoadedNet));
	loadednet->name = STRCLONE(getLoadedNet(indices[0])->name);		// Child name is the same as that of parent 1
	NeuralNetwork *child = NULL;

	for (int i = 0; i < numParents; i++)
	{
		NeuralNetwork *currentNet = getLoadedNet(indices[i])->net;
		if (i == 0) child = (loadednet->net = neuralnetwork_clone(currentNet));
		else for (int L = 0; L < child->numberOfLayers; L++)
		{
			matrix_add(child->layers[L].weights, currentNet->layers[L].weights);
			matrix_add(child->layers[L].biases, currentNet->layers[L].biases);
		}
	}
	free(indices);

	for (int L = 0; L < child->numberOfLayers; L++)			// Average
	{
		matrix_multiply_scalar(child->layers[L].weights, (double)1 / numParents);
		matrix_multiply_scalar(child->layers[L].biases, (double)1 / numParents);
	}

	llist_add(loadednets, loadednet);
	selectednet = llist_count(loadednets);
}

void crossover_mix(int argc, char *argv[])
{
	const int numParents = argc-2;
	int *indices = (int *)malloc(sizeof(int) * numParents);
	for (int i = 0; i < numParents; i++)
	{
		if (sscanf(argv[i+2], "%d", &indices[i]) != 1 || indices[i] < 1 || indices[i] > llist_count(loadednets))
			{ errPut(INVALID_VALUE_FOR, argv[i+2], "parent_index"); free(indices); return; }
		if (i > 0 && !isSameTopology(getLoadedNet(indices[0])->net, getLoadedNet(indices[i])->net))
			{ errPut(TOPOLOGY_MISMATCH); free(indices); return; }
	}

	LoadedNet *loadednet = (LoadedNet *)malloc(sizeof(LoadedNet));
	loadednet->name = STRCLONE(getLoadedNet(indices[0])->name);
	loadednet->net = neuralnetwork_clone(getLoadedNet(indices[0])->net);
	NeuralNetwork *child = loadednet->net;

	for (int L = 0; L < child->numberOfLayers; L++)
	{
		for (int i = 0; i < child->layers[L].weights->height; i++)
			for (int j = 0; j < child->layers[L].weights->width; j++)
				matrix_set(child->layers[L].weights, i, j,
				matrix_get(getLoadedNet(indices[rand() % numParents])->net->layers[L].weights, i, j));
		for (int i = 0; i < child->layers[L].biases->height; i++)
			matrix_set(child->layers[L].biases, i, 0,
			matrix_get(getLoadedNet(indices[rand() % numParents])->net->layers[L].biases, i, 0));
	}
	free(indices);

	llist_add(loadednets, loadednet);
	selectednet = llist_count(loadednets);
}

static int isSameTopology(NeuralNetwork *net1, NeuralNetwork *net2)
{
	if (net1->inputSize != net2->inputSize)
		return 0;
	if (net1->activation.type != net2->activation.type)
		return 0;
	if (net1->numberOfLayers != net2->numberOfLayers)
		return 0;
	for (int L = 0; L < net1->numberOfLayers; L++)
		if (net1->layers[L].biases->height != net2->layers[L].biases->height)
			return 0;
	return 1;
}
