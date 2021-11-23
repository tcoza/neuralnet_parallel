#include <stdio.h>
#include <stdlib.h>
#include "../strmat.c"
#include "../Random/random.h"
#include "../ErrorF/errorf.h"
#include "commands.h"

static void mutate_random(double rate);
static void mutate_negate(double rate);
static void mutate_gaussian(double rate);

void mutate(int argc, char *argv[])
{
	const char *usage = "%s {random|negate|gaussian} rate";

	if (argc != 3)
		{ printUsage(usage, argv[0]); return; }

	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	double rate;
	if (sscanf(argv[2], "%lf", &rate) != 1 || rate < 0 || rate > 1)
		{ errPut(INVALID_VALUE_FOR, argv[2], "rate"); return; }

	switch (strmat(argv[1], "random", "negate", "gaussian", NULL))
	{
	case 1: mutate_random(rate); break;
	case 2: mutate_negate(rate); break;
	case 3: mutate_gaussian(rate); break;
	default:
		errPut(INVALID_OPTION, argv[1]);
		return;
	}
}

static void mutate_random(double rate)
{
	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	// Note that that is not dynamically allocated
	Layer layer;
	for (int L = 0; L < net->numberOfLayers; L++)
	{
		layer.weights = rectarr_clone(net->layers[L].weights);
		layer.biases = rectarr_clone(net->layers[L].biases);
		net->activation.initLayer(&layer);

		for (int i = 0; i < layer.weights->height; i++)
		{
			for (int j = 0; j < layer.weights->width; j++)
				if (random01() < rate)
					matrix_set(net->layers[L].weights, i, j,
					matrix_get(layer.weights, i, j));
			if (random01() < rate)
					matrix_set(net->layers[L].biases, i, 0,
					matrix_get(layer.biases, i, 0));
		}
		rectarr_free(layer.weights);
		rectarr_free(layer.biases);
	}
}

static void mutate_negate(double rate)
{
	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	for (int L = 0; L < net->numberOfLayers; L++)
		for (int i = 0; i < net->layers[L].weights->height; i++)
		{
			for (int j = 0; j < net->layers[L].weights->width; j++)
				if (random01() < rate)
					matrix_set(net->layers[L].weights, i, j,
					-matrix_get(net->layers[L].weights, i, j));
			if (random01() < rate)
					matrix_set(net->layers[L].biases, i, 0,
					-matrix_get(net->layers[L].biases, i, 0));
		}
}

static void mutate_gaussian(double rate)
{
	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	for (int L = 0; L < net->numberOfLayers; L++)
		for (int i = 0; i < net->layers[L].weights->height; i++)
		{
			for (int j = 0; j < net->layers[L].weights->width; j++)
				if (random01() < rate)
				{
					double v = matrix_get(net->layers[L].weights, i, j);
					matrix_set(net->layers[L].weights, i, j, gaussian(v, v));
				}
			if (random01() < rate)
			{
				double v = matrix_get(net->layers[L].biases, i, 0);
				matrix_set(net->layers[L].biases, i, 0, gaussian(v, v));
			}
		}
}


