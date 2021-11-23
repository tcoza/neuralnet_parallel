#include <stdio.h>
#include "../strmat.c"
#include "../ErrorF/errorf.h"
#include "matrix.h"
#include "commands.h"

void set_weight(int, char **);
void set_bias(int, char **);

void set(int argc, char *argv[])
{
	const char *usage = "%s {weight|bias} ... value";

	if (argc == 1)
		{ printUsage(usage, argv[0]); return; }

	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	switch (strmat(argv[1], "weight", "bias", NULL))
	{
	case 1: set_weight(argc, argv); break;
	case 2: set_bias(argc, argv); break;
	default: printUsage(usage, argv[0]); return;
	}
}

void set_weight(int argc, char *argv[])
{
	const char *usage = "%s %s layerindex i j value";

	if (argc != 6)
		{ printUsage(usage, argv[0], argv[1]); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	int layerIndex, i, j;
	if (sscanf(argv[2], "%d", &layerIndex) != 1 || layerIndex <= 0 || layerIndex > net->numberOfLayers)
		{ errPut(INVALID_VALUE_FOR, argv[2], "layerindex"); return; }
	if (sscanf(argv[3], "%d", &i) != 1 || i < 0 || i >= net->layers[layerIndex-1].weights->height)
		{ errPut(INVALID_VALUE_FOR, argv[3], "i"); return; }
	if (sscanf(argv[4], "%d", &j) != 1 || j < 0 || j >= net->layers[layerIndex-1].weights->width)
		{ errPut(INVALID_VALUE_FOR, argv[4], "j"); return; }

	double value;
	if (sscanf(argv[5], "%lf", &value) != 1)
		{ errPut(INVALID_VALUE_FOR, argv[5], "value"); return; }

	matrix_set(net->layers[layerIndex-1].weights, i, j, value);
}

void set_bias(int argc, char *argv[])
{
	const char *usage = "%s %s layerindex i value";

	if (argc != 5)
		{ printUsage(usage, argv[0], argv[1]); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	int layerIndex, i;
	if (sscanf(argv[2], "%d", &layerIndex) != 1 || layerIndex <= 0 || layerIndex > net->numberOfLayers)
		{ errPut(INVALID_VALUE_FOR, argv[2], "layerindex"); return; }
	if (sscanf(argv[3], "%d", &i) != 1 || i < 0 || i >= net->layers[layerIndex-1].biases->height)
		{ errPut(INVALID_VALUE_FOR, argv[3], "i"); return; }

	double value;
	if (sscanf(argv[4], "%lf", &value) != 1)
		{ errPut(INVALID_VALUE_FOR, argv[4], "value"); return; }

	matrix_set(net->layers[layerIndex-1].biases, i, 0, value);
}
