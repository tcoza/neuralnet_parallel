#include <stdio.h>
#include "../strmat.c"
#include "../ErrorF/errorf.h"
#include "matrix.h"
#include "commands.h"

void get_inputsize(int, char **);
void get_numlayers(int, char **);
void get_layersize(int, char **);
void get_weight(int, char **);
void get_bias(int, char **);

void get(int argc, char *argv[])
{
	const char *usage = "%s {inputsize|numlayers|layersize|weight|bias} ...";

	if (argc == 1)
		{ printUsage(usage, argv[0]); return; }

	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	switch (strmat(argv[1], "inputsize", "numlayers", "layersize", "weight", "bias", NULL))
	{
	case 1: get_inputsize(argc, argv); break;
	case 2: get_numlayers(argc, argv); break;
	case 3: get_layersize(argc, argv); break;
	case 4: get_weight(argc, argv); break;
	case 5: get_bias(argc, argv); break;
	default: printUsage(usage, argv[0]); return;
	}
}

void get_inputsize(int argc, char *argv[])
{
	const char *usage = "%s %s";

	if (argc != 2)
		{ printUsage(usage, argv[0], argv[1]); return; }

	printf("%d\n", getLoadedNet(selectednet)->net->inputSize);
}

void get_numlayers(int argc, char *argv[])
{
	const char *usage = "%s %s";

	if (argc != 2)
		{ printUsage(usage, argv[0], argv[1]); return; }

	printf("%d\n", getLoadedNet(selectednet)->net->numberOfLayers);
}

void get_layersize(int argc, char *argv[])
{
	const char *usage = "%s %s layerindex";

	if (argc != 3)
		{ printUsage(usage, argv[0], argv[1]); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;
	int layerIndex;
	if (sscanf(argv[2], "%d", &layerIndex) != 1 || layerIndex < 0 || layerIndex > net->numberOfLayers)
		{ errPut(INVALID_VALUE_FOR, argv[2], "layerindex"); return; }

	if (layerIndex == 0) get_inputsize(2, argv);
	else printf("%d\n", net->layers[layerIndex-1].biases->height);
}

void get_weight(int argc, char *argv[])
{
	const char *usage = "%s %s layerindex i j";

	if (argc != 5)
		{ printUsage(usage, argv[0], argv[1]); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	int layerIndex, i, j;
	if (sscanf(argv[2], "%d", &layerIndex) != 1 || layerIndex <= 0 || layerIndex > net->numberOfLayers)
		{ errPut(INVALID_VALUE_FOR, argv[2], "layerindex"); return; }
	if (sscanf(argv[3], "%d", &i) != 1 || i < 0 || i >= net->layers[layerIndex-1].weights->height)
		{ errPut(INVALID_VALUE_FOR, argv[3], "i"); return; }
	if (sscanf(argv[4], "%d", &j) != 1 || j < 0 || j >= net->layers[layerIndex-1].weights->width)
		{ errPut(INVALID_VALUE_FOR, argv[4], "j"); return; }

	printDouble(matrix_get(net->layers[layerIndex-1].weights, i, j));
	putchar('\n');
}

void get_bias(int argc, char *argv[])
{
	const char *usage = "%s %s layerindex i";

	if (argc != 5)
		{ printUsage(usage, argv[0], argv[1]); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	int layerIndex, i;
	if (sscanf(argv[2], "%d", &layerIndex) != 1 || layerIndex <= 0 || layerIndex > net->numberOfLayers)
		{ errPut(INVALID_VALUE_FOR, argv[2], "layerindex"); return; }
	if (sscanf(argv[3], "%d", &i) != 1 || i < 0 || i >= net->layers[layerIndex-1].biases->height)
		{ errPut(INVALID_VALUE_FOR, argv[3], "i"); return; }

	printDouble(matrix_get(net->layers[layerIndex-1].biases, i, 0));
	putchar('\n');
}
