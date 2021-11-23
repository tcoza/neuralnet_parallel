#include <stdio.h>
#include <stdlib.h>
#include "../ErrorF/errorf.h"
#include "neuralnetwork.h"
#include "commands.h"

void info(int argc, char *argv[])
{
	const char *usage = "%s";

	if (argc != 1)
		{ printUsage(usage, argv[0]); return; }

	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	const char *int_format = "    %-32s%16d\n";
	const char *str_format = "    %-32s%16s\n";

	printf("\nInformation for '%s':\n\n", getLoadedNet(selectednet)->name);
	printf(int_format, "Size of input:", net->inputSize);
	printf(int_format, "Number of layers:", net->numberOfLayers);
	printf(str_format, "Activation function:", AF_TYPE_STRINGS[net->activation.type]);
	for (int i = 1; i < net->numberOfLayers; i++)
	{
		char *hidden_layer_format = "Size of hidden layer %d:";
		size_t size = snprintf(NULL, 0, hidden_layer_format, i);
		char *buff = (char *)malloc(sizeof(char) * (size + 1));
		sprintf(buff, hidden_layer_format, i);
		printf(int_format, buff, net->layers[i-1].biases->height);
		free(buff);
	}
	if (net->numberOfLayers > 0)
		printf(int_format, "Size of output:", net->layers[net->numberOfLayers-1].biases->height);
}
