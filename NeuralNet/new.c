#include <stdio.h>
#include <stdlib.h>
#include "neuralnetwork.h"
#include "commands.h"
#include "../ErrorF/errorf.h"
#include "../_ret1free2.c"

void new_net(int argc, char **argv)
{
	char *usage = "%s name activation_function inputsize [layer1size [layer2size...]]";

	if (argc < 4)
		{ printUsage(usage, argv[0]); return; }

	// Get ActivationFunction
	enum AF_TYPE af_type;
	for (af_type = 0; af_type < AF_TYPE_COUNT; af_type++)
		if (!strcmp(AF_TYPE_STRINGS[af_type], argv[2]))
			break;
	if (af_type == AF_TYPE_COUNT)
		{ errPut(INVALID_VALUE_FOR, argv[2], "activation_function"); return; }

	int *sizes = (int *)malloc(sizeof(int) * (argc-3));
	for (int i = 3; i < argc; i++)
		if (sscanf(argv[i], "%d", &sizes[i-3]) != 1)
			{ errPut(INVALID_VALUE_FOR, argv[i], (i==3) ? "inputsize" : "layersize"); free(sizes); return; }

	NeuralNetwork *net = neuralnetwork_new(sizes[0], argc-4, &sizes[1], af_type);
	free(sizes);

	LoadedNet *loadednet = (LoadedNet *)malloc(sizeof(LoadedNet));
	loadednet->name = STRCLONE(argv[1]);
	loadednet->net = net;

	llist_add(loadednets, loadednet);
	selectednet = llist_count(loadednets);
	return;
}

void rename_net(int argc, char *argv[])
{
	const char *usage = "%s name";

	if (argc != 2)
		{ printUsage(usage, argv[0]); return; }

	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	free(getLoadedNet(selectednet)->name);
	getLoadedNet(selectednet)->name = STRCLONE(argv[1]);
}

void clone_net(int argc, char *argv[])
{
	const char *usage = "%s";

	if (argc != 1)
		{ printUsage(usage, argv[0]); return; }
	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	LoadedNet *loadednet = (LoadedNet *)malloc(sizeof(LoadedNet));
	loadednet->name = STRCLONE(getLoadedNet(selectednet)->name);
	loadednet->net = neuralnetwork_clone(getLoadedNet(selectednet)->net);

	llist_add(loadednets, loadednet);
	selectednet = llist_count(loadednets);
}
