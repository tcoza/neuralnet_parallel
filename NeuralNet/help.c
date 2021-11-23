#include <stdio.h>

void help()
{
	const char *format = "    %-15s%s\n";
	printf("\nHelp:\n\n");
	printf(format, "clone", "clone selected neural network");
	printf(format, "crossover", "combine specified parent networks into a new child network");
	printf(format, "get", "get specified property of selected neural network");
	printf(format, "help", "prints this message");
	printf(format, "info", "prints information about the selected neural network");
	printf(format, "load", "loads specified network or file into memory");
	printf(format, "mutate", "randomly changes weights and biases in the selected neural network");
	printf(format, "new", "creates new neural network with the specified parameters");
	printf(format, "output", "prints the selected neural network's output of the specified input");
	printf(format, "print", "prints the selected neural network as a series of matrices");
	printf(format, "quit", "exits program");
	printf(format, "rename", "renames the selected neural network");
	printf(format, "save", "write the selected neural network to its file");
	printf(format, "select", "prints loaded files and networks and selects from them");
	printf(format, "set", "set weight or bias of selected neural network");
	printf(format, "train", "trains the selected neural network using the selected training set");
	printf(format, "unload", "unloads specified neural network or file from memory");
}
