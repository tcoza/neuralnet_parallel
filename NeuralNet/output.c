#include <stdio.h>
#include "../ErrorF/errorf.h"
#include "matrix.h"
#include "neuralnetwork.h"
#include "commands.h"

void output(int argc, char *argv[])
{
	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;

	if (argc > 1 || net->inputSize == 0)
	{
		if (argc-1 != net->inputSize)
		{
			errPut(INPUT_SIZE_MISMATCH, net->inputSize, argc-1);
			return;
		}

		Matrix *input = matrix_new(net->inputSize, 1);
		for (int i = 0; i < input->height; i++)
		{
			double v;
			if (sscanf(argv[i+1], "%lf", &v) != 1)
			{
				errPut(INVALID_VALUE_FOR, argv[i+1], "input");
				rectarr_free(input);
				return;
			}
			matrix_set(input, i, 0, v);
		}
		rectarr_free(matrix_print(neuralnetwork_output(net, input)));
		rectarr_free(input);
	}
	else
	{
		// Check file selected
		if (selectedfile == 0)
			{ errPut(NO_FILE_SELECTED); return; }

		// Check file type
		int type = getLoadedFile(selectedfile)->type;
		if (type != LF_TYPE_INPUTDATA)
		{
			errPut(INCORRECT_FILE_TYPE,
					LF_TYPE_STRINGS[LF_TYPE_INPUTDATA],
					LF_TYPE_STRINGS[type]);
			return;
		}

		// Check input size
		int size = getLoadedFile(selectedfile)->data.input->height;
		if (size != net->inputSize)
		{
			errPut(INPUT_SIZE_MISMATCH, net->inputSize, size);
			return;
		}

		// Print
		rectarr_free(matrix_print(neuralnetwork_output(net, getLoadedFile(selectedfile)->data.input)));
	}

}
