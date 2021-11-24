#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <math.h>
#include "../ErrorF/errorf.h"
#include "../strmat.c"
#include "commands.h"

int getMaxIndex(Matrix *v);

volatile int interrupt = 0;
void handler(int signal) { interrupt = 1; }

void train(int argc, char *argv[])
{
	const char *usage = "%s batchsize step iterations -[c|t|m|e|p]";

	if (argc < 4)
	{
		printUsage(usage, argv[0]);
		return;
	}

	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }
	if (selectedfile == 0)
		{ errPut(NO_FILE_SELECTED); return; }

	int type = getLoadedFile(selectedfile)->type;
	if (type != LF_TYPE_TRAININGSET)
	{
		errPut(INCORRECT_FILE_TYPE,
				LF_TYPE_STRINGS[LF_TYPE_TRAININGSET],
				LF_TYPE_STRINGS[type]);
		return;
	}

	TrainingSet *trainingset = &getLoadedFile(selectedfile)->data.trainingset;
	if (trainingset->length == 0)
		{ errPut(EMPTY_TRAININGSET); return; }

	NeuralNetwork *net = getLoadedNet(selectednet)->net;
	if (net->inputSize != trainingset->array[0].input->height)
		{ errPut(INVALID_TRAININGSET_DIMENSIONS); return; }
	int outputSize = net->numberOfLayers == 0 ? net->inputSize : net->layers[net->numberOfLayers-1].biases->height;
	if (outputSize != trainingset->array[0].output->height)
		{ errPut(INVALID_TRAININGSET_DIMENSIONS); return; }

	const int batchsize_max = trainingset->length;

	int batchsize, iterations;
	double step;

	if (sscanf(argv[1], "%d", &batchsize) != 1 || batchsize <= 0 || batchsize > batchsize_max)
		{ errPut(INVALID_VALUE_FOR, argv[1], "batchsize"); return; }
	if (sscanf(argv[2], "%lf", &step) != 1)
		{ errPut(INVALID_VALUE_FOR, argv[2], "step"); return; }
	if (sscanf(argv[3], "%d", &iterations) != 1)
		{ errPut(INVALID_VALUE_FOR, argv[3], "iterations"); return; }

	int show_cost = 0, show_max_index = 0, show_time = 0, show_example = 0, parallel = 0;
	for (int i = 4; i < argc; i++)
		switch (strmat(argv[i], "-c", "-m", "-t", "-e", "-p", NULL))
		{
		case 1: show_cost = 1; break;
		case 2: show_max_index = 1; break;
		case 3: show_time = 1; break;
		case 4: show_example = 1; break;
		case 5: parallel = 1; break;
		default:
			errPut(INVALID_OPTION, argv[i]);
			return;
		}

	/*struct sigaction sa, old_sa;
	sa.sa_handler = handler;
	memset(&sa.sa_mask, 0, sizeof(sigset_t));
	sa.sa_flags = 0;
	sa.sa_restorer = NULL;
	sigaction(SIGINT, &sa, &old_sa);*/

	while ((iterations < 0 || iterations-- > 0) && !interrupt)
	{
		shuffle_array(trainingset->array, sizeof(TrainingExample), trainingset->length, batchsize);

		clock_t begin = clock();
		double cost = neuralnetwork_train(net, trainingset->array, batchsize, step, parallel);
		clock_t end = clock();

		if (show_cost + show_time + show_max_index + show_example > 1)
			printf("\n");

		if (show_cost) printf("Cost: %lf\n", cost);
		if (show_time) printf("Time elapsed: %lfs\n", (double)(end-begin)/CLOCKS_PER_SEC);
		if (show_max_index)
		{
			int success_count = 0;
			for (int x = 0; x < batchsize; x++)
			{
				Matrix *ex_output = trainingset->array[x].output;
				Matrix *output = neuralnetwork_output(net, trainingset->array[x].input);

				if (getMaxIndex(ex_output) == getMaxIndex(output))
					success_count++;

				rectarr_free(output);
			}
			printf("Max index success rate: %.1lf%%\n", batchsize > 0 ? (double)(success_count*100)/batchsize : NAN);
		}
		if (show_example)
		{
			Matrix *ex_output = trainingset->array[0].output;
			Matrix *output = neuralnetwork_output(net, trainingset->array[0].input);

			printf("Example: %d -> %d\n", getMaxIndex(ex_output), getMaxIndex(output));

			rectarr_free(output);
		}
	}

	//sigaction(SIGINT, &old_sa, NULL);
}

int getMaxIndex(Matrix *v)
{
	if (v->height == 0)
		return -1;

	int max = 0;
	for (int i = 0; i < v->height; i++)
		if (matrix_get(v, i, 0) > matrix_get(v, max, 0))
			max = i;

	return max;
}

// Shuffles the first elements of the specified array.
// Size is the size of one element in the array, length is the number of elements in the array.
void shuffle_array(void *array, size_t size, int length, int first)
{
	void *temp = malloc(size);
	for (int i = 0; i < first; i++)
	{
		int random_i = rand() % (length - i) + i;
		if (random_i == i)
			continue;
		memcpy(temp, (char *)array + size * i, size);
		memcpy((char *)array + size * i, (char *)array + size * random_i, size);
		memcpy((char *)array + size * random_i, temp, size);
	}
	free(temp);
}
