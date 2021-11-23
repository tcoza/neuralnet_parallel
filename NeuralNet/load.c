#include <stdio.h>
#include "msgpack_reader.h"
#include <errno.h>
#include <ErrorF/errorf.h>
#include <strmat.c>
#include <msgpack.h>
#include "commands.h"
#include <sys/stat.h>
#include <string.h>
#include <_ret1free2.c>

void load_trainingset(char *filename, ino_t inode);
void load_inputdata(char *filename, ino_t inode);
void load_network(char *filename);
int isFileLoaded(ino_t inode);
void print_loaded();

void load(int argc, char *argv[])
{
	const char *usage = "%s [netfile|{inputdata|trainingset} filename]";

	if (argc > 3)
		{ printUsage(usage, argv[0]); return; }

	if (argc == 1)
		{ load_network(NULL); return; }			// Reload selected network

	if (argc == 2)
		{ load_network(argv[1]); return; }

	// argc==3
	struct stat sb;
	if (stat(argv[2], &sb) == -1)
		{ errMsg(ERR_MSG, argv[2]); return; }
	/*int index = isFileLoaded(sb.st_ino);				// Remove check for windows
	if (index)
		{ errPut(FILE_ALREADY_LOADED_AT, argv[2], index); return; }*/

	switch (strmat(argv[1], "inputdata", "trainingset", NULL))
	{
	case 1:
		load_inputdata(argv[2], sb.st_ino);
		break;
	case 2:
		load_trainingset(argv[2], sb.st_ino);
		break;
	default:
		errPut(INVALID_OPTION, argv[1]);
		return;
	}

}

void save(int argc, char *argv[])
{
	if (argc != 1) { printUsage("%s", argv[0]); return; }
	if (selectednet == 0)
		{ errPut(NO_NET_SELECTED); return; }

	errno = 0;
	LoadedNet *loadednet = getLoadedNet(selectednet);
	if (!neuralnetwork_serialize(loadednet->net, loadednet->name))
		errMsg(ERR_MSG, loadednet->name);
}

void load_network(char *filename)
{
	int loadSelected = (filename == NULL);
	if (loadSelected)
		if (selectednet == 0) { errPut(NO_NET_SELECTED); return; }
		else filename = getLoadedNet(selectednet)->name;

	errno = 0;
	NeuralNetwork *net = neuralnetwork_deserialize(filename);
	if (net == NULL)
		if (errno) { errMsg(ERR_MSG, filename); return; }
		else { errPut(INVALID_FILE_FORMAT, filename); return; }

	if (loadSelected)
	{
		neuralnetwork_free(getLoadedNet(selectednet)->net);
		getLoadedNet(selectednet)->net = net;
	}
	else
	{
		LoadedNet *loadednet = (LoadedNet *)malloc(sizeof(LoadedNet));
		loadednet->name = STRCLONE(filename);
		loadednet->net = net;

		llist_add(loadednets, loadednet);
		selectednet = llist_count(loadednets);
	}
}

void load_inputdata(char *filename, ino_t inode)
{
	msgpack_reader *reader = msgpack_reader_new(filename, 64);
	if (reader == NULL)
		{ errMsg(ERR_MSG, filename); return; }

	uint64_t inputSize;
	msgpack_object_type type[] = { MSGPACK_OBJECT_POSITIVE_INTEGER };
	errno = 0;
	if (msgpack_reader_read(reader, type, 1, &inputSize) != 1)
	{
		if (errno) errMsg(ERR_MSG, filename);
		else errPut(INVALID_FILE_FORMAT, filename);
		msgpack_reader_free(reader);
		return;
	}

	LoadedFile *loadedfile = (LoadedFile *)malloc(sizeof(LoadedFile));
	loadedfile->type = LF_TYPE_INPUTDATA;
	loadedfile->data.input = matrix_new(inputSize, 1);

	errno = 0;
	Matrix *input = loadedfile->data.input;
	for (int i = 0; i < inputSize; i++)
		if (msgpack_reader_next_asdouble(reader, (double *)rectarr_get(input, i, 0)) == -1)
		{
			if (errno) errMsg(ERR_MSG, filename);
			else errPut(INVALID_FILE_FORMAT, filename);
			loadedfile_free(loadedfile);
			msgpack_reader_free(reader);
			return;
		}

	loadedfile->inode = inode;
	loadedfile->filename = STRCLONE(filename);
	llist_add(loadedfiles, loadedfile);
	selectedfile = llist_count(loadedfiles);

	msgpack_reader_free(reader);
	return;
}

void load_trainingset(char *filename, ino_t inode)
{
	msgpack_reader *reader = msgpack_reader_new(filename, 64);
	if (reader == NULL)
		{ errMsg(ERR_MSG, filename); return; }

	uint64_t numberOfExamples, inputSize, outputSize;
	msgpack_object_type types[] =
	{
		MSGPACK_OBJECT_POSITIVE_INTEGER,
		MSGPACK_OBJECT_POSITIVE_INTEGER,
		MSGPACK_OBJECT_POSITIVE_INTEGER
	};
	errno = 0;
	if (msgpack_reader_read(reader, types, 3, &numberOfExamples, &inputSize, &outputSize) != 3)
	{
		if (errno) errMsg(ERR_MSG, filename);
		else errPut(INVALID_FILE_FORMAT, filename);
		msgpack_reader_free(reader);
		return;
	}

	LoadedFile *loadedfile = (LoadedFile *)malloc(sizeof(LoadedFile));
	loadedfile->type = LF_TYPE_TRAININGSET;
	loadedfile->data.trainingset.length = numberOfExamples;
	loadedfile->data.trainingset.array = (TrainingExample *)malloc(sizeof(TrainingExample) * numberOfExamples);

	for (int x = 0; x < numberOfExamples; x++)
	{
		if (stdoutIsTTY)
			printf("Loading %d...\r", x+1);

		errno = 0;

		Matrix *input = loadedfile->data.trainingset.array[x].input = matrix_new(inputSize, 1);
		Matrix *output = loadedfile->data.trainingset.array[x].output = matrix_new(outputSize, 1);

		for (int i = 0; i < inputSize; i++)
			if (msgpack_reader_next_asdouble(reader, (double *)rectarr_get(input, i, 0)) == -1)
				goto read_error;

		for (int i = 0; i < outputSize; i++)
			if (msgpack_reader_next_asdouble(reader, (double*)rectarr_get(output, i, 0)) == -1)
				goto read_error;

		continue;
	read_error:
		fflush(stdout);
		if (errno) errMsg(ERR_MSG, filename);
		else errPut(INVALID_FILE_FORMAT, filename);
		for (; x >= 0; x--)
			rectarr_free(loadedfile->data.trainingset.array[x].input),
			rectarr_free(loadedfile->data.trainingset.array[x].output);
		free(loadedfile->data.trainingset.array);
		free(loadedfile);
		msgpack_reader_free(reader);
		return;
	}
	if (numberOfExamples > 0)
		printf("\n");

	loadedfile->inode = inode;
	loadedfile->filename = STRCLONE(filename);
	llist_add(loadedfiles, loadedfile);
	selectedfile = llist_count(loadedfiles);

	msgpack_reader_free(reader);
	return;
}

int isFileLoaded(ino_t inode)
{
	for (int i = 1; i <= llist_count(loadedfiles); i++)
		if (getLoadedFile(i)->inode == inode)
			return i;
	return 0;
}

void loadedfile_free(LoadedFile *that)
{
	switch (that->type)
	{
	case LF_TYPE_INPUTDATA:
		rectarr_free(that->data.input);
		break;
	case LF_TYPE_TRAININGSET:
		for (int i = 0; i < that->data.trainingset.length; i++)
			rectarr_free(that->data.trainingset.array[i].input),
			rectarr_free(that->data.trainingset.array[i].output);
		free(that->data.trainingset.array);
		break;
	}
	free(that->filename);
	free(that);
}

void loadednet_free(LoadedNet *that)
{
	free(that->name);
	neuralnetwork_free(that->net);
	free(that);
}
