#ifndef COMMANDS_H
#define COMMANDS_H

#include <sys/types.h>
#include "../LList/llist.h"
#include "neuralnetwork.h"

typedef struct
{
	int length;
	TrainingExample *array;
} TrainingSet;

enum LF_TYPE
{
	LF_TYPE_INPUTDATA,
	LF_TYPE_TRAININGSET
};

static char *LF_TYPE_STRINGS[] = { "INPUTDATA", "TRAININGSET" };
typedef struct
{
	ino_t inode;		// To check if file is already loaded
	char *filename;
	enum LF_TYPE type;
	union
	{
		Matrix *input;
		TrainingSet trainingset;
	} data;
} LoadedFile;

typedef struct
{
	char *name;
	NeuralNetwork *net;
} LoadedNet;

#define NO_NET_SELECTED					"Error: no network selected"
#define NO_FILE_SELECTED				"Error: no file selected"
#define INVALID_VALUE_FOR				"Error: %s: invalid value for %s"
#define INVALID_FILE_FORMAT				"Error: %s: invalid file format"
#define ERR_MSG							"Error: %s"
#define FILE_ALREADY_LOADED_AT			"Error: %s is already loaded at index %d"
#define INVALID_OPTION					"Error: invalid option '%s'"
#define COMMAND_NOT_FOUND				"%s: command not found"
//#define NAME_IN_USE						"Error: name '%s' already in use"		// NO MORE NAMES!
#define INPUT_SIZE_MISMATCH				"Error: input size mismatch: expected %d, received %d"
#define INCORRECT_FILE_TYPE				"Error: incorrect type for selected file: expected '%s', provided '%s'"
#define EMPTY_TRAININGSET				"Error: training set is empty!"
#define INVALID_TRAININGSET_DIMENSIONS	"Error: invalid trainingset dimensions"
#define TOPOLOGY_MISMATCH				"Error: parent neural networks are topologically dissimilar"

void get(int argc, char *argv[]);
void set(int argc, char *argv[]);
void output(int argc, char *argv[]);
void info(int argc, char *argv[]);
void new_net(int argc, char *argv[]);
void clone_net(int argc, char *argv[]);
void crossover(int argc, char *argv[]);
void mutate(int argc, char *argv[]);
void rename_net(int argc, char *argv[]);
void load(int argc, char *argv[]);
void save(int argc, char *argv[]);
void unload(int argc, char *argv[]);
void select_index(int argc, char *argv[]);
void train(int argc, char *argv[]);
void help(void);

void loadedfile_free(LoadedFile *);
void loadednet_free(LoadedNet *);
void shuffle_array(void *array, size_t size, int length, int first);

#include <string.h>
#define STRCLONE(str) strcpy((char *)malloc(sizeof(char) * (strlen(str) + 1)), str)

#ifdef MAIN
#define PREFIX
#else
#define PREFIX extern
#endif

PREFIX LinkedList loadednets;
PREFIX LinkedList loadedfiles;
PREFIX int selectednet;				// ''
PREFIX int selectedfile;			// Starts at 1. 0 means no selected file. Careful!
PREFIX int stdoutIsTTY;
PREFIX int stdinIsTTY;
PREFIX int stderrIsTTY;

#undef PREFIX

static LoadedNet *getLoadedNet(int index) { return (LoadedNet *)llist_get(loadednets, index-1); }
static LoadedFile *getLoadedFile(int index) { return (LoadedFile *)llist_get(loadedfiles, index-1); }

//static int isNameUsed(char *name)
//{
//	NAMES ARE HEREBY DECLARED IRRELEVANT AND PURELY A CONVENIENCE TO THE USER!!
//	for (int i = 1; i <= llist_count(loadednets); i++)
//		if (!strcmp(getLoadedNet(i)->name, name))
//			return i;
//	return 0;
//}

#endif

