#include <stdio.h>
#include "commands.h"
#include "../ErrorF/errorf.h"

static int getindex(char *, int max);
void print_loadednets();
void print_loadedfiles();

void select_index(int argc, char *argv[])
{
	const char *usage = "%s [index|file [index]]";

	if (argc == 1)
		{ print_loadednets(); return; }
	if (argc == 2)
		if (!strcmp(argv[1], "file"))
			{ print_loadedfiles(); return; }
		else
		{
			int index = getindex(argv[1], llist_count(loadednets));
			if (index != -1)
				selectednet = index;
			return;
		}
	if (argc == 3)
		if (!strcmp(argv[1], "file"))
		{
			int index = getindex(argv[2], llist_count(loadedfiles));
			if (index != -1)
				selectedfile = index;
			return;
		}

	// argc > 3
	printUsage(usage, argv[0]);
}

void unload(int argc, char *argv[])
{
	const char *usage = "%s {index|file index}";

	if (argc == 1 || argc > 3)
		{ printUsage(usage, argv[0]); return; }

	if (argc == 2)
	{
		int index = getindex(argv[1], llist_count(loadednets));
		if (index == -1)
			return;
		loadednet_free((LoadedNet *)llist_remove(loadednets, index-1));
		if (selectednet == index)
			selectednet = 0;
		else if (selectednet > index)
			selectednet--;
	}
	if (argc == 3)
	{
		int index = getindex(argv[2], llist_count(loadedfiles));
		if (index == -1)
			return;
		loadedfile_free((LoadedFile *)llist_remove(loadedfiles, index-1));
		if (selectedfile == index)
			selectedfile = 0;
		else if (selectedfile > index)
			selectedfile--;
	}
}

void print_loadedfiles()
{
	printf(" %c %-5s    %-30s%-20s%-10s\n", ' ', "Index", "File name", "Type", "Size");
	printf("------------------------------------------------------------------------\n");

	for (int i = 1; i <= llist_count(loadedfiles); i++)
	{
		int size;
		switch (getLoadedFile(i)->type)
		{
		case LF_TYPE_INPUTDATA:
			size = getLoadedFile(i)->data.input->height;
			break;
		case LF_TYPE_TRAININGSET:
			size = getLoadedFile(i)->data.trainingset.length;
			break;
		}
		printf(" %c %5d    %-30s%-20s%10d\n", selectedfile == i ? '*' : ' ', i, getLoadedFile(i)->filename, LF_TYPE_STRINGS[getLoadedFile(i)->type], size);
	}
}

void print_loadednets()
{
	printf(" %c %-5s    %s\n", ' ', "Index", "Name");
	printf("------------------------------------------------------------------------\n");

	for (int i = 1; i <= llist_count(loadednets); i++)
		printf(" %c %5d    %s\n", selectednet == i ? '*' : ' ', i, getLoadedNet(i)->name);
}

static int getindex(char *indexstr, int max)
{
	int index;
	if (sscanf(indexstr, "%d", &index) != 1 || index < 1 || index > max)
		{ errPut(INVALID_VALUE_FOR, indexstr, "index"); return -1; }
	else return index;
}
