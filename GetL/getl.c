#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../LList/llist.h"
#include "getl.h"

#define BUFFSIZE 64

static int fillbuff(FILE *handle, char *buffer, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		buffer[i] = fgetc(handle);
		if (buffer[i] == '\n' || feof(handle))
		{
			buffer[i] = '\0';
			return i;
		}
	}
	return i;
}

char *fgetl(FILE *handle)
{
	struct llist *list = llist_new();
	int currsize = BUFFSIZE;
	int totalsize = 0;
	while (currsize >= BUFFSIZE)
	{
		char *buffer = (char *)malloc(BUFFSIZE);
		currsize = fillbuff(handle, buffer, BUFFSIZE);
		llist_add(list, buffer);
		totalsize += currsize;
	}

	char *line = (char *)malloc(totalsize + 1);
	int lineindex = 0;
	while (llist_count(list) != 0)
	{
		char *current = llist_remove(list, 0);
		if (llist_count(list) == 0)	memcpy(&line[lineindex], current, currsize);
		else memcpy(&line[lineindex], current, BUFFSIZE);
		free(current);
		lineindex += BUFFSIZE;
	}
	line[totalsize] = '\0';
	llist_destroy(list);
	return line;
}

char *getl(void)
{
	return fgetl(stdin);
}
