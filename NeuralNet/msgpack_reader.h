#ifndef MSGPACK_READER_H
#define MSGPACK_READER_H

#include <stdio.h>
#include "../msgpack.h"

typedef struct
{
	FILE *file;
	msgpack_unpacked unpacked;
	char *buffer;
	size_t buff_size;
	size_t len;
	size_t off;
	int done;
} msgpack_reader;

msgpack_reader *msgpack_reader_new(char *filename, size_t buff_size);
msgpack_object *msgpack_reader_next(msgpack_reader *that);
int msgpack_reader_read(msgpack_reader *that, msgpack_object_type types[], int length, ...);
msgpack_object_type msgpack_reader_next_asdouble(msgpack_reader *that, double *ptr);
void msgpack_reader_free(msgpack_reader *that);

#endif
