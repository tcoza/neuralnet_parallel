#include <stdio.h>
#include <string.h>
#include "../msgpack.h"
#include <stdarg.h>

#include "msgpack_reader.h"

// Returns NULL on failure.
msgpack_reader *msgpack_reader_new(char *filename, size_t buff_size)
{
	msgpack_reader *this = (msgpack_reader *)malloc(sizeof(msgpack_reader));

	this->file = fopen(filename, "rb");
	if (!this->file) { free(this); return NULL; }

	msgpack_unpacked_init(&this->unpacked);

	this->buffer = (char *)malloc(buff_size);

	// To trick msgpack_next into doing the work.
	this->buff_size = buff_size;
	this->len = this->buff_size;
	this->off = this->buff_size;

	this->done = 0;

	return this;
}

// REturn NULL on failure
msgpack_object *msgpack_reader_next(msgpack_reader *this)
{
	if (this->done)
		return NULL;

	size_t off_prev = this->off;
	switch (msgpack_unpack_next(&this->unpacked, this->buffer, this->len, &this->off))
	{
	case MSGPACK_UNPACK_SUCCESS:
	case MSGPACK_UNPACK_EXTRA_BYTES:
		return &this->unpacked.data;
	case MSGPACK_UNPACK_CONTINUE:
		if (off_prev == 0)
			return NULL;		// Buffer too small
		this->off = off_prev;
		memmove(this->buffer, &this->buffer[this->off], this->len - this->off);
		this->off = this->len - this->off;
		this->len = fread(&this->buffer[this->off], sizeof(char), this->buff_size - this->off, this->file);
		if (this->len == 0)
			this->done = 1;
		this->len += this->off;
		this->off = 0;

		return msgpack_reader_next(this);
	case MSGPACK_UNPACK_PARSE_ERROR:
	case MSGPACK_UNPACK_NOMEM_ERROR:
		return NULL;
	}
}

// Rrturns the number of objects successfully read from this
int msgpack_reader_read(msgpack_reader *this, msgpack_object_type types[], int length, ...)
{
	va_list list;
	va_start(list, length);

	int i;
	for (i = 0; i < length; i++)
	{
		msgpack_object *obj = msgpack_reader_next(this);
		if (!obj || obj->type != types[i])
			break;

		switch (types[i])
		{
		case MSGPACK_OBJECT_NIL:
			va_arg(list, void *);
			break;
		case MSGPACK_OBJECT_BOOLEAN:
			*va_arg(list, bool *) = obj->via.boolean;
			break;
		case MSGPACK_OBJECT_POSITIVE_INTEGER:
			*va_arg(list, uint64_t *) = obj->via.u64;
			break;
		case MSGPACK_OBJECT_NEGATIVE_INTEGER:
			*va_arg(list, int64_t *) = obj->via.i64;
			break;
		case MSGPACK_OBJECT_FLOAT32:
		case MSGPACK_OBJECT_FLOAT64:
			*va_arg(list, double *) = obj->via.f64;
			break;
		case MSGPACK_OBJECT_STR:
			*va_arg(list, msgpack_object_str *) = obj->via.str;
			break;
		case MSGPACK_OBJECT_ARRAY:
			*va_arg(list, msgpack_object_array *) = obj->via.array;
			break;
		case MSGPACK_OBJECT_MAP:
			*va_arg(list, msgpack_object_map *) = obj->via.map;
			break;
		case MSGPACK_OBJECT_BIN:
			*va_arg(list, msgpack_object_bin *) = obj->via.bin;
			break;
		case MSGPACK_OBJECT_EXT:
			*va_arg(list, msgpack_object_ext *) = obj->via.ext;
			break;
		default:
			// Shouldn't happen. Impossible!
			break;
		}
	}
	va_end(list);
	return i;
}

// Returns the type of the converted value, -1 if failure (no object or unconvertible type)
msgpack_object_type msgpack_reader_next_asdouble(msgpack_reader *this, double *ptr)
{
	msgpack_object *obj = msgpack_reader_next(this);
	if (!obj) return -1;

	switch (obj->type)
	{
		case MSGPACK_OBJECT_NIL:
			*ptr = (double)0;
			break;
		case MSGPACK_OBJECT_BOOLEAN:
			*ptr = (double)(obj->via.boolean ? 1 : 0);
			break;
		case MSGPACK_OBJECT_POSITIVE_INTEGER:
			*ptr = (double)obj->via.u64;
			break;
		case MSGPACK_OBJECT_NEGATIVE_INTEGER:
			*ptr = (double)obj->via.i64;
			break;
		case MSGPACK_OBJECT_FLOAT32:
		case MSGPACK_OBJECT_FLOAT64:
			*ptr = obj->via.f64;
			break;
		default:
			return -1;
	}
	return obj->type;
}

void msgpack_reader_free(msgpack_reader *this)
{
	fclose(this->file);
	free(this->buffer);
	msgpack_unpacked_destroy(&this->unpacked);
	free(this);
}
