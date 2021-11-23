#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <string.h>

#include "errorf.h"

#pragma warning(disable : 4996)

void errExit(const char *format, ...)
{
	int errornumber=errno;
	va_list arglist;
	va_start(arglist,format);
	vfprintf(stderr,format,arglist);
	fprintf(stderr,": %s\n",strerror(errornumber));
	va_end(arglist);
	exit(1);
}

void errMsg(const char *format, ...)
{
	int errornumber=errno;
	va_list arglist;
	va_start(arglist,format);
	vfprintf(stderr,format,arglist);
	fprintf(stderr,": %s\n",strerror(errornumber));
	va_end(arglist);
	errno=errornumber;
}

void errUsage(const char *format, ...)
{
	va_list arglist;
	va_start(arglist,format);
	fputs("Usage: ",stderr);
	vfprintf(stderr,format,arglist);
	fputc('\n',stderr);
	va_end(arglist);
	exit(1);
}

void printUsage(const char *format, ...)
{
	va_list arglist;
	va_start(arglist,format);
	fputs("Usage: ",stderr);
	vfprintf(stderr,format,arglist);
	fputc('\n',stderr);
	va_end(arglist);
}

void errCmdline(const char *argv0, const char *format, ...)
{
	va_list arglist;
	va_start(arglist,format);
	fprintf(stderr,"%s: ",argv0);
	vfprintf(stderr,format,arglist);
	fputc('\n',stderr);
	fprintf(stderr,"Try '%s --help' for more information.\n",argv0);
	va_end(arglist);
	exit(1);
}

void errPut(const char *format, ...)
{
	va_list arglist;
	va_start(arglist,format);
	vfprintf(stderr,format,arglist);
	fputc('\n',stderr);
	va_end(arglist);
}
