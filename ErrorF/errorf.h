#ifndef ERRORF_H
#define ERRORF_H

/* Print format + ": " + strerror + "\n" */
/* Exit with code 1 */
void errExit(const char *format, ...);

/* Print format + ": " + strerror + "\n" */
void errMsg(const char *format, ...);

/* Print "Usage:\t" + format + "\n" */
/* Exit with code 1 */
void errUsage(const char *format, ...);
/* Same as previous except it doesn't exit */
void printUsage(const char *format, ...);

/* Print argv0 + ": " + format + "\n" */
/* Print "Try '" + argv0 + " --help' for more information.\n" */
/* Exit with code 1 */
void errCmdline(const char *argv0, const char *format, ...);

/* Print format + "\n" */
void errPut(const char *format, ...);

#endif
