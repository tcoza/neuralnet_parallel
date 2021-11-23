#include <stdio.h>
#include <stdarg.h>

// Tries to match s0 with one of the args and returns the argno (or 0)
static int strmat(char *s0, ...)
{
    va_list list;
    va_start(list, s0);

    char *si;
    int i = 1;
    while (1)
    {
        si = va_arg(list, char *);
        if (si == NULL) break;
        if (!strcmp(s0, si)) break;
        i++;
    }
    va_end(list);

    if (si == NULL) return 0;
    else return i;
}
