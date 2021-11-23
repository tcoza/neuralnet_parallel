#include <stdlib.h>

static void *_ret1free2(void *, void *);
static void *_ret1free23(void *, void *, void *);
static void *_ret1free2f(void *, void *, void (*)(void *));

static void *_ret1free2(void *arg1, void *arg2)
{
  free(arg2);
  return arg1;
}

static void *_ret1free23(void *arg1, void *arg2, void *arg3)
{
  free(arg2);
  free(arg3);
  return arg1;
}

// If arg2 is NULL, f is not called.
static void *_ret1free2f(void *arg1, void *arg2, void (*f)(void *))
{
  if (arg2) f(arg2);
  return arg1;
}
