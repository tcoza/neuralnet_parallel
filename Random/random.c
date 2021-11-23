#include <stdlib.h>
#include <math.h>
#include "random.h"

double random01()
{
	int rnd;
	//do rnd = random();
	do rnd = rand();
	while (rnd == RAND_MAX);
	return (double)rnd/RAND_MAX;
}

double uniform(double min, double max)
{
	return ((double)rand()/RAND_MAX)*(max-min)+min;
}

/* "Polar method" */
/* Code from https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/ */
double gaussian(double mu, double sigma)
{
	double u1, u2, w, mult;
	static double x1, x2;
	static int call = 0;

	if (call)
	{
		call = !call;
		return mu + sigma * (double)x2;
	}

	do
	{
		u1 = uniform(-1, +1);
		u2 = uniform(-1, +1);
		w = u1*u1 + u2*u2;
	}
	while (w >= 1 || w == 0);

	mult = sqrt(-2*log(w)/w);
	x1 = u1 * mult;
	x2 = u2 * mult;

	call = !call;

	return mu + sigma * (double)x1;
}

int rround(double d)
{
	double f = floor(d);
	if (random01() < d - f)
		return (int)f + 1;
	else
		return (int)f;
}
