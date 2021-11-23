#ifndef RANDOM_H
#define RANDOM_H

// Generate random number in the continuous range [0,1)
double random01();

// Generate random number in the continuous range [min,max]
double uniform(double min, double max);

// Generate random number in the continuous range (-inf,inf) following ~N(mu,sigma)
double gaussian(double mu, double sigma);

// Round x to one of its two nearest integer inversely proportional to the distance from it.
int rround(double x);

#endif
