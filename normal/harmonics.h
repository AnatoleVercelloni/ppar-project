#ifndef GEODESY_H
#define GEODESY_H
#include <assert.h>

/* representation of spherical harmonics coefficients */
struct spherical_harmonics {
	int lmax;
	double *CS;
	double *A;
	double *B;
};

/* representation of a sequence of data points */
struct data_points {
	int npoint;
	double *phi;
	double *lambda;
	double *V;
};

/* these 3 functions help compute indices in arrays containing
   triangular matrices */
static inline int PT(int l, int m)   /* classic triangular w/ diagonal */
{
	return m + l * (l + 1) / 2;
}

static inline int CT(int l, int m)
{
	assert(m <= l);
	return m + l * l;
}

static inline int ST(int l, int m)   /* all Sx0 are missing */
{
	assert(m <= l);
	assert(1 <= m);
	return m + (l+1) * l;
}

/* wall-clock seconds (with high precision) elapsed since some given point in the past */
double wtime();

/* represent n in <= 8 char, in a human-readable way  */
void human_format(char * target, long n);

/* allocates memory for self */
void setup_spherical_harmonics(int lmax, struct spherical_harmonics *self);

/*
 * Load spherical harmonics from a file.  File format:
 * each line contains 2 integers and 2 floating-point numbers separated by tabs.
 * Each line contains (l, m, c, s) with 0 <= m <= l <= lmax.
 * If m == 0, then s == 0.
 */
void load_spherical_harmonics(const char *filename, int lmax, struct spherical_harmonics *self);

/*
 * Load data points from a file into "self".  File format:
 * each line contains 3 floating-point numbers separated by tabs.
 * Each line contains (lambda, phi, V) where V == f(phi, lambda).
 */
void load_data_points(const char *filename, int npoint, struct data_points *self);

/*
 * Compute all the (fully normalized) Associated Legendre function of degree <= lmax.
 * On exit, P_{l,m} (with 0 <= m <= l <= lmax) can be found in P[PT(l, m)].
 * P must be preallocated of size (lmax + 1) * (lmax + 2) / 2.
 */
void computeP(const struct spherical_harmonics *self, double *P, double sinphi);

/* phi   : elevation angle (latitude) in degrees north of the equator plane, 
           in the range -PI/2 <= phi <= PI/2 
   lambda: azimuth angle (longitude) measured in degrees east or west from some
           conventional reference meridian.
   P must be previously evaluated with a call to computeP(self, P, sin(phi));
*/
double evaluate(const struct spherical_harmonics *self, const double *P, double lambda);

#endif