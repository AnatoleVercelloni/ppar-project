#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <assert.h>
#include <sys/time.h>
#include <inttypes.h>
#include <string.h>

#include "harmonics.h"

double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double) ts.tv_sec + ts.tv_usec / 1e6;
}

/* represent n in <= 8 char  */
void human_format(char * target, long n) {
        if (n < 1000) {
                sprintf(target, "%ld" PRId64, n);
                return;
        }
        if (n < 1000000) {
                sprintf(target, "%.1f K", n / 1e3);
                return;
        }
        if (n < 1000000000) {
                sprintf(target, "%.1f M", n / 1e6);
                return;
        }
        if (n < 1000000000000ll) {
                sprintf(target, "%.1f G", n / 1e9);
                return;
        }
        if (n < 1000000000000000ll) {
                sprintf(target, "%.1f T", n / 1e12);
                return;
        }
}


void setup_spherical_harmonics(int lmax, struct spherical_harmonics *self)
{
	self->lmax = lmax;
	int sizeCS = (lmax + 1) * (lmax + 1);
	int sizeAB = (lmax + 1) * (lmax + 2) / 2;

	self->CS = malloc(sizeCS * sizeof(double));
	self->A = malloc(sizeAB * sizeof(double));
	self->B = malloc(sizeAB * sizeof(double));
	if (self->CS == NULL || self->A == NULL || self->B == NULL)
		err(1, "cannot allocate harmonics\n");

	/* compute the A, B coefficients */
	for (int l = 2; l <= lmax; l++) {
		double ls = l * l;
		double lm1s = (l - 1) * (l - 1);
		for (int m = 0; m < l - 1; m++) {
			double ms = m * m;
			self->A[PT(l, m)] = sqrt((4 * ls - 1) / (ls - ms));
			self->B[PT(l, m)] = -sqrt((lm1s - ms) / (4 * lm1s - 1));
		}
	}
}

void load_data_points(const char *filename, int npoint, struct data_points *self)
{
	self->npoint = npoint;
	self->phi = malloc(npoint * sizeof(double));
	self->lambda = malloc(npoint * sizeof(double));
	self->V = malloc(npoint * sizeof(double));
	if (self->phi == NULL || self->lambda == NULL || self->V == NULL)
		err(1, "cannot allocate data points\n");

	FILE *f = fopen(filename, "r");
	if (f == NULL)
		err(1, "cannot open %s\n", filename);
	
	//On recherche le nombre de points total dans le fichier "tot"
	int tot = 0;
	char * tabfile[4] = {"../ETOPO1_small.csv", "../ETOPO1_medium.csv", "../ETOPO1_high.csv", "../ETOPO1_ultra.csv"};
	int tabtot[4] = {64800, 583000, 6480000, 233280000};
	for (int i = 0; i < 4; i++){
		if (strcmp(filename, tabfile[i]) == 0) {
			tot = tabtot[i];
		}
	}
	if (tot == 0) {
		err(1, "%s not in the database\n", filename);
	}
	
	//On parcours tout le fichier, mais on n enregistre le point que selon un certain pas "step"
	// Et jusqu a obtenir npoint (grace a la variable "j")
	int step = tot / npoint;
	int j = 0;
	for (int i = 0; i < tot; i++) {
		double temp[3];
		int k = fscanf(f, "%lg %lg %lg", &temp[0], &temp[1], &temp[2]);
		if (i % step == 0) {
			self->lambda[j] = temp[0];
			self->phi[j] = temp[1];
			self->V[j] = temp[2];
			if (++j == npoint) break;
		}

		if (k == EOF) {
			if (ferror(f))
				err(1, "read error");
			errx(1, "premature end-of-file after %d records", i);
		}
		if (k != 3)
			errx(1, "parse error on line %d", i+1);
	}
	fclose(f);
}

void load_spherical_harmonics(const char *filename, int lmax, struct spherical_harmonics *self)
{
	FILE *g = fopen(filename, "r");
	if (g == NULL)
		err(1, "cannot open %s", filename);
	setup_spherical_harmonics(lmax, self);
	for (;;) {
		int l, m;
		double c, s;
		int k = fscanf(g, "%d %d %lg %lg", &l, &m, &c, &s);
		if (m == 0 && s != 0)
			errx(1, "non-zero S coefficient with l=%d and m=0", l);
		self->CS[CT(l, m)] = c;
		if (m > 0)
			self->CS[ST(l, m)] = s;
		if (k == EOF) {
			if (ferror(g))
				err(1, "read error");
			break;
		}
		if (k != 4)
			errx(1, "parse error");
	}
	fclose(g);
}

/*
 * Compute all the (fully normalized) Associated Legendre function of degree <= lmax.
 * On exit, P_{l,m} (with 0 <= m <= l <= lmax) can be found in P[PT(l, m)].
 * P must be preallocated of size (lmax * lmax + 3) / 2.
 * The constants A and B must be precomputed.
 */
void computeP(const struct spherical_harmonics *self, double *P, double sinphi)
{
	double cosphi = sqrt(1 - sinphi * sinphi);
	double temp = 1;
	P[PT(0, 0)] = temp;
	if (self->lmax == 0)
		return;
	P[PT(1, 0)] = sinphi * sqrt(3) * temp;
	temp = 0.5 * sqrt(3) * cosphi * temp;
	P[PT(1, 1)] = temp;
	for (int l = 2; l <= self->lmax; l++) {
		for (int m = 0; m < l - 1; m++)
			P[PT(l, m)] = self->A[PT(l, m)] * (sinphi * P[PT(l - 1, m)] + self->B[PT(l, m)] * P[PT(l - 2, m)]);
		P[PT(l, l - 1)] = sinphi * sqrt(2 * (l - 1) + 3) * temp;
		temp = -sqrt(1.0 + 0.5 / l) * cosphi * temp;
		P[PT(l, l)] = temp;
	}
}

double evaluate(const struct spherical_harmonics *self, const double *P, double lambda)
{
	int lmax = self->lmax;
	int sizeCS = (lmax + 1) * (lmax + 1);
	double scratch[sizeCS];

	for (int l = 0; l <= lmax; l++) {
		/* zonal term */
		scratch[CT(l, 0)] = P[PT(l, 0)];

		/* tesseral terms */
		for (int m = 1; m <= l; m++) {
			scratch[CT(l, m)] = P[PT(l, m)] * cos(m * lambda);
			scratch[ST(l, m)] = P[PT(l, m)] * sin(m * lambda);
		}
	}

	/* dot product */
	double V = 0;	
	for (int i = 0; i < sizeCS; i++)
		V += scratch[i] * self->CS[i];
	return V;
}