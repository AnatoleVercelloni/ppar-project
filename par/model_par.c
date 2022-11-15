#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <err.h>
#include<mpi.h>


#include "harmonics.h"

int lmax = -1;
int npoint;
char * data_filename;
char * model_filename;

void usage(char ** argv)
{
	printf("%s [OPTIONS]\n\n", argv[0]);
	printf("Options:\n");
	printf("--data FILENAME              input file containing experimental data points\n");
	printf("--model FILENAME             output file containing the model\n");
	printf("--npoint N                   number of points to read\n");
	printf("--lmax N                     order of the model\n");
	printf("\n");
	exit(0);
}

void process_command_line_options(int argc, char ** argv)
{
	struct option longopts[5] = {
		{"data", required_argument, NULL, 'd'},
		{"npoint", required_argument, NULL, 'n'},
		{"lmax", required_argument, NULL, 'l'},
		{"model", required_argument, NULL, 'm'},
		{NULL, 0, NULL, 0}
	};
	char ch;
	while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
		switch (ch) {
		case 'd':
			data_filename = optarg;
			break;
		case 'm':
			model_filename = optarg;
			break;
		case 'n':
			npoint = atoi(optarg);
			break;
		case 'l':
			lmax = atoll(optarg);
			break;
		default:
			errx(1, "Unknown option\n");
		}
	}
	/* missing required args? */
	if (data_filename == NULL || model_filename == NULL || lmax < 0 || npoint <= 0)
		usage(argv);
}

/**************************** LINEAR ALGEBRA *********************************/


/*
 * Return the euclidean norm of x[0:n] using tricks for a greater precision
 */
double norm(int n, double const *x)
{
   double rdwarf = 3.834e-20, rgiant = 1.304e19;
   double s1 = 0, s2 = 0, s3 = 0;
   double x1max = 0, x3max = 0;
   double agiant = rgiant / ((double) n);
   for (int i = 0; i < n; i++) {
       double xabs = fabs(x[i]);
       if (xabs > rdwarf && xabs < agiant) {  // sum for intermediate components
	   s2 += xabs * xabs;
	   continue;
       }
       if (xabs <= rdwarf) {                         // sum for small components 
	   if (xabs > x3max) {
	       double d3 = x3max / xabs;
	       s3 = 1 + s3 * (d3 * d3);
	       x3max = xabs;
	       continue;
	   }
	   if (xabs != 0) {
	       double d4 = xabs / x3max;
	       s3 += d4 * d4;
	   }
	   continue;
       }
       if (xabs <= x1max) {                          // sum for large components
	   double d2 = xabs / x1max;
	   s1 += d2 * d2;
	   continue;
       }
       double d1 = x1max / xabs;
       s1 = 1 + s1 * (d1 * d1);
       x1max = xabs;
   }
   if (s1 == 0) {                                         // calculation of norm
       if (s2 == 0)
	   return x3max * sqrt(s3);
       if (s2 >= x3max)
	   return sqrt(s2 * (1 + x3max / s2 * (x3max * s3)));
       if (s2 < x3max)
	   return sqrt(x3max * (s2 / x3max + x3max * s3));
   }
   return x1max * sqrt(s1 + s2 / x1max / x1max);
}

/*
 * Apply a real elementary reflector H to a real m-by-n matrix C. H is
 * represented in the form
 *
 *       H = I - tau * v * v**T
 *
 * where tau is a real scalar and v is a real vector.
 *
 * C is a 2D array of dimension (ldc, n).  On exit, C is overwritten with H*C.
 * It is required that ldc >= m.
 */
void multiply_householder(int m, int n, double *v, double tau, double *c, int ldc)
{
	for (int j = 0; j < n; j++) {
		double sum = 0;
		for (int i = 0; i < m; i++)
			sum += c[j * ldc + i] * v[i];
		for (int i = 0; i < m; i++)
			c[j * ldc + i] -= tau * v[i] * sum;
	}
}

/*
 * Compute a QR factorization of a real m-by-n matrix A (with m >= n).
 *
 * A = Q * ( R ),       where:        Q is a m-by-n orthogonal matrix
 *         ( 0 )                      R is an upper-triangular n-by-n matrix
 *                                    0 is a (m-n)-by-n zero matrix
 *
 * A is a 2D array of dimension (m, n)
 * On exit, the elements on and above the diagonal contain R; the elements below
 * the diagonal, with the array tau, represent the orthogonal matrix Q.
 *
 * Q is represented as a product of n elementary reflectors
 *
 *     Q = H(1) * H(2) * ... * H(n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau[i] * v * v**T
 *
 * where tau[i] is a real scalar, and v is a real vector with v[0:i-1] = 0 and 
 * v[i] = 1; v[i+1:m] is stored on exit in A[i+1:m, i].
 */
void QR_factorize(int m, int n, double * A, double * tau)
{
	for (int i = 0; i < n; ++i) {
		/* Generate elementary reflector H(i) to annihilate A(i+1:m,i) */
		double aii = A[i + i * m];
		double anorm = -norm(m - i, &A[i + i * m]);
		if (aii < 0)
			anorm = -anorm;
		tau[i] = (anorm - aii) / anorm;
		for (int j = i + 1; j < m; j++)
			A[i * m + j] /= (aii - anorm);

		/* Apply H(i) to A(i:m,i+1:n) from the left */
		A[i + i * m] = 1;
		multiply_householder(m-i, n-i-1, &A[i*m + i], tau[i], &A[(i+1)*m + i], m);
		A[i + i * m] = anorm;
	}
}

/*
 * Overwrite vector c with transpose(Q) * c where Q is a
 * real m-by-m orthogonal matrix defined as the product of k elementary
 * reflectors
 *
 *       Q = H(1) * H(2) ... H(k)
 * 
 * A is a 2D array of dimension (m, k), which contains a QR factorisation
 * computed by QR_factorize().  A is not modified.
 *
 * tau is an array of dimension k. tau[i] must contain the scalar factor of the
 * elementary reflector H(i), as returned by QR_factorize().  tau is read-only.
 *
 * c is a vector of dimension m.  On exit, c is overwritten by transpose(Q)*c.
 */
void multiply_Qt(int m, int k, double * A, double * tau, double * c)
{
	for (int i = 0; i < k; i++) {
		/* Apply H(i) to A[i:m] */
		double aii = A[i + i * m];
		A[i + i * m] = 1;
		multiply_householder(m-i, 1, &A[i*m + i], tau[i], &c[i], m);
		A[i + i * m] = aii;
	}
}

/*
 * Solve the triangular linear system U*x == b
 *
 * U is a 2D array of dimension (ldu, n) with non-zero diagonal entries. Only
 * the upper-triangle is read by this function. b and x are n element vectors.
 * On exit, b is overwritten with x.
 */
void triangular_solve(int n, const double *U, int ldu, double *b) 
{
	for (int k = n - 1; k >= 0; k--) {
	   b[k] /= U[k * ldu + k];
	   for (int i = 0; i < k; i++)
	       b[i] -= b[k] * U[i + k*ldu];
    }
}

/*
 * Solve the least-Squares Problem min || A*x - b || for overdetermined real
 * linear systems involving an m-by-n matrix A using a QR factorization of A.
 * It is assumed that A has full rank (and m >= n).
 *
 * A is a 2D array of dimension (m, n).  On exit, A is overwritten by the 
 * details of its QR factorization (cf. QR_factorize).
 *
 * b is a vector of size m.  On exit, b[0:n] contain the least squares solution 
 * vector; the residual sum of squares for the solution is given by the sum of 
 * squares of b[n:m].
 */
void linear_least_squares(int m, int n, double *A, double *b)
{
	assert(m >= n);

	double tau[n];
	QR_factorize(m, n, A, tau);                    /* QR factorization of A */
	multiply_Qt(m, n, A, tau, b);                /* B[0:m] := Q**T * B[0:m] */
	triangular_solve(n, A, m, b);          /* B[0:n] := inv(R) * B[0:n] */
}

/*****************************************************************************/

int main(int argc, char ** argv)
{


	MPI_Init(&argc, &argv);
	int rank = 0;
	int p = 0;
	p = rank;
	PI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	process_command_line_options(argc, argv);


	int nvar = (lmax + 1) * (lmax + 1); //changer ca
	printf("Linear Least Squares with dimension %d x %d\n", npoint, nvar);
	if (nvar > npoint)
		errx(1, "not enough data points");
	
	int step = ceil(nvar/p); //changer ca

	long matrix_size = sizeof(double) * slice * npoint; //chaque process a une petite partie de la matrice
	char hsize[16];
	human_format(hsize, matrix_size);
	printf("Matrix size: %sB\n", hsize); //facteur limitant taille, parallelisation

	double *A = malloc(matrix_size);
	if (A == NULL)
		err(1, "cannot allocate matrix");

	// bonne chose de diviser ca aussi mais plus tard
	double * P = malloc((lmax + 1) * (lmax + 1) * sizeof(*P));
	double * v = malloc(npoint * sizeof(*v));
	
	if (P == NULL || v == NULL)
		err(1, "cannot allocate data points\n");


	printf("Reading data points from %s\n", data_filename);
	struct data_points data;
	load_data_points(data_filename, npoint, &data); //modifie 
	printf("Successfully read %d data points\n", npoint);
	
	printf("Building matrix\n");
	struct spherical_harmonics model;
	setup_spherical_harmonics(lmax, &model); //a modifier
	

	// modifie, a verifier
	for (int i = 0; i < npoint; i++) {
		computeP(&model, P, sin(data.phi[i]));
		
		for (int l = 0; l <= lmax + 1; l++) { //Je suis pas du tout sur...
			/* zonal term */
			A[i + npoint * CT(l, 0)] = P[PT(l, 0)];
	
			/* tesseral terms */
			for (int m = 1; m <= l; m++) {
				A[i + npoint * CT(l, m)] = P[PT(l, m)] * cos(m * data.lambda[i]);
				A[i + npoint * ST(l, m)] = P[PT(l, m)] * sin(m * data.lambda[i]);
			}
		}
	}
	
	double FLOP = 2. * nvar * nvar * npoint;
	char hflop[16];
	human_format(hflop, FLOP);
	printf("Least Squares (%sFLOP)\n", hflop);
	double start = wtime();
	
	/* the real action takes place here */
	linear_least_squares(npoint, nvar, A, data.V); 
	//chaque process lance  cette fonction avec sa partie de la matrice
	
	double t = wtime()  - start;

	// MPI_Finalize();

	double FLOPS = FLOP / t;
	char hflops[16];
	human_format(hflops, FLOPS);
	printf("Completed in %.1f s (%s FLOPS)\n", t, hflops);

	double res = 0;
	for (int j = nvar; j < npoint; j++)
		res += data.V[j] * data.V[j];
	printf("residual sum of squares %g\n", res);

	printf("Saving model in %s\n", model_filename);
	FILE *g = fopen(model_filename, "w");
	if (g == NULL)
		err(1, "cannot open %s for writing\n", model_filename);
	for (int l = 0; l <= lmax; l++) {
		fprintf(g, "%d\t0\t%.18g\t0\n", l, data.V[CT(l, 0)]);
		for (int m = 1; m <= l; m++)
			fprintf(g, "%d\t%d\t%.18g\t%.18g\n", l, m, data.V[CT(l, m)], data.V[ST(l, m)]);
	}
}
