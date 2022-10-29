#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <err.h>

#include "harmonics.h"

int lmax = -1;
char * data_filename;
char * model_filename;

void usage(char ** argv)
{
	printf("%s [OPTIONS]\n\n", argv[0]);
	printf("Options:\n");
	printf("--data FILENAME              CSV file containing experimental data points\n");
	printf("--model FILENAME             file containing the model\n");        
	printf("--lmax N                     order of the model\n");
	printf("\n");
	exit(0);
}

void process_command_line_options(int argc, char ** argv)
{
	struct option longopts[5] = {
		{"data", required_argument, NULL, 'd'},
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
		case 'l':
			lmax = atoll(optarg);
			break;
		default:
			errx(1, "Unknown option\n");
		}
	}
	/* missing required args? */
	if (data_filename == NULL || model_filename == NULL || lmax < 0)
		usage(argv);
}


int main(int argc, char ** argv)
{
	process_command_line_options(argc, argv);

	/* preparations and memory allocation */
	double * P = malloc((lmax + 1) * (lmax + 1) * sizeof(*P));
	if (P == NULL)
		err(1, "cannot allocate data points\n");

	printf("Reading order-%d model from %s\n", lmax, model_filename);
	struct spherical_harmonics model;
	load_spherical_harmonics(model_filename, lmax, &model);
	printf("Successfully read model\n");

	/* evaluate the model on corresponding input points */
	FILE *f = fopen(data_filename, "r");
	if (f == NULL)
		err(1, "cannot open %s", data_filename);
	
	printf("Reading data points from %s\n", data_filename);
	double max_error = 0;
	double error_l1 = 0;
	double error_l2 = 0;
	int npoint = 0;
	double start = wtime();
	for (;;) {
		/* read data point */
		double lambda, phi, v;
		int k = fscanf(f, "%lg %lg %lg", &lambda, &phi, &v);
		if (k == EOF) {
			if (ferror(f))
				err(1, "read error");
			if (feof(f))
				break;
		}
		npoint += 1;
		if (k != 3)
			errx(1, "parse error on line %d", npoint);

		/* compute error */
		computeP(&model, P, sin(phi));
		double vmodel = evaluate(&model, P, lambda);
		double error = fabs(v - vmodel);
		if (error > max_error)
			max_error = error;
		error_l1 += error;
		error_l2 += error * error;
	}
	printf("Completed in %.1f s\n", wtime() - start);
	printf("Average error %g meters\n", error_l1 / npoint);
	printf("Max error %g meters\n", max_error);
	printf("Standard Deviation %g\n", sqrt(error_l2 / npoint));
}