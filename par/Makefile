LDLIBS = -lm 
CFLAGS = -Wall -g -O3
#LDFLAGS = -pg

ALL: model validate model_par

model_par: model_par.o harmonics.o
validate: validate.o harmonics.o 




model_par.o: model_par.c harmonics.h
	mpicc -c model_par.c

quality.o: validate.c harmonics.h
	mpicc -c validate.c

harmonics.o: harmonics.c harmonics.h
	mpicc -c harmonics.c
.PHONY: clean

clean:
	rm -f model validate model_par *.o