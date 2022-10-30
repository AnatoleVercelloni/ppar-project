LDLIBS = -lm 
CFLAGS = -Wall -g -O3
#LDFLAGS = -pg

ALL: model validate model_par

model_par: model_par.o harmonics.o
model: model.o harmonics.o
validate: validate.o harmonics.o 
model.o: harmonics.h
model_par.o: harmonics.h
quality.o: harmonics.h
harmonics.o: harmonics.h

.PHONY: clean

clean:
	rm -f model validate model_par *.o