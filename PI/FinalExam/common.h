#pragma once

#ifndef PARALLELCOMPUTINGUNAL_COMMON_H
#define PARALLELCOMPUTINGUNAL_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
#define SEED 1 // (unsigned int) time(NULL)

void printLog(double precision, double pi, double error, unsigned long i);

double getError(double pi);

void getParams(int argc, char *const *argv, unsigned int *threads, unsigned int *digits, double *precision);

inline void printLog(double precision, double pi, double error, unsigned long i) {
    if (i % LOG == 0 || error < precision || i == LIMIT) {
        printf("Iteration\t%ld\n", i);
        printf("Precision\t%d (%s)\n", (unsigned int) floor(-log10(error)),
               i == LIMIT ? "Timeout" : error > precision ? "Still Working" : "Reached");
        printf("PI\t\t%.100f\n", pi);
        printf("Error\t\t%.100f\n", error);
    }
}

inline double getError(double pi) {
    double error = PI - pi;
    error = error > 0 ? error : -error;
    return error;
}

inline void getParams(int argc, char *const *argv, unsigned int *threads, unsigned int *digits, double *precision) {
    *digits = 10u;
#ifdef __CUDACC__
    *threads = 1024u;
#else
    *threads = 8u;
#endif
    if (argc > 1) {
        *digits = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        *threads = (unsigned int) atoi(argv[2]);
    }
    *precision = pow(10, -((double) *digits));
}

#endif //PARALLELCOMPUTINGUNAL_COMMON_H
