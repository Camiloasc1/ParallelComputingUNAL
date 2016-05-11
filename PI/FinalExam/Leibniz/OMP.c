#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
#define CHUNK 1024*1024 // Run CHUNK iterations and check error
#define LOG 1024
#define LIMIT 1024*1024

int main(int argc, char *argv[]) {
    unsigned int digits = 10;
    unsigned int threads = 8u;
    if (argc > 1) {
        digits = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        threads = (unsigned int) atoi(argv[2]);
    }
    double prec = pow(10, -((double) digits));

    double sum, pi, error;
    sum = 0.0;
    pi = 0.0;
    error = 1.0;

    omp_set_num_threads(threads);
    unsigned long i = 0;
    while (error > prec && i < LIMIT) {
#pragma omp parallel for reduction(+:sum)
        for (unsigned long n = i * CHUNK; n < (i + 1) * CHUNK; ++n) {
            if (n % 2 == 0)
                sum += 1.0 / ((n << 1) + 1);
            else
                sum -= 1.0 / ((n << 1) + 1);
        }
        pi = 4.0 * sum;
        error = PI - pi;
        error = error > 0 ? error : -error;
        i++;
        if (i % LOG == 0 || error < prec || i == LIMIT) {
            printf("Iteration\t%ld\n", i);
            printf("Precision\t%d (%s)\n", (unsigned int) floor(-log10(error)),
                   i == LIMIT ? "Timeout" : error > prec ? "Still Working" : "Reached");
            printf("PI\t\t%.100f\n", pi);
            printf("Error\t\t%.100f\n", error);
        }
    }

    return EXIT_SUCCESS;
}
