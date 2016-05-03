#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    unsigned int processes = 8u;
    unsigned long iter = 1000000000ul; // 1E9
    if (argc > 1) {
        processes = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        iter = (unsigned long) atol(argv[2]);
    }

    long double pi = 0.0;

    omp_set_num_threads(processes);
#pragma omp parallel for reduction(+:pi)
    for (unsigned long i = 0ul; i < iter; ++i) {
        if (i % 2 == 0)
            pi += 1.0 / ((i << 1) + 1);
        else
            pi -= 1.0 / ((i << 1) + 1);
    }

    printf("Processes\t%d\n", processes);
    printf("Iterations\t%ld\n", iter);
    printf("Sum\t\t%.100Lf\n", pi);
    printf("PI\t\t%.100Lf\n", pi * 4.0);
    return EXIT_SUCCESS;
}
