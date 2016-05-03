#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

unsigned long num_steps = 1000000000ul; // 1E9
double step;

#define NUM_THREADS 2

int main() {

    int nthreads = 0;
    long double pi, sum[NUM_THREADS];
    step = 1.0 / (double) num_steps;
    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + nthrds) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    int i;
    for (i = 0, pi = 0.0; i < nthreads; i++) pi += sum[i] * step;

    printf("Iterations\t%ld\n", num_steps);
    printf("PI\t\t%.100Lf\n", pi);
    return EXIT_SUCCESS;
}