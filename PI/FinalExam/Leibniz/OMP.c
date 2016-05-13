#define CHUNK 1024*1024 // Run CHUNK iterations and check error
#define LOG 1024 // Print progress each LOG iterations
#define LIMIT 1024*1024 // LIMIT of iterations

#include "../common.h"

int main(int argc, char *argv[]) {
    unsigned int digits;
    unsigned int threads;
    double precision;
    getParams(argc, argv, &threads, &digits, &precision);

    double sum= 0.0, pi, error= 1.0;

    omp_set_num_threads(threads);
    unsigned long i = 0;
    while (error > precision && i < LIMIT) {
#pragma omp parallel for reduction(+:sum)
        for (unsigned long n = i * CHUNK; n < (i + 1) * CHUNK; ++n) {
            if (n % 2 == 0)
                sum += 1.0 / ((n << 1) + 1);
            else
                sum -= 1.0 / ((n << 1) + 1);
        }
        pi = 4.0 * sum;
        error = getError(pi);
        printLog(precision, pi, error, ++i);
    }

    return EXIT_SUCCESS;
}
