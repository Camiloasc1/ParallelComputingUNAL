#define CHUNK 1024 // Run CHUNK iterations and check error
#define LOG 1024 // Print progress each LOG iterations
#define LIMIT 1024*1024 // LIMIT of iterations

#include "../common.h"

int main(int argc, char *argv[]) {
    unsigned int digits;
    unsigned int threads;
    double precision;
    getParams(argc, argv, &threads, &digits, &precision);

    srand(SEED);

    unsigned long inside, outside;
    inside = 0;
    outside = 0;

    double pi, error;
    pi = 0.0;
    error = 1.0;

    omp_set_num_threads(threads);
    unsigned long i = 0;
    while (error > precision && i < LIMIT) {
#pragma omp parallel for reduction(+:inside) reduction(+:outside)
        for (unsigned long n = 0ul; n < CHUNK; ++n) {
            double x = (double) rand() / (double) RAND_MAX;
            double y = (double) rand() / (double) RAND_MAX;
            if (x * x + y * y < 1.0) {
                inside++;
            }
            else {
                outside++;
            }
        }
        pi = 4.0 * inside / (outside + inside);
        error = getError(pi);
        printLog(precision, pi, error, ++i);
    }

    return EXIT_SUCCESS;
}
