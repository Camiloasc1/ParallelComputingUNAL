#define CHUNK 1 // Run CHUNK iterations and check error
#define LOG 1 // Print progress each LOG iterations
#define LIMIT 1024 // LIMIT of iterations

#include "../common.h"

int main(int argc, char *argv[]) {
    unsigned int digits;
    unsigned int threads;
    double precision;
    getParams(argc, argv, &threads, &digits, &precision);

    double pi=0.0, error=1.0;
    double a[2], b[2], t[2], p[2];

    a[0] = 1.0;
    b[0] = 1.0 / sqrt(2.0);
    t[0] = 1.0 / 4;
    p[0] = 1.0;

    unsigned long i = 0;
    while (error > precision && i < LIMIT) {
        a[1] = (a[0] + b[0]) / 2;
        b[1] = sqrt(a[0] * b[0]);
        t[1] = t[0] - p[0] * (a[0] - a[1]) * (a[0] - a[1]);
        p[1] = 2 * p[0];
        pi = (a[1] + b[1]) * (a[1] + b[1]) / (4 * t[1]);
        a[0] = a[1];
        b[0] = b[1];
        t[0] = t[1];
        p[0] = p[1];
        error = getError(pi);
        printLog(precision, pi, error, ++i);
    }

    return EXIT_SUCCESS;
}
