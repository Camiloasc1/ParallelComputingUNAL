#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679

int main(int argc, char *argv[]) {
    unsigned int digits = 10;
    unsigned int threads = 8u;
    if (argc > 1) {
        digits = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        threads = (unsigned int) atoi(argv[2]);
    }
    double prec = pow(10,(double)-digits);
    printf("Prec\t%f\n", (double)-digits);

    long double pi, error;
    long double* a,*b,*t,*p;
    a = (long double *) malloc(digits * sizeof(long double));
    b = (long double *) malloc(digits * sizeof(long double));
    t = (long double *) malloc(digits * sizeof(long double));
    p = (long double *) malloc(digits * sizeof(long double));

    a[0] = 1.0;
    b[0] = 1.0/sqrt(2.0);
    t[0] = 1.0/4;
    p[0] = 1.0;
    pi = 3.0;
    error = 1.0;
    unsigned long i = 0ul;

    while(error > prec) {
        a[i+1] = (a[i]+b[i])/2;
        b[i+1] = sqrt(a[i]*b[i]);
        t[i+1] = t[i] - p[i]*(a[i]-a[i+1])*(a[i]-a[i+1]);
        p[i+1] = 2*p[i];
        pi = (a[i+1]+b[i+1])*(a[i+1]+b[i+1])/(4*t[i+1]);
        error = PI - pi;
        i++;
    }

    printf("Iterations\t%ld\n", i);
    printf("PI\t\t%.100Lf\n", pi);
    printf("Error\t\t%.100Lf\n", error);
    return EXIT_SUCCESS;
}
