#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
#define CHUNK 1 //Run CHUNK blocks, each with 1024 threads (or with the specified argument) and check error
#define LOG 1024
#define LIMIT 1024*1024
#define SEED 1 // (unsigned int) time(NULL)

__global__ void MontecarloPI(unsigned long *inside, unsigned long *outside, double *X, double *Y) {

    __shared__ unsigned int in;
    __shared__ unsigned int out;

    if (threadIdx.x == 0) {
        in = 0;
        out = 0;
    }

    __syncthreads();

    double x = X[threadIdx.x];
    double y = Y[threadIdx.x];
    if (x * x + y * y < 1.0) {
        atomicInc(&in, blockDim.x + 1); // At most blockDim.x threads will sum here
    }
    else {
        atomicInc(&out, blockDim.x + 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        *inside += in;
        *outside += out;
    }
}

int main(int argc, char *argv[]) {
    unsigned int digits = 10;
    unsigned int threads = 1024u;
    if (argc > 1) {
        digits = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        threads = (unsigned int) atoi(argv[2]);
    }
    double prec = pow(10, -((double) digits));

    srand(SEED);

    unsigned int randomSize = threads;
    double *X, *Y;
    cudaMalloc((void **) &X, randomSize * sizeof(double));
    cudaMalloc((void **) &Y, randomSize * sizeof(double));

    curandGenerator_t rnd;
    curandCreateGenerator(&rnd, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(rnd, SEED);

    unsigned long h_inside, h_outside;
    unsigned long *d_inside, *d_outside;

    h_inside = 0;
    h_outside = 0;

    cudaMalloc((void **) &d_inside, sizeof(unsigned long));
    cudaMalloc((void **) &d_outside, sizeof(unsigned long));

    cudaMemcpy(d_inside, &h_inside, sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outside, &h_outside, sizeof(unsigned long), cudaMemcpyHostToDevice);

    double pi, error;
    pi = 0.0;
    error = 1.0;
    unsigned long i = 0;
    while (error > prec && i < LIMIT) {
        curandGenerateUniformDouble(rnd, X, randomSize);
        curandGenerateUniformDouble(rnd, Y, randomSize);
        //@formatter:off
        MontecarloPI<<<CHUNK, threads>>>(d_inside, d_outside, X, Y);
        //@formatter:on
        cudaDeviceSynchronize();

        cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_outside, d_outside, sizeof(unsigned long), cudaMemcpyDeviceToHost);

        pi = 4.0 * h_inside / (h_outside + h_inside);
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

    cudaFree(d_inside);
    cudaFree(d_outside);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
