#include <curand.h>

#define CHUNK 1 //Run CHUNK blocks, each with 1024 threads (or with the specified argument) and check error
#define LOG 1024 // Print progress each LOG iterations
#define LIMIT 1024*1024 // LIMIT of iterations

#include "../common.h"

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
    unsigned int digits;
    unsigned int threads;
    double precision;
    getParams(argc, argv, &threads, &digits, &precision);

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
    while (error > precision && i < LIMIT) {
        curandGenerateUniformDouble(rnd, X, randomSize);
        curandGenerateUniformDouble(rnd, Y, randomSize);
        //@formatter:off
        MontecarloPI<<<CHUNK, threads>>>(d_inside, d_outside, X, Y);
        //@formatter:on
        cudaDeviceSynchronize();

        cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_outside, d_outside, sizeof(unsigned long), cudaMemcpyDeviceToHost);

        pi = 4.0 * h_inside / (h_outside + h_inside);
        error = getError(pi);
        printLog(precision, pi, error, ++i);
    }

    cudaFree(d_inside);
    cudaFree(d_outside);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
