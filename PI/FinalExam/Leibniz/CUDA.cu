#define CHUNK 1 //Run CHUNK blocks, each with 1024 threads (or with the specified argument) and check error
#define LOG 1024 // Print progress each LOG iterations
#define LIMIT 1024*1024 // LIMIT of iterations

#include "../common.h"

__device__ double atomicAdd(double *, double);

__global__ void LeibnizPI(double *sum, unsigned long offset) {

    __shared__ double partialSum;

    if (threadIdx.x == 0) {
        partialSum = 0.0;
    }

    __syncthreads();

    unsigned long n = (offset * CHUNK + blockIdx.x) * blockDim.x + threadIdx.x;
    double val = ((n % 2 == 0) ? 1.0 : -1.0) / ((n << 1) + 1);
    atomicAdd(&partialSum, val);

    __syncthreads();

    if (threadIdx.x == 0) {
        *sum += partialSum;
    }
}

__device__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull =
            (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

int main(int argc, char *argv[]) {
    unsigned int digits;
    unsigned int threads;
    double precision;
    getParams(argc, argv, &threads, &digits, &precision);

    double h_sum = 0.0;
    double *d_sum;

    cudaMalloc((void **) &d_sum, sizeof(double));

    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);

    double pi = 0.0, error = 1.0;
    unsigned long i = 0;
    while (error > precision && i < LIMIT) {
        //@formatter:off
        LeibnizPI<<<CHUNK, threads>>>(d_sum, i);
        //@formatter:on
        cudaDeviceSynchronize();

        cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

        pi = 4.0 * h_sum;
        error = getError(pi);
        printLog(precision, pi, error, ++i);
    }

    cudaFree(d_sum);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
