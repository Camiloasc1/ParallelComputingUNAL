#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void PartialMM(double *A, double *B, double *C, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    double c = 0.0;
#pragma unroll
    for (unsigned int k = 0u; k < N; ++k) {
        c += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = c;
}

int main(int argc, char *argv[]) {

    unsigned int threads = 1024u;
    unsigned int N = 1024u;
    if (argc > 1) {
        threads = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        N = (unsigned int) atoi(argv[2]);
    }
    srand((unsigned int) time(NULL));

    unsigned int size = N * N * sizeof(double);
    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;

    h_A = (double *) malloc(size);
    h_B = (double *) malloc(size);
    h_C = (double *) malloc(size);

    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);

    for (unsigned int k = 0u; k < N * N; ++k) {
        h_A[k] = 2.0;//rand();
        h_B[k] = 3.0;//rand();
        h_C[k] = 0.0;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    dim3 block(1, 1);
    switch (threads) {
        case 1:
            block.x = 1;
            block.y = 1;
            break;
        case 2:
            block.x = 2;
            block.y = 1;
            break;
        case 4:
            block.x = 2;
            block.y = 2;
            break;
        case 8:
            block.x = 4;
            block.y = 2;
            break;
        case 16:
            block.x = 4;
            block.y = 4;
            break;
        case 32:
            block.x = 8;
            block.y = 4;
            break;
        case 64:
            block.x = 8;
            block.y = 8;
            break;
        case 128:
            block.x = 16;
            block.y = 8;
            break;
        case 256:
            block.x = 16;
            block.y = 16;
            break;
        case 512:
            block.x = 32;
            block.y = 16;
            break;
        case 1024:
            block.x = 32;
            block.y = 32;
            break;
    }
    dim3 grid(N / block.x, N / block.y);

    //@formatter:off
    PartialMM<<<grid, block>>>(d_A, d_B, d_C, N);
    //@formatter:on
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Size\t\t%dx%d\n", N, N);
    printf("Result:");
    for (unsigned int k = 0u; k < N * N; ++k) {
        if (k % N == 0)
            printf("\n");
        printf("%f ", h_C[k]);
    }

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
