#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double **AllocateValueMatrix(unsigned int N, double value) {
    double **M;
    M = (double **) malloc(N * sizeof(double *));
    for (int i = 0; i < N; ++i) {
        M[i] = (double *) malloc(N * sizeof(double));
        for (int j = 0; j < N; ++j) {
            M[i][j] = value;
        }
    }
    return M;
}

double **AllocateRandomMatrix(unsigned int N) {
    double **M;
    M = (double **) malloc(N * sizeof(double *));
    for (int i = 0; i < N; ++i) {
        M[i] = (double *) malloc(N * sizeof(double));
        for (int j = 0; j < N; ++j) {
            M[i][j] = rand();
        }
    }
    return M;
}

void FreeMatrix(double **M, unsigned int N) {
    for (int i = 0; i < N; ++i) {
        free(M[i]);
    }
    free(M);
}

void PrintMatrix(double **M, unsigned int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", M[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {

    unsigned int threads = 8u;
    unsigned int N = 1024u;
    if (argc > 1) {
        threads = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        N = (unsigned int) atoi(argv[2]);
    }

    double **A;
    double **B;
    double **C;
    A = AllocateRandomMatrix(N);
    B = AllocateRandomMatrix(N);
    C = AllocateValueMatrix(N, 0.0);

    omp_set_num_threads(threads);
#pragma omp parallel for
    for (unsigned int i = 0u; i < N; ++i) {
        for (unsigned int j = 0u; j < N; ++j) {
            for (unsigned int k = 0u; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    printf("Threads\t%d\n", threads);
    printf("Size\t%dx%d\n", N, N);
    //printf("Result\n");
    //PrintMatrix(C, N);
    FreeMatrix(A, N);
    FreeMatrix(B, N);
    FreeMatrix(C, N);
    return EXIT_SUCCESS;
}
