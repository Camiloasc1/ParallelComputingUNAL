#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

struct Task {
    double **A;
    double **B;
    double **C;
    unsigned int N;
};

void *PartialMM(void *task) {
    struct Task *myTask = (struct Task *) task;
    for (unsigned int i = 0u; i < myTask->N; ++i) {
        for (unsigned int j = 0u; j < myTask->N; ++j) {
            if (myTask->C[i][j] > 0.1) {
                myTask->C[i][j] = 0.0;
                for (unsigned int k = 0u; k < myTask->N; ++k) {
                    myTask->C[i][j] += myTask->A[i][k] * myTask->B[k][j];
                }
            }
        }
    }
    return NULL;
}

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

    unsigned int processes = 8u;
    unsigned int N = 1024u;
    if (argc > 1) {
        processes = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        N = (unsigned int) atoi(argv[2]);
    }
    processes = N * N < processes ? N * N : processes;
    unsigned int n = N * N / processes;

    srand((unsigned int) time(NULL));

    pthread_t *threads;
    struct Task *tasks;
    threads = (pthread_t *) malloc(processes * sizeof(pthread_t));
    tasks = (struct Task *) malloc(processes * sizeof(struct Task));

    //Map tasks
    double **A;
    double **B;
    double **C;
    A = AllocateRandomMatrix(N);
    B = AllocateRandomMatrix(N);
    C = AllocateValueMatrix(N, 0.0);
    for (unsigned int p = 0; p < processes; ++p) {
        tasks[p].A = A;
        tasks[p].B = B;
        tasks[p].C = AllocateValueMatrix(N, 0.0);
        for (unsigned int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i * N + j >= n * (p) && i * N + j < n * (p + 1)) {
                    tasks[p].C[i][j] = 1.0;
                } else {
                    tasks[p].C[i][j] = 0.0;
                }
            }
        }
        tasks[p].N = N;
        if (pthread_create(&threads[p], NULL, PartialMM, (void *) &tasks[p])) {
            perror("pthread_create() failed");
            return EXIT_FAILURE;
        }
    }

    //Reduce results
    for (int p = 0; p < processes; ++p) {
        pthread_join(threads[p], NULL);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += tasks[p].C[i][j];
            }
        }
        FreeMatrix(tasks[p].C, N);
    }
    printf("Processes\t%d\n", processes);
    printf("Size\t\t%dx%d\n", N, N);
    //printf("Result\n");
    //PrintMatrix(C, N);
    FreeMatrix(A, N);
    FreeMatrix(B, N);
    FreeMatrix(C, N);
    free(threads);
    free(tasks);
    return EXIT_SUCCESS;
}
