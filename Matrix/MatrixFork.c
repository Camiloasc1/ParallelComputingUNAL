#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>

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

    int taskpipe[2], resultpipe[2];
    if (pipe(taskpipe) == -1 || pipe(resultpipe)) {
        perror("pipe() failed");
        return EXIT_FAILURE;
    }
    srand((unsigned int) time(NULL));

    //Map tasks
    double **A;
    double **B;
    double **C;
    A = AllocateRandomMatrix(N);
    B = AllocateRandomMatrix(N);
    C = AllocateValueMatrix(N, 0.0);
    {
        // First task
        unsigned int p = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i * N + j >= n * (p + 0) && i * N + j < n * (p + 1)) {
                    C[i][j] = 1.0;
                } else {
                    C[i][j] = 0.0;
                }
            }
        }
    }

    //Fork
    for (unsigned int p = 0; p < processes; ++p) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child
            struct Task myTask;
            myTask.A = A;
            myTask.B = B;
            myTask.C = C;
            myTask.N = N;
            PartialMM(&myTask);
            {
                // Wait
                int status;
                read(taskpipe[0], &status, sizeof(int));
            }
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    write(resultpipe[1], &C[i][j], sizeof(double));
                }
            }
            FreeMatrix(A, N);
            FreeMatrix(B, N);
            FreeMatrix(C, N);
            return EXIT_SUCCESS;
        }
        else if (pid > 0) {
            // Parent
            // Update next task
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (i * N + j >= n * (p + 1) && i * N + j < n * (p + 2)) {
                        C[i][j] = 1.0;
                    } else {
                        C[i][j] = 0.0;
                    }
                }
            }
        }
        else {
            // Error
            perror("fork() failed");
            return EXIT_FAILURE;
        }
    }

    //Reduce results
    double res;
    for (int p = 0; p < processes; ++p) {
        {
            // Ready
            int status;
            write(taskpipe[1], &status, sizeof(int));
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                read(resultpipe[0], &res, sizeof(double));
                C[i][j] += res;
            }
        }
    }
    printf("Processes\t%d\n", processes);
    printf("Size\t\t%dx%d\n", N, N);
    //printf("Result\n");
    //PrintMatrix(C, N);
    FreeMatrix(A, N);
    FreeMatrix(B, N);
    FreeMatrix(C, N);
    return EXIT_SUCCESS;
}
