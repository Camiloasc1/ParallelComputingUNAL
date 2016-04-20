#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <pthread.h>

struct Task {
    unsigned long start;
    unsigned long end;
    double res;
};

void *PartialLeibniz(void *task) {
    struct Task *myTask = (struct Task *) task;
    myTask->res = 0.0;
    for (long i = myTask->start; i < myTask->end; ++i) {
        if (i % 2 == 0)
            myTask->res += 1.0 / ((i << 1) + 1);
        else
            myTask->res -= 1.0 / ((i << 1) + 1);
    }
    return NULL;
}

int main(int argc, char *argv[]) {

    unsigned int processes = 8u;
    unsigned long iter = 1000000000ul; // 1E9
    if (argc > 1) {
        processes = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        iter = (unsigned long) atol(argv[2]);
    }

    pthread_t *threads;
    struct Task *tasks;
    threads = (pthread_t *) malloc(processes * sizeof(pthread_t));
    tasks = (struct Task *) malloc(processes * sizeof(struct Task));

    //Map tasks
    for (int i = 0; i < processes; ++i) {
        //tasks[i] = Task();
        tasks[i].start = iter / processes * i;
        tasks[i].end = iter / processes * (i + 1);
        if (pthread_create(&threads[i], NULL, PartialLeibniz, (void *) &tasks[i])) {
            perror("pthread_create() failed");
            return EXIT_FAILURE;
        }
    }

    //Reduce results
    double sum = 0.0;
    for (int i = 0; i < processes; ++i) {
        pthread_join(threads[i], NULL);
        sum += tasks[i].res;
    }
    printf("Processes\t%d\n", processes);
    printf("Iterations\t%ld\n", iter);
    printf("Sum\t\t%f\n", sum);
    printf("PI\t\t%f\n", sum * 4.0);
    free(threads);
    free(tasks);
    return EXIT_SUCCESS;
}
