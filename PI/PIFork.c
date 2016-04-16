#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include  <sys/types.h>

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

    int taskpipe[2], respipe[2];
    if (pipe(taskpipe) == -1 || pipe(respipe)) {
        perror("pipe() failed");
        return EXIT_FAILURE;
    }

    //Map tasks
    for (int i = 0; i < processes; ++i) {
        struct Task task;
        task.start = iter / processes * i;
        task.end = iter / processes * (i + 1);
        write(taskpipe[1], &task, sizeof(struct Task));
    }

    //Fork
    for (int i = 0; i < processes; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child
            struct Task myTask;
            read(taskpipe[0], &myTask, sizeof(struct Task));
            PartialLeibniz(&myTask);
            write(respipe[1], &myTask, sizeof(struct Task));
            return EXIT_SUCCESS;
        }
        else if (pid > 0) {
            // Parent
        }
        else {
            // Error
            perror("fork() failed");
            return EXIT_FAILURE;
        }
    }

    //Reduce results
    double sum = 0.0;
    for (int i = 0; i < processes; ++i) {
        struct Task res;
        read(respipe[0], &res, sizeof(struct Task));
        sum += res.res;
    }
    printf("Processes\t%d\n", processes);
    printf("Iterations\t%ld\n", iter);
    printf("Sum\t\t%f\n", sum);
    printf("PI\t\t%f\n", sum * 4.0);
    return EXIT_SUCCESS;
}
