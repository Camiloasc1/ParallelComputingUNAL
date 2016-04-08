#include <iostream>
#include <unistd.h>

#define LIMIT (unsigned long) 1E10
#define PROCESSES 8

using namespace std;

struct Task {
    unsigned long start;
    unsigned long end;
    double res;
};

void *PartialLeibniz(void *task) {
    Task *myTask = (Task *) task;
    myTask->res = 0.0;
    for (auto i = myTask->start; i < myTask->end; ++i) {
        if (i % 2 == 0)
            myTask->res += 1.0 / ((i << 1) + 1);
        else
            myTask->res -= 1.0 / ((i << 1) + 1);
    }
    return nullptr;
}

int main() {

    int taskpipe[2], respipe[2];
    if (pipe(taskpipe) == -1 || pipe(respipe)) {
        cerr << "pipe() failed" << endl;
        exit(EXIT_FAILURE);
    }

    for (auto i = 0u; i < PROCESSES; ++i) {
        Task task;
        task.start = LIMIT / PROCESSES * i;
        task.end = LIMIT / PROCESSES * (i + 1);
        write(taskpipe[1], &task, sizeof(Task));
    }

    for (auto i = 0u; i < PROCESSES; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child
            Task myTask;
            read(taskpipe[0], &myTask, sizeof(Task));
            PartialLeibniz(&myTask);
            write(respipe[1], &myTask, sizeof(Task));
            return EXIT_SUCCESS;
        }
        else if (pid > 0) {
            // Parent
        }
        else {
            // Error
            cerr << "fork() failed" << endl;
            return EXIT_FAILURE;
        }
    }
    double sum = 0.0;
    for (auto i = 0u; i < PROCESSES; ++i) {
        Task res;
        read(respipe[0], &res, sizeof(Task));
        sum += res.res;
    }
    cout << "Sum\t" << sum << endl;
    cout << "PI\t" << sum * 4.0 << endl;
    return EXIT_SUCCESS;
}