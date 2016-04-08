#include <iostream>
#include <unistd.h>

#define LIMIT (unsigned int) 1E9

using namespace std;

struct Task {
    unsigned int start;
    unsigned int end;
    double res;
};

void *PartialLeibniz(void *task) {
    Task *myTask = (Task *) task;
    myTask->res = 0.0;
    for (auto i = myTask->start; i < myTask->end; i++) {
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

    pid_t pid = fork();
    if (pid == 0) {
        // Child
        Task myTask;
        read(taskpipe[0], &myTask, sizeof(Task));
        PartialLeibniz(&myTask);
        write(respipe[1], &myTask, sizeof(Task));
    }
    else if (pid > 0) {
        // Parent
        Task task;
        task.start = 0;
        task.end = LIMIT;
        write(taskpipe[1], &task, sizeof(Task));
        read(respipe[0], &task, sizeof(Task));
        cout << "Sum\t" << task.res << endl;
        cout << "PI\t" << task.res * 4.0 << endl;
    }
    else {
        // Error
        cerr << "fork() failed" << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}