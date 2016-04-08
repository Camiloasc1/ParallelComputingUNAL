#include <iostream>
#include <unistd.h>
#include <limits>

typedef std::numeric_limits< double > dbl;

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

int main(int argc, char *argv[]) {

    auto processes = 8u;
    auto iter = 1000000000ul; // 1E9
    if (argc > 1) {
        processes = (unsigned int) atoi(argv[1]);
    }
    if (argc > 2) {
        iter = (unsigned long) atol(argv[2]);
    }

    int taskpipe[2], respipe[2];
    if (pipe(taskpipe) == -1 || pipe(respipe)) {
        cerr << "pipe() failed" << endl;
        exit(EXIT_FAILURE);
    }

    //Map tasks
    for (auto i = 0u; i < processes; ++i) {
        Task task;
        task.start = iter / processes * i;
        task.end = iter / processes * (i + 1);
        write(taskpipe[1], &task, sizeof(Task));
    }

    //Fork
    for (auto i = 0u; i < processes; ++i) {
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

    //Reduce results
    double sum = 0.0;
    for (auto i = 0u; i < processes; ++i) {
        Task res;
        read(respipe[0], &res, sizeof(Task));
        sum += res.res;
    }
    cout << "Processes\t" << processes << endl;
    cout << "Iterations\t" << iter << endl;
    cout.precision(dbl::max_digits10);
    cout << "Sum\t\t" << sum << endl;
    cout << "PI\t\t" << sum * 4.0 << endl;
    return EXIT_SUCCESS;
}