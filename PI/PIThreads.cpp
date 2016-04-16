#include <iostream>
#include <unistd.h>
#include <limits>
#include <vector>

typedef std::numeric_limits<double> dbl;

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

    vector<pthread_t> threads(processes);
    vector<Task> tasks (processes);

    //Map tasks
    for (auto i = 0u; i < processes; ++i) {
        tasks[i] = Task();
        tasks[i].start = iter / processes * i;
        tasks[i].end = iter / processes * (i + 1);
        if (pthread_create(&threads[i], NULL, PartialLeibniz, (void *) &tasks[i] )) {
            cerr << "pthread_create() failed" << endl;
            return EXIT_FAILURE;
        }
    }

    //Reduce results
    double sum = 0.0;
    for (auto i = 0u; i < processes; ++i) {
        pthread_join(threads[i], NULL);
        sum += tasks[i].res;
    }
    cout << "Threads\t" << processes << endl;
    cout << "Iterations\t" << iter << endl;
    cout.precision(dbl::max_digits10);
    cout << "Sum\t\t" << sum << endl;
    cout << "PI\t\t" << sum * 4.0 << endl;
    return EXIT_SUCCESS;
}