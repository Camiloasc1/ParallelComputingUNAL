build:
	gcc -std=c11 -o MatrixThreads ../MatrixThreads.c -lpthread
	gcc -std=c11 -o MatrixFork ../MatrixFork.c

all: build

run: build
	./run.sh

clean:
	rm MatrixThreads MatrixFork Speedup.dat Time.dat
