
all: example1 example2 example3

example1: example1.cu
	nvcc -o $@ -g $< -arch sm_61 --default-stream per-thread

example2: example2.cu
	nvcc -o $@ -g $< -arch sm_61 --default-stream per-thread

example3: example3.cu
	nvcc -o $@ -g $< -arch sm_61 --default-stream per-thread

clean:
	rm -f example1 example2 example3
