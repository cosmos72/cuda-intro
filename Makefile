CUDA_ARCH=52

CUFLAGS=-arch sm_$(CUDA_ARCH) -DCUDA_ARCH="$(CUDA_ARCH)" --default-stream per-thread

all: example1 example2 example3 mean matmul

example1: example1.cu
	nvcc -o $@ -g $< $(CUFLAGS)

example2: example2.cu
	nvcc -o $@ -g $< $(CUFLAGS)

example3: example3.cu
	nvcc -o $@ -g $< $(CUFLAGS)

mean: mean.cu
	nvcc -o $@ -g $< $(CUFLAGS)

matmul: matmul.cu
	nvcc -o $@ -g $< $(CUFLAGS) -lcublas

clean:
	rm -f example1 example2 example3 mean matmul
