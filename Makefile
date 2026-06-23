CUDA_ARCH=52

CUFLAGS=-O2 -arch sm_$(CUDA_ARCH) -DCUDA_ARCH="$(CUDA_ARCH)" --default-stream per-thread

all: example1 example2 example3 benchmark mean matmul

example1: example1.cu alloc.h check.h
	nvcc -o $@ -g $< $(CUFLAGS)

example2: example2.cu alloc.h check.h
	nvcc -o $@ -g $< $(CUFLAGS)

example3: example3.cu alloc.h check.h
	nvcc -o $@ -g $< $(CUFLAGS)

benchmark: benchmark.cu alloc.h check.h check_host.h timer.h
	nvcc -o $@ -g $< $(CUFLAGS)

mean: mean.cu alloc.h check.h
	nvcc -o $@ -g $< $(CUFLAGS)

matmul: matmul.cpp alloc.h check.h check_blas.h
	c++ -o $@ -g $< -DCUDA_ARCH="$(CUDA_ARCH)" -lcublas -lcuda -lcudart

clean:
	rm -f example1 example2 example3 benchmark mean matmul
