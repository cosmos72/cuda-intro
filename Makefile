CUDA_ARCH=52

CXXFLAGS=-g -O2 -DCUDA_ARCH=$(CUDA_ARCH)

CUFLAGS=$(CXXFLAGS) -arch sm_$(CUDA_ARCH) --default-stream per-thread

all: add1 add2 add3 add mean matmul

add1: add1.cu alloc.h check_cuda.h
	nvcc -o $@ $< $(CUFLAGS)

add2: add2.cu alloc.h check_cuda.h
	nvcc -o $@ $< $(CUFLAGS)

add3: add3.cu alloc.h check_cuda.h
	nvcc -o $@ $< $(CUFLAGS)

###############################################################

add_cuda.o: add_cuda.cu alloc.h check_cuda.h timer.h
	nvcc -o $@ -c $< $(CUFLAGS)

add_host.o: add_host.cpp alloc.h check_host.h timer.h
	c++ -o $@ -c $< $(CXXFLAGS) -std=c++20

add.o: add.cpp check_cuda.h
	c++ -o $@ -c $< $(CXXFLAGS)

add: add_cuda.o add_host.o add.o
	c++ -o $@ $^ $(CXXFLAGS) -lcuda -lcudart

###############################################################

mean: mean.cu alloc.h check_cuda.h
	nvcc -o $@ $< $(CUFLAGS)

matmul: matmul.cpp alloc.h check_cuda.h check_cublas.h
	c++ -o $@ $< $(CXXFLAGS) -lcublas -lcuda -lcudart

clean:
	rm -f add1 add2 add3 add mean matmul
