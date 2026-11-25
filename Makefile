# Builds the DARP solver into bin/.
#   make cpu    -> bin/solver_cpu    CPU + OpenMP (what the experiments use)
#   make stub   -> bin/solver_stub   GPU code path emulated serially on the CPU (tests the GPU logic without a GPU)
#   make gpu    -> bin/solver_gpu    CUDA build for H100 (sm_90); needs nvcc. Set CUDA_HOME if it is not /usr/local/cuda.
#   make clean
# -ffp-contract=off (and -fmad=false for nvcc) keep CPU and GPU arithmetic bit-identical.
CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS  = -O2 -std=c++17 -fopenmp -ffp-contract=off
CUDA_ARCH ?= sm_90
SRC = solver/solver.cpp
HDR = solver/gpu.h

.PHONY: all cpu stub gpu clean
all: cpu stub
cpu: bin/solver_cpu
stub: bin/solver_stub
gpu: bin/solver_gpu

bin:
	mkdir -p bin

bin/solver_cpu: $(SRC) | bin
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

bin/solver_stub: $(SRC) $(HDR) solver/gpu_stub.cpp | bin
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -o $@ $(SRC) solver/gpu_stub.cpp

bin/solver_gpu: $(SRC) $(HDR) solver/gpu.cu | bin
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -fmad=false -c solver/gpu.cu -o bin/gpu.o
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -c $(SRC) -o bin/solver_g.o
	$(CXX) -fopenmp -o $@ bin/solver_g.o bin/gpu.o -L$${CUDA_HOME:-/usr/local/cuda}/lib64 -lcudart

clean:
	rm -rf bin
