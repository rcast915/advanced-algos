# CPU only:           make cpu      -> ./solver_cpu
# CPU emulation test: make stub     -> ./solver_stub   (runs the GPU algorithm serially; needs no GPU)
# H100:               make gpu      -> ./solver_gpu    (set CUDA_HOME if nvcc is not on PATH; sm_90 = H100)
CXXFLAGS = -O2 -std=c++17 -fopenmp -ffp-contract=off
cpu: solver.cpp ; g++ $(CXXFLAGS) -o solver_cpu solver.cpp
stub: solver.cpp gpu.h gpu_stub.cpp ; g++ $(CXXFLAGS) -DUSE_CUDA -o solver_stub solver.cpp gpu_stub.cpp
gpu: solver.cpp gpu.h gpu.cu
	nvcc -O2 -std=c++17 -arch=sm_90 -fmad=false -c gpu.cu -o gpu.o
	g++ $(CXXFLAGS) -DUSE_CUDA -c solver.cpp -o solver_g.o
	g++ -fopenmp -o solver_gpu solver_g.o gpu.o -L$${CUDA_HOME:-/usr/local/cuda}/lib64 -lcudart
