#!/usr/bin/env bash
# Build the CUDA solver (first real compile of solver/gpu.cu) and check it against the CPU solver.
set -euo pipefail
nvcc --version | tail -2
make cpu gpu
PARITY_N="${PARITY_N:-100,500}" python3 experiments/parity.py bin/solver_cpu bin/solver_gpu DARP_GPU=1 DARP_GPU_MIN=1
