#!/usr/bin/env bash
# Light sanity check (seconds of CPU, no GPU work): build the CPU and stub solvers, then compare them on small instances.
set -euo pipefail
make cpu stub
PARITY_N="${PARITY_N:-100}" python3 experiments/parity.py bin/solver_cpu bin/solver_stub DARP_GPU=1 DARP_GPU_MIN=1
