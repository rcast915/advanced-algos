# Optional H100 add-on for the DARP static solver

What runs on the GPU: for each request, every candidate insertion (vehicle, pickup position, delivery position) is evaluated in parallel
(one thread block per route, threads stride over position pairs; block-level min reduction with ties broken by lowest (a, b), the CPU scan order).
Everything else (ordering, local-search control, route updates) stays on the CPU. The report's results do not depend on this add-on.

Files: `solver/gpu.h` (shared evaluation core, host+device), `solver/gpu.cu` (kernel + launch), `solver/gpu_stub.cpp` (serial emulation for testing without a GPU),
`solver/solver.cpp` (the solver; GPU code sits behind `#ifdef USE_CUDA`, so the CPU build is unchanged), the root `Makefile`, `experiments/parity.py`, `experiments/bench.py`.

## Steps on the H100 node (run from the repo root)
1. `make cpu stub gpu`            (set CUDA_HOME if nvcc is elsewhere; `CUDA_ARCH=sm_90` is the default, H100)
2. `python3 experiments/parity.py bin/solver_cpu bin/solver_gpu DARP_GPU=1 DARP_GPU_MIN=1`   must end with PARITY OK (same routes, same served set, same Z to 1e-9, valid)
3. `python3 experiments/bench.py` timing table for n = 500 ... 10000

Without a GPU, step 2 can be run against `bin/solver_stub` to test everything except the CUDA launch itself.

## Status
* Tested on an H100 80GB HBM3 (punakha, driver 595.71.05, CUDA 13.0, job 66729): `gpu.cu` built unchanged; PARITY OK on 16 instances
  (F1, F2, F5, F6 at n = 100, 500; static and dynamic) under the route-level check: identical routes, served sets and Z, all valid.
  `-fmad=false` is required for exact parity.
* Baseline timing (job 66730, `results/gpu/bench_h100_66730.csv`, F1 seed 1, static, wall time incl. CUDA init):
  GPU / 1-thread CPU speedup 0.13x (n = 500), 0.31x (2000), 0.58x (5000), 0.72x (10000). Two CPU threads beat the GPU at every size.
* Step 1a (two-phase kernel: O(1) pickup test per position from precomputed route states, then (a, b) pairs only for feasible a)
  and step 1b (routes resident on the GPU in fixed slots; only changed routes re-sent, about 1 per call): PARITY OK again
  (stub and H100, same 16 instances). `results/gpu/bench_h100_step1b.csv` (interactive session, 2 CPUs):
  GPU / 1-thread CPU 0.14x (n = 500), 1.42x (2000), 4.58x (5000), 8.64x (10000). Wall times include about 0.6-0.7 s of CUDA init.
  Profile (`experiments/profile_gpu.py`, n = 5000): kernel 66 %, copies 29 %, host prep 2 % of the GPU search time.
* Step 1c (exact delivery-range pruning: for each feasible pickup, delivery positions past the first replay failure are skipped):
  38x fewer (a, b) pairs; `results/gpu/bench_h100_step1c.csv`: 9.84x at n = 10000.
* Step 2 (batched local search): the next K requests are evaluated against the current solution in one GPU call and decided in
  order; at the first accepted move the rest of the batch is discarded and re-evaluated. Same decisions as the sequential loop, so
  results stay bit-identical (PARITY OK, stub and H100). K adapts between 4 and `DARP_GPU_BATCH` (default 128; 1 disables batching).
  `results/gpu/bench_h100_step2.csv`: GPU / 1-thread CPU 0.15x (n = 500), 2.74x (2000), 15.72x (5000), 34.84x (10000);
  vs 2 CPU threads at n = 10000: 22.9x. GPU wall time at n = 10000: 254 s (baseline) -> 5.26 s.
* Before steps 1a/1b the GPU evaluated each (a, b) from scratch (no prefix reuse), about L times more arithmetic than the CPU.
* Routes longer than 126 stops fall back to the CPU. Step counters do not include GPU work.
