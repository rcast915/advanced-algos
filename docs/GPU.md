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

## Status (be honest in the report if you cite it)
* Tested here: the host marshalling and the evaluation logic, through gpu_stub.cpp, which calls the same gpu_eval_one() the kernel uses: PARITY OK on 16 instances (under the earlier, weaker check: served % and Z only; re-run with the current route-level check) (F1, F2, F5, F6 at n = 100, 500; static and dynamic).
* NOT tested: gpu.cu itself (the launch, memory copies, block reduction) because no GPU or nvcc was available. Expect to fix small build errors. `-fmad=false` is required for exact parity.
* The GPU evaluates each (a, b) from scratch (no prefix reuse), so it does about L times more arithmetic than the CPU code; expect a gain only when there are many candidate routes (n of a few thousand and up). Small instances will be slower on the GPU because each call pays a transfer and launch cost; DARP_GPU_MIN keeps small calls on the CPU.
* Routes longer than 126 stops fall back to the CPU. Step counters do not include GPU work.
