# Dial-a-Ride (DARP): Heuristics, Exact Solver, Online Variant, GPU Add-on

Team 6 – Advanced Algorithms (Alexander Karl, Jenna Murphree, Ray Castaneda)

One project, built in three stages on the same formal model (v2 from Task 2: exactly-once service, same vehicle for pickup and dropoff,
precedence, timing, time windows, return by horizon H, maximum ride time, capacity — constraints C1–C8):

| Stage | Task | What | Where |
|---|---|---|---|
| 1 | 3–4 | Python prototype: insertion + relocate heuristic, validated against an exact MIP (HiGHS) | `prototype/` |
| 2 | 5–6 | Same algorithm in C++/OpenMP on road-network graphs, plus an online (dynamic) mode; six instance families, sweeps, pre-registered predictions | `solver/`, `experiments/` |
| 3 | optional | CUDA offload of candidate-insertion evaluation (H100) | `solver/gpu*` |

## Layout

```
Makefile                 builds bin/solver_cpu | bin/solver_stub | bin/solver_gpu
solver/
  solver.cpp             C++ solver: static, static_nols, dynamic modes (GPU hook behind #ifdef USE_CUDA)
  gpu.h gpu.cu           CUDA candidate evaluator (shared host/device core in gpu.h)
  gpu_stub.cpp           serial CPU emulation of the GPU path, for testing without a GPU
prototype/               Task 3 (pure Python + scipy)
  darp.py                instance generator, validator, insertion heuristic, relocate local search
  exact.py               exact 3-index MIP via scipy.optimize.milp (HiGHS)
  experiments.py         small (vs exact) and large (heuristic only) runs
experiments/             Tasks 5-6 pipeline (drives the C++ solver)
  paths.py               all repo paths in one place (env overrides: DARP_SOLVER, DARP_RESULTS)
  gen.py                 graphs, the six instance families, instance file writer
  validate.py            independent validator + metrics; run() calls the solver binary
  run_experiments.py     main | scaling | threads | graphs | sweeps
  analyze.py tables.py   figures, fitted exponents, markdown tables
  parity.py bench.py     CPU vs GPU/stub parity check, GPU timing
  run_all.sh             every experiment step, then analysis
results/
  task3/                 results_small.csv, results_large.csv, env.txt
  task56/                results_*.csv, predictions (+ sha256, timestamp), prediction_vs_actual.csv, figs/, tables/
docker/Dockerfile        toolchain image (CUDA 13 devel + g++ + Python stack)
submit_punakha.sh        sbatch wrapper: build image, run DARP_JOB inside it on an H100
jobs/                    check.sh (cpu+stub parity), gpu.sh (CUDA build + parity), bench.sh (timing)
```

## Quick start

Requirements: Python 3 with numpy, scipy, pandas, matplotlib; g++ with OpenMP; nvcc only for the GPU build.
Run everything from the repo root.

```bash
make cpu                                         # -> bin/solver_cpu
make stub                                        # -> bin/solver_stub (GPU logic, emulated on the CPU)

# Stage 1 (Task 3)
python3 prototype/experiments.py small           # heuristic vs exact, n = 4-10 (minutes; 120 s MIP limit per instance)
python3 prototype/experiments.py large           # heuristic only, n = 10-100

# Stage 2 (Tasks 5-6)
python3 experiments/run_experiments.py main      # or scaling | threads | graphs | sweeps
python3 experiments/analyze.py                   # figures -> results/task56/figs
python3 experiments/tables.py                    # tables  -> results/task56/tables
experiments/run_all.sh                           # all of the above, logs in results/task56/logs

# Stage 3 (GPU) — see "GPU solver" below
python3 experiments/parity.py bin/solver_cpu bin/solver_stub DARP_GPU=1 DARP_GPU_MIN=1
```

The experiment scripts write into `results/` and **overwrite the committed results**. For a re-run, point them elsewhere:
`DARP_RESULTS=/path/to/scratch python3 experiments/run_experiments.py main`. To use a different solver binary, set `DARP_SOLVER`.

**On punakha (H100, Slurm + rootless Docker):** nothing runs on the login node. `submit_punakha.sh` builds the image in
`docker/Dockerfile` (CUDA 13, g++/OpenMP, Python stack) and runs one script from `jobs/` inside it, with the repo mounted at `/work`:

```bash
mkdir -p logs/output logs/error
sbatch submit_punakha.sh                          # jobs/check.sh: build cpu + stub, parity at n = 100
DARP_JOB=jobs/gpu.sh sbatch submit_punakha.sh     # build CUDA solver, parity vs CPU at n = 100, 500
DARP_JOB=jobs/bench.sh sbatch --time=2:00:00 submit_punakha.sh
```

`prototype/experiments.py small`, the `scaling`, `threads` and `sweeps` steps, `run_all.sh` and `bench.py` are long-running;
give them their own job with a suitable `--time`.

Using the prototype directly:

```python
import sys; sys.path.insert(0, 'prototype'); import darp
I = darp.make_instance(n=20, seed=1)
out = darp.solve_heuristic(I)          # dict: routes, T, Z, parts, Z_construct, ls_iters (or None on failure)
ok, violations = darp.validate(I, out["routes"], out["T"])
```

## The algorithm (shared by both implementations)

**Objective.** `Z = 1.0·(total travel time) + 0.5·(total wait) + 0.5·(total excess ride time) + 1.0·(total wait beyond 10 min)`.
Wait is measured from the start of each pickup window.

**Insertion construction.** Requests are taken in order of earliest pickup. For each request, every vehicle and every
(pickup, dropoff) position pair is tried; the cheapest feasible placement is kept.

**Feasibility check.** A route is scheduled forward with every stop served as early as allowed. One pass checks time windows,
capacity, ride-time limits and the return deadline, and computes the route's cost.

**Local search (relocate).** Remove one request and reinsert it at its best position in any route; accept only strict cost
decreases; stop when nothing improves (or at the iteration/pass cap).

**Validators.** `prototype/darp.py:validate` and `experiments/validate.py` re-check finished schedules against C1–C8 (and, for the
online mode, release times and commitments) without using the solver's internal bookkeeping.

### Differences between the stages

| | Prototype (`prototype/`) | C++ solver (`solver/`) |
|---|---|---|
| Geography | Euclidean 5×5 km plane, 0.5 km/min | road graphs (grid, 30×30 grid, ring, clusters, geometric) with all-pairs shortest paths |
| Unplaceable request | whole heuristic returns `None` | request rejected; quality metric becomes **served %**; local search also retries rejected requests |
| Modes | static | `static`, `static_nols` (insertion only), `dynamic` (online) |
| Parallelism | none | OpenMP over vehicles; optional CUDA over (vehicle, a, b) |

**Dynamic mode.** Requests are revealed at release time (window start minus a lead time) and inserted immediately, without local search.
At each decision time, stops already served are frozen, plus one pending stop per vehicle (the vehicle is committed to its next stop).

Solver usage: `bin/solver_cpu instance.txt MODE [threads] [ls_passes]`; it prints one JSON line (routes, times, served flags,
runtime, step/eval counters, peak memory). Instance files are written by `experiments/gen.py:write_instance`.

## GPU solver

`make gpu` builds `bin/solver_gpu` (nvcc, `CUDA_ARCH=sm_90` for the H100 by default). Enable the GPU at run time with `DARP_GPU=1`.
Results are bit-identical to the CPU solver (same routes, served set and Z); this needs `-ffp-contract=off` (g++) and
`-fmad=false` (nvcc), which the Makefile sets. `experiments/parity.py` checks it; `experiments/profile_gpu.py` (with `DARP_PROF=1`)
shows where the time goes; `experiments/bench.py` times CPU vs GPU.

How it works: one thread block per candidate route; phase 1 tests every pickup position in O(1) from precomputed route states,
phase 2 evaluates (pickup, delivery) pairs only for feasible pickups and only up to the first stop that the new passenger makes
infeasible. In static mode the routes stay on the GPU (only changed routes are re-sent) and local search evaluates batches of
requests per call, re-evaluating the rest of a batch after any accepted move so the decisions match the sequential loop.

Run-time switches: `DARP_GPU_MIN` (fewest candidate routes worth a GPU call, default 32), `DARP_GPU_RESIDENT=0` (re-send all routes
every call), `DARP_GPU_BATCH` (max local-search batch, default 128; 1 = no batching). Routes over 126 stops fall back to the CPU.
Benchmarks: `results/gpu/` (H100; at n = 10000, 5.26 s on the GPU vs 183 s on one CPU thread).

## Stage 1 results (Task 3)

**Instances.** n requests on a 5×5 km grid, depot at the center; 15-minute pickup windows; 1-minute service; maximum ride 2× direct
(direct trips ≥ 3 min); capacity 4; horizon 480 min; fleet ceil(n/4) (min 2). Same seed → same instance.

| | Small set | Large set |
|---|---|---|
| Requests (n) | 4, 6, 8, 10 | 10, 20, 40, 60, 80, 100 |
| Instances per size | 30, 30, 20, 10 (seeds 1…) | 10 (seeds 1001–1010) |
| Methods | insertion, + local search, exact | insertion, + local search |
| Exact time limit | 120 s | n/a |

Small instances. Gap = (heuristic − exact) / exact.

| Requests | Runs | Optimal | Mean gap, insertion | Mean gap, + LS | Worst gap | Mean time, heuristic | Mean time, exact |
|---|---|---|---|---|---|---|---|
| 4 | 30 | 30 | 3.0% | 0.8% | 15.1% | 0.001 s | 0.06 s |
| 6 | 30 | 30 | 6.2% | 1.9% | 13.4% | 0.004 s | 0.18 s |
| 8 | 20 | 20 | 6.5% | 1.2% | 5.9% | 0.010 s | 0.8 s |
| 10 | 10 | 10 | 10.1% | 2.9% | 6.4% | 0.022 s | 6.5 s |

- All 90 heuristic and all 90 exact schedules passed the validator; the heuristic never beat the optimum.
- Local search brings the mean gap to about 1–3%; single instances can be up to about 15% worse.

Larger instances.

| Requests | Feasible and valid | LS improvement over insertion | Mean time, insertion | Mean time, + LS |
|---|---|---|---|---|
| 10 | 10/10 | 3.0% | 0.006 s | 0.03 s |
| 20 | 10/10 | 5.6% | 0.03 s | 0.19 s |
| 40 | 10/10 | 5.1% | 0.16 s | 1.4 s |
| 60 | 10/10 | 5.4% | 0.44 s | 4.4 s |
| 80 | 10/10 | 6.7% | 1.0 s | 12.9 s |
| 100 | 10/10 | 6.0% | 1.9 s | 26.1 s |

Runtime grows polynomially (about n^2.5 to n^3 in unoptimized Python). There is no optimum at these sizes, so these rows show scaling
and feasibility, not distance from optimal.

## Stage 2 experiments (Tasks 5–6)

Common setup: 12×12 grid road network (144 nodes, 2 min per edge) unless stated, depot at the most central node, H = 480, Q = 4,
service 1 min, pickups in [10, 360].

| Family | Design | Stresses |
|---|---|---|
| F1 Baseline | grid, n/m = 10, 15-min windows, ρ = 2, lead 30 | — |
| F2 Tight windows | as F1, 4-min windows | window width |
| F3 Scarce fleet | as F1, n/m = 25 | fleet vs demand |
| F4 Sparse ring | 100-node ring with chords | sparse, high-diameter network |
| F5 Adversary A | 30×30 grid, identical-O/D pairs that can only pool if the first pickup is delayed | earliest-start scheduling |
| F6 Adversary B | long decoy trips occupy every vehicle just before a burst | online commitments |

Experiment steps (`run_experiments.py`): `main` (6 families × n ∈ {20, 100, 500} × 3 modes), `scaling` (F1, n up to 5000),
`threads` (1 vs 2 OpenMP threads), `graphs` (five network types), `sweeps` (fleet ratio × window width).
Hypotheses were written and hashed before the experiment seeds ran (`results/task56/predictions.md`, `.sha256`, `.timestamp`;
check with `cd results/task56 && sha256sum -c predictions.sha256`). `results_f5_control.csv` is a post-hoc control added after the main run.
Results, figures and their discussion are in the report.

## Correctness and complexity notes

- **Valid output.** A request is inserted only if the forward-scheduled route passes every constraint, and local search accepts only
  feasible moves, so every returned schedule satisfies C1–C8; the validators confirm this empirically.
- **Termination.** Insertion handles each request once; local search accepts only strict decreases over a finite set (plus a cap).
- **Not optimal, not complete.** It can return a worse-than-optimal schedule and can fail to serve requests on a feasible instance.
- **Cost.** One candidate route evaluation is O(L) for L stops; trying all vehicle/position pairs for one request is about O(K·L³) in the
  both implementations (the C++ solver reuses the prefix state and stops early on infeasibility, but the suffix is still re-scanned per
  (a, b) pair, so it remains O(L³) per route in the worst case). Forward time slack would cut the per-candidate check to O(1) (not implemented).

## Limitations

- Instance parameters and objective weights (1, 0.5, 0.5, 1) are modeling choices.
- Earliest-start scheduling never delays a stop on purpose, while the exact solver picks times freely; part of the gap comes from this
  (F5 is built to exploit it).
- The optimum is known only up to 10 requests; the 1–3% figure should not be extended to larger instances.
- `exact.py` preallocates a 60,000-row constraint matrix, so it will fail (not just slow down) well beyond n ≈ 10.
- GPU solver: bit-identical to the CPU solver on an H100; about 35x faster than one CPU thread at n = 10000 (static mode).
