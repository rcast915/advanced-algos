# Static Dial-a-Ride (DARP): Heuristic vs. Exact Solver

Team 6 – Advanced Algorithms (Alexander Karl, Jenna Murphree, Ray Castaneda)

This folder holds the code and results for Task 3 (a heuristic for the static DARP, benchmarked against an exact solver) and the experiment data behind Tasks 3–4. It implements the v2 formal model from our Task 2 description: exactly-once service, same vehicle for pickup and dropoff, precedence, timing, time windows, return by horizon H, maximum ride time, and capacity (C1–C8).

## Files

| File | What it is |
|---|---|
| `darp.py` | Instance generator, independent validator, insertion heuristic, relocate local search |
| `exact.py` | Exact 3-index mixed-integer program solved with HiGHS (via `scipy.optimize.milp`) |
| `experiments.py` | Runs both experiment sets and writes the CSVs |
| `results_small.csv` | 90 runs, n = 4–10 requests: heuristic vs. exact |
| `results_large.csv` | 60 runs, n = 10–100 requests: heuristic only |
| `env.txt` | Python, numpy, scipy versions and platform used |
| `darp_small_results.xlsx` | Summary table (formulas) and all small-instance runs |

## Quick start

Requires Python 3, numpy, scipy and pandas (pandas only for summarizing). Keep the three `.py` files in one folder.

```
python experiments.py small    # heuristic vs. exact, writes results_small.csv (a few minutes)
python experiments.py large    # heuristic only, writes results_large.csv (several minutes)
```

`darp.py` uses only the standard library:

```python
import darp
I = darp.make_instance(n=20, seed=1)
out = darp.solve_heuristic(I)          # dict: routes, T, Z, parts, Z_construct, ls_iters (or None on failure)
ok, violations = darp.validate(I, out["routes"], out["T"])
```

## How each part works

### `darp.py`

**Instances.** `make_instance(n, seed)` places n requests on a 5×5 km grid with the depot at the center. Travel time is Euclidean distance at 0.5 km/min. Each request has a 15-minute pickup window starting at a random time, a delivery window derived from it, a 1-minute service time, and a maximum ride time of 2× the direct trip (direct trips are at least 3 minutes). Vehicles have capacity 4, the horizon is 480 minutes, and the fleet is about n/4 vehicles. The same seed always gives the same instance.

**Objective.** `Z = 1.0·(total travel time) + 0.5·(total wait) + 0.5·(total excess ride time) + 1.0·(total wait beyond 10 min)`. Wait is measured from the start of each pickup window.

**Insertion construction.** Requests are taken in order of earliest pickup. For each request, every vehicle and every position pair (pickup, dropoff) is tried. The cheapest placement that passes the feasibility check is kept. If no placement exists for some request, the heuristic returns `None` (it is sound but not complete).

**Feasibility check.** A route is scheduled forward with every stop served as early as allowed. This one pass checks time windows, capacity, ride-time limits and the return deadline, and computes the route's cost.

**Local search (relocate).** Remove one request from its route and reinsert it at the best position in any route. Accept the move only if total cost strictly decreases. Stop when no request can be improved (or at the iteration cap).

**Validator.** `validate` re-checks a finished schedule against C1–C8 and recomputes the objective without using the heuristic's internal bookkeeping. Both the heuristic's and the exact solver's output are passed through it.

### `exact.py`

`solve_exact(I, time_limit)` builds the standard 3-index model (one arc variable per vehicle and pair of stops, plus time, load, wait, excess and long-wait variables) and solves it with HiGHS. Unlike the heuristic, it chooses all stop times freely. It returns the optimal cost, or the best solution and a dual bound if the time limit hits. Practical up to about 10 requests.

### `experiments.py`

Runs the experiments below and records, per instance: cost of insertion alone, cost after local search, exact cost, solver status, runtimes, and validator results.

## Experimental setup

| | Small set | Large set |
|---|---|---|
| Requests (n) | 4, 6, 8, 10 | 10, 20, 40, 60, 80, 100 |
| Instances per size | 30, 30, 20, 10 (seeds 1…) | 10 (seeds 1001–1010) |
| Methods | insertion, insertion + local search, exact | insertion, insertion + local search |
| Exact time limit | 120 s | n/a |
| Fleet | ceil(n/4) vehicles (min 2) | same |

Environment details are in `env.txt`. Runs are single-threaded and deterministic given the seed; runtimes will vary by machine.

## What the results show

### Small instances: quality against the true optimum

Gap = (heuristic cost − exact cost) / exact cost.

| Requests | Runs | Solved to optimality | Mean gap, insertion only | Mean gap, + local search | Worst gap | Mean time, heuristic | Mean time, exact |
|---|---|---|---|---|---|---|---|
| 4 | 30 | 30 | 3.0% | 0.8% | 15.1% | 0.001 s | 0.06 s |
| 6 | 30 | 30 | 6.2% | 1.9% | 13.4% | 0.004 s | 0.18 s |
| 8 | 20 | 20 | 6.5% | 1.2% | 5.9% | 0.010 s | 0.8 s |
| 10 | 10 | 10 | 10.1% | 2.9% | 6.4% | 0.022 s | 6.5 s |

- All 90 heuristic schedules and all 90 exact schedules passed the validator.
- The heuristic never beat the exact optimum (a sanity check on both).
- Local search cuts the average gap to about 1–3%, but single instances can be much worse (worst case about 15%).
- Exact solve time grows roughly 8× from 8 to 10 requests; the heuristic stays in milliseconds.

### Larger instances: feasibility and scaling

| Requests | Feasible and valid | Local search improvement over insertion only | Mean time, insertion only | Mean time, + local search |
|---|---|---|---|---|
| 10 | 10/10 | 3.0% | 0.006 s | 0.03 s |
| 20 | 10/10 | 5.6% | 0.03 s | 0.19 s |
| 40 | 10/10 | 5.1% | 0.16 s | 1.4 s |
| 60 | 10/10 | 5.4% | 0.44 s | 4.4 s |
| 80 | 10/10 | 6.7% | 1.0 s | 12.9 s |
| 100 | 10/10 | 6.0% | 1.9 s | 26.1 s |

- Every schedule was valid.
- Local search improves on insertion by about 5–7% from n = 20 up, and accounts for most of the runtime.
- Runtime grows polynomially (roughly n^2.5 to n^3 in this unoptimized Python), not exponentially.
- There is no optimal value at these sizes, so these rows show scaling and feasibility, not distance from optimal.

## Correctness and complexity notes (for Task 4)

- **Valid output.** A request is inserted only if the forward-scheduled route passes every constraint, and local search accepts only feasible moves, so any returned schedule satisfies C1–C8. The validator confirms this empirically.
- **Termination.** Insertion handles each request once. Local search accepts only strict cost decreases over a finite set of schedules, so it cannot cycle (an iteration cap is also in place).
- **Not optimal, not complete.** It can return a worse-than-optimal schedule, and it can fail to find any schedule on a feasible instance.
- **Cost.** Evaluating one candidate route is O(L) for a route of L stops. Trying all vehicle and position pairs for one request is about O(K·L³) with this direct implementation. Total insertion cost is therefore about O(n·K·L³) and each local search pass repeats a similar search for all n requests. Forward time slack would cut the per-candidate check to O(1) (not implemented).

## Limitations and assumptions

- **Instance parameters are a modeling choice.** Grid size, window width, ride factor, capacity, fleet size and the objective weights (1, 0.5, 0.5, 1) were chosen for these experiments and should be confirmed yall.
- **Earliest-start scheduling.** The heuristic never delays a stop on purpose, while the exact solver picks times freely. Part of the measured gap comes from this, not only from routing.
- **Small exact range.** The optimum is known only up to 10 requests, so the 1–3% figure should not be extended to larger instances.
- **Runtime.** Times come from unoptimized Python on one machine and show trends, not benchmarks.
- **Not covered here.** The dynamic (online) problem and the written report and slides are not part of this folder.
