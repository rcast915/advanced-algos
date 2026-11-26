# Pre-registered hypotheses and predictions (written BEFORE the experiment seeds were run)

## What had and had not been run when this was written
- The C++ solver was validated against the earlier Python heuristic (same Z to 6 digits, 5 instances).
- Families F5 and F6 (the two adversarial families) were debugged on throwaway seeds (97, 98) until they behaved as designed; that is the only information about them that informs the predictions below (served % at n = 20, 100, 500 on seed 98).
- F1 was run once at n = 12 (seed 99) and n = 500 (seed 99) as a speed test; both served 100 % in all modes.
- F2, F3, F4 have never been run. Their predictions come from reasoning only.
- The experiment seeds (1..5 for n = 20, 100; 1..3 for n = 500) have NOT been run.

## Common setup
12 x 12 grid road network (144 nodes, 264 edges, 2 min per edge) unless stated; depot at the most central node; horizon H = 480 min; capacity Q = 4;
service time 1 min; pickup arrivals in [10, 360]. Static = insertion + relocate local search. Dynamic = online insertion, no local search, requests
revealed at release = window start minus lead time, one pending commitment per vehicle.
n in {20, 100, 500}. "Served %" = requests served / requests.

## Families, what each stresses, hypothesis

| Family | Design | Stresses | Hypothesis (stated in advance) |
|---|---|---|---|
| F1 Baseline | grid, n/m = 10, windows 15 min, rho = 2, lead 30 | nothing in particular | Nearly everything is served in both modes; dynamic is at most a few points worse and costs somewhat more per request. |
| F2 Tight windows | as F1 but windows 4 min | window width | Tight windows make insertion reject requests: served % drops 10-25 points versus F1, dynamic more than static. |
| F3 Scarce fleet | as F1 but n/m = 25 | fleet size versus demand | Served % is capped by capacity (about m * H / work per request, plus pooling) so it falls to roughly half; local search cannot fix this; dynamic is lower than static. |
| F4 Sparse ring | ring of 100 nodes (avg degree 2.1, long trips), n/m = 10, windows 15 | sparse high-diameter structure, long rides | Longer trips use up vehicle time, so served % is below F1 and the mean excess ride time is higher; pooling is easier on a ring than on a grid. |
| F5 Adversary A (AI-designed, aimed at the static heuristic) | 30 x 30 grid, requests come in identical-origin/destination pairs; partner window opens 2 min after the first window closes; rho = 1.1; four waves; n/m = 8 | the earliest-start assumption | The only way to serve both members of a pair on one vehicle is to delay the first pickup to the end of its window, which earliest-start scheduling never does. Because n/m = 8, about half of the requests need pooling, so served % is about 60-75 % while a solver allowed to delay would serve all. Dynamic is no better. |
| F6 Adversary B (AI-designed, aimed at the online algorithm) | m long "decoy" requests known at t = 0 whose windows open at about 140 and stay open 120 min, then 4m clustered requests with 8-min windows revealed 1 min ahead at 150-156, plus background requests | irrevocable commitments | The online algorithm has every vehicle started on a decoy trip when the burst arrives and cannot divert them, so it rejects most of the burst; the static algorithm can insert burst requests before each decoy (decoys have slack). Static serves clearly more than dynamic (about 15-20 points). |

## Predictions: served %, point estimate (plausible range)

| Family | n | Static | Dynamic |
|---|---|---|---|
| F1 | 20 | 93 (80-100) | 88 (70-100) |
| F1 | 100 | 99 (93-100) | 97 (88-100) |
| F1 | 500 | 100 (97-100) | 99 (94-100) |
| F2 | 20 | 75 (55-90) | 65 (45-85) |
| F2 | 100 | 80 (65-92) | 72 (55-88) |
| F2 | 500 | 85 (70-95) | 78 (60-90) |
| F3 | 20 | 40 (20-60) | 35 (15-55) |
| F3 | 100 | 52 (38-65) | 45 (30-60) |
| F3 | 500 | 58 (45-72) | 52 (38-65) |
| F4 | 20 | 80 (60-95) | 70 (50-90) |
| F4 | 100 | 85 (70-95) | 78 (60-90) |
| F4 | 500 | 90 (75-98) | 84 (65-95) |
| F5 | 20 | 62 (50-80) | 60 (45-75) |
| F5 | 100 | 65 (55-80) | 60 (48-75) |
| F5 | 500 | 72 (60-85) | 68 (55-80) |
| F6 | 20 | 65 (50-80) | 45 (30-60) |
| F6 | 100 | 80 (68-92) | 64 (50-76) |
| F6 | 500 | 87 (75-95) | 67 (55-78) |

## Other predictions (all families)
- Dynamic never serves more than about 3 points above static on any family and size (static knows the future and also runs local search).
- Mean waiting time (pickup time minus window start) stays below 5 minutes everywhere (stops are served as early as allowed); F2 is the lowest, F3 and F5 the highest.
- Mean ride time exceeds the mean direct trip by at most 25 % on average (ratio of mean ride to mean direct below 1.25), highest on F4.
- Runtime: static at n = 500 is under 2 s on one core for every family; dynamic at n = 500 is under 0.2 s. Static runtime grows between n^1.5 and n^2.5; dynamic between n^1.1 and n^2.
- Peak memory is dominated by the distance matrix: under 30 MB for the 144-node graphs and under 80 MB for the 30 x 30 grid.
- Two OpenMP threads will not speed up the solver at n = 500 (the work per request is too small compared with thread start-up cost).
- The property that most affects quality is the ratio of fleet to demand (n/m) and the window width, not the graph's diameter or degree. The property that most affects running time is the number of stops per route, L, which grows as the number of requests per vehicle, n/m, grows.
