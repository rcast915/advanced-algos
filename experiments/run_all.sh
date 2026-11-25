#!/bin/bash
# Full Tasks 5-6 experiment pipeline. Long-running: on the cluster, submit it as a batch job, not on a login node.
# Needs bin/solver_cpu (run `make cpu` from the repo root first). Honors DARP_SOLVER and DARP_RESULTS (see paths.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DARP_RESULTS:-$HERE/../results/task56}"
mkdir -p "$OUT/logs"
for step in main scaling threads graphs sweeps; do
  python3 -I "$HERE/run_experiments.py" "$step" > "$OUT/logs/log_$step.txt" 2>&1
  echo "$step done $(date -u +%H:%M:%S)" >> "$OUT/logs/progress.txt"
done
python3 -I "$HERE/analyze.py" > "$OUT/logs/log_analyze.txt" 2>&1
python3 -I "$HERE/tables.py" > "$OUT/logs/log_tables.txt" 2>&1
echo ALLDONE >> "$OUT/logs/progress.txt"
