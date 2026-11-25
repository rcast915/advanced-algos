#!/usr/bin/env bash
# CPU vs GPU timing, n = 500 ... 10000. Long: submit with a larger --time. Run only after jobs/gpu.sh reports PARITY OK.
set -euo pipefail
make cpu gpu
python3 experiments/bench.py
