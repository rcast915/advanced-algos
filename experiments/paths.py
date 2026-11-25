"""Single source of truth for repository paths used by the Tasks 5-6 scripts.

Override with environment variables:
  DARP_SOLVER   solver binary to call (default: <repo>/bin/solver_cpu, built by `make cpu`)
  DARP_RESULTS  directory for CSVs, figures and tables (default: <repo>/results/task56).
                Point this somewhere else for a re-run so the committed results are not overwritten.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / 'bin'
SOLVER = Path(os.environ.get('DARP_SOLVER', BIN / 'solver_cpu'))
RESULTS = Path(os.environ.get('DARP_RESULTS', ROOT / 'results' / 'task56'))
FIGS = RESULTS / 'figs'
TABLES = RESULTS / 'tables'
