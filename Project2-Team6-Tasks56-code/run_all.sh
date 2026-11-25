#!/bin/bash
cd /home/claude/darp2
for step in main scaling threads graphs sweeps; do
  python3 -I run_experiments.py $step > log_$step.txt 2>&1
  echo "$step done $(date -u +%H:%M:%S)" >> progress.txt
done
echo ALLDONE >> progress.txt
