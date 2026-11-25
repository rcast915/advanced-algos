#!/usr/bin/env bash
#SBATCH --job-name=darp
#SBATCH --account=punakha_partner_sdriver
#SBATCH --partition=dgx
#SBATCH --qos=punakha_dgx2_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/output/darp_%j.out
#SBATCH --error=logs/error/darp_%j.err
#
# Runs one job script from jobs/ inside the project's Docker image on a punakha H100 node.
# Submit from the repo root:
#
#   mkdir -p logs/output logs/error                         # sbatch needs these to exist
#   sbatch submit_punakha.sh                                # default: jobs/check.sh
#   DARP_JOB=jobs/gpu.sh sbatch submit_punakha.sh
#   DARP_JOB=jobs/bench.sh sbatch --time=2:00:00 submit_punakha.sh
#
# Watch: tail -f logs/output/darp_<JOBID>.out
# Passed into the container if set: DARP_RESULTS, PARITY_N.
# The image is (re)built at the start of each job; after the first build Docker's layer cache makes this quick
# if the rootless Docker storage persists between jobs.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JOB="${DARP_JOB:-jobs/check.sh}"
if [[ ! -f "$PROJECT_DIR/$JOB" ]]; then echo "No $JOB in $PROJECT_DIR (submit from the repo root)." >&2; exit 1; fi

echo "Job $SLURM_JOB_ID on $(hostname), $(date -u +%FT%TZ), running $JOB"
echo "Commit: $(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo none)$(git -C "$PROJECT_DIR" diff --quiet 2>/dev/null || echo ' (+ uncommitted changes)')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

module load docker/27.3.1/rootless-docker
start_rootless_docker.sh --quiet
sleep 10

IMAGE=darp-cuda13
docker build --network=host -t "$IMAGE" "$PROJECT_DIR/docker"

docker run --rm --gpus all --network=host \
  -v "$PROJECT_DIR":/work -w /work \
  -e DARP_RESULTS -e PARITY_N \
  "$IMAGE" bash "$JOB" "${SLURM_JOB_ID:-manual}"
