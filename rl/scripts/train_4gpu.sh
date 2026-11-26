#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

run_dir=${1:?Usage: bash scripts/train_4gpu.sh runs/name [trainer options]}
shift
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec uv run --locked --extra cuda python -m torch.distributed.run \
  --standalone --nproc-per-node=4 --module -- smat_rl.train \
  --run "$run_dir" --device cuda --lp-backend path-cuda --slot-compatibility footprint \
  --seed 42 --cycles 120 --envs-per-rank 64 --lp-workers 24 --cpu-threads 1 \
  --gamma 1 --gae-lambda 1 --ppo-epochs 4 --decision-samples 256 \
  --gradient-batch 16384 --gpu-memory-fraction .93 --eval-every 0 \
  --policy-snapshot-every 10 --completion-prior "$@"
