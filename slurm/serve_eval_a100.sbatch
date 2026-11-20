#!/bin/bash -l
# Serve the adapter with vLLM on one A100 and run the eval gate against it (notebook 05's checks, as a batch job).
#   mkdir -p slurm-logs && ADAPTER=outputs/finsight-earnings-a100/final sbatch -G a100:1 slurm/serve_eval_a100.sbatch
# Needs: bash slurm/setup.sh serve-env (once). The server stops when the job ends.
#SBATCH --job-name=finsight-serve-eval
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
set -euo pipefail
source slurm/env.sh
source "$SERVE_VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ADAPTER="${ADAPTER:-outputs/finsight-earnings-a100/final}"
PORT="${PORT:-8000}"
nvidia-smi --query-gpu=name,memory.total --format=csv

PORT="$PORT" bash scripts/serve_vllm.sh "$ADAPTER" > "slurm-logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 120); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null; then echo "vLLM is up"; break; fi
  if ! kill -0 $server 2>/dev/null; then echo "vLLM exited:"; tail -50 "slurm-logs/vllm-$SLURM_JOB_ID.log"; exit 1; fi
  sleep 10
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "vLLM did not start in 20 minutes"; exit 1; }

# 50 fixed test releases through the served model; prints accuracy and valid-JSON rate.
python -m finsight.ci_eval --endpoint "http://localhost:$PORT" --served-model finsight \
  --baseline results/ci_baseline_a100.json --sample results/ci_sample_a100.jsonl --write-baseline
echo "done: eval gate run against $ADAPTER"
