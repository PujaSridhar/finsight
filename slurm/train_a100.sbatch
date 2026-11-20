#!/bin/bash -l
# Longer FinSight QLoRA run on one A100 (configs/earnings_a100.yaml), then the fine-tuned model on the 100 test
# releases. Submit from the repo directory on a Rutgers login node:
#   mkdir -p slurm-logs && sbatch -G a100:1 slurm/train_a100.sbatch
# Check the GPU type name first with: sinfo -o "%25N %50f %G". Resubmitting the same command resumes from the
# latest checkpoint, which covers the cluster's one-week limit or a pre-empted job.
#SBATCH --job-name=finsight-train
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
set -euo pipefail
source slurm/env.sh
source "$TRAIN_VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights come from $HF_HOME, filled by setup.sh download
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/earnings_a100.yaml}"
RUN_EVAL="${RUN_EVAL:-1}"
out_dir="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")"

echo "job $SLURM_JOB_ID on $(hostname), config $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv

resume=""
if compgen -G "$out_dir/checkpoint-*" > /dev/null; then resume="--resume"; echo "resuming from $out_dir"; fi
python -m finsight.train --config "$CONFIG" $resume

if [ "$RUN_EVAL" = "1" ]; then
  # Same 100 test releases and scoring as notebook 03. Latency here is A100 latency, not comparable with T4 rows.
  python -m finsight.compare --system finetuned --adapter "$out_dir/final" --out-dir results/compare_a100
fi
echo "done: adapter in $out_dir/final"
