#!/bin/bash -l
# Run on a Rutgers login node (iLab or rLab), never inside a job:
#   bash slurm/setup.sh env        create the training venv (about 10 minutes, about 6 GB)
#   bash slurm/setup.sh serve-env  create the vLLM venv, only needed for slurm/serve_eval_a100.sbatch
#   bash slurm/setup.sh download   fetch Mistral-7B into $HF_HOME; needs HF_TOKEN in the environment
#   bash slurm/setup.sh check      show what is in place
# Jobs run offline from these caches, so they work even if compute nodes have no internet.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
cd "$FINSIGHT_REPO"

case "${1:-check}" in
  env)
    [ -x "$TRAIN_VENV/bin/python" ] || "$PYTHON_BIN" -m venv "$TRAIN_VENV"
    "$TRAIN_VENV/bin/pip" install -q --upgrade pip
    # torch first (the CUDA 12.x wheel carries its own CUDA runtime; CS machines need driver support for 12.6+)
    "$TRAIN_VENV/bin/pip" install -q "torch>=2.5"
    "$TRAIN_VENV/bin/pip" install -q -r requirements.txt -r requirements-dev.txt
    "$TRAIN_VENV/bin/python" -c "import torch, transformers, peft, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__)"
    ;;
  serve-env)
    [ -x "$SERVE_VENV/bin/python" ] || "$PYTHON_BIN" -m venv "$SERVE_VENV"
    "$SERVE_VENV/bin/pip" install -q --upgrade pip
    "$SERVE_VENV/bin/pip" install -q -r requirements-serve.txt pyyaml
    "$SERVE_VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
    ;;
  download)
    if [ -z "${HF_TOKEN:-}" ]; then
      echo "HF_TOKEN is not set. Export it in this shell (it is not written to disk), then run again." >&2
      exit 2
    fi
    "$TRAIN_VENV/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    os.environ["MODEL_ID"],
    token=os.environ["HF_TOKEN"],
    # the safetensors shards, config and tokenizer; skips consolidated.safetensors, a second 14 GB copy
    allow_patterns=["*.json", "model-*.safetensors", "tokenizer*", "*.model"],
)
print("model cached at", path)
PY
    ;;
  check)
    echo "root:  $FINSIGHT_ROOT"
    echo "repo:  $(git -C "$FINSIGHT_REPO" log --oneline -1 2>/dev/null || echo "$FINSIGHT_REPO (copied without git)")"
    for f in data/earnings/train.jsonl data/earnings/val.jsonl data/earnings/test_earnings.jsonl; do
      if [ -f "$f" ]; then echo "data:  $f ($(wc -l < "$f") rows)"; else echo "data:  MISSING $f"; fi
    done
    [ -x "$TRAIN_VENV/bin/python" ] && echo "venv:  $TRAIN_VENV" || echo "venv:  MISSING (bash slurm/setup.sh env)"
    [ -x "$SERVE_VENV/bin/python" ] && echo "vllm:  $SERVE_VENV" || echo "vllm:  not set up (only needed for serve_eval)"
    if [ -d "$HF_HOME/hub/models--${MODEL_ID//\//--}" ]; then
      echo "model: cached ($(du -sh "$HF_HOME/hub/models--${MODEL_ID//\//--}" | cut -f1))"
    else
      echo "model: MISSING (HF_TOKEN=... bash slurm/setup.sh download)"
    fi
    quota -vs 2>/dev/null | tail -n +1 || true
    ;;
  *)
    echo "usage: bash slurm/setup.sh {env|serve-env|download|check}" >&2; exit 2 ;;
esac
