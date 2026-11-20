# Sourced by the Rutgers setup script and the Slurm jobs. Change FINSIGHT_ROOT to move everything big.
# /common/users/$USER has a larger quota than home (/common/home is 50 GB for students), and compute nodes see it.
export FINSIGHT_ROOT="${FINSIGHT_ROOT:-/common/users/$USER/finsight}"
export FINSIGHT_REPO="${FINSIGHT_REPO:-$FINSIGHT_ROOT/repo}"
export HF_HOME="${HF_HOME:-$FINSIGHT_ROOT/hf-cache}"          # Mistral-7B weights, about 15 GB
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$FINSIGHT_ROOT/pip-cache}"
export TMPDIR="$FINSIGHT_ROOT/tmp"                            # /tmp is capped at 3 GB on CS machines
export TRAIN_VENV="${TRAIN_VENV:-$FINSIGHT_ROOT/venv-train}"    # transformers + peft + bitsandbytes
export SERVE_VENV="${SERVE_VENV:-$FINSIGHT_ROOT/venv-serve}"    # vLLM pins its own torch, so it gets its own venv
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
export MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
mkdir -p "$FINSIGHT_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
