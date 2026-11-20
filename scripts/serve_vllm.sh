#!/usr/bin/env bash
# Serve Mistral-7B-Instruct-v0.3 with the FinSight LoRA adapter through vLLM's OpenAI-compatible server.
#
#   pip install -r requirements-serve.txt      # in a fresh runtime: vLLM installs its own torch
#   bash scripts/serve_vllm.sh outputs/finsight-mistral-7b-qlora/final
#
# --quantization fp8 quantizes the fp16 weights to FP8 at load time (about 7.5 GB instead of 14.5 GB), which leaves
# a 16 GB T4 room for the KV cache; on a T4 vLLM runs it as a weight-only FP8 kernel. The adapter was trained on a
# 4-bit NF4 base, so serve-time numbers can differ slightly from notebook 03: re-run the eval gate against this
# endpoint (python -m finsight.ci_eval --endpoint http://localhost:8000 --served-model finsight).
# Not run in this repo's CI (no GPU). If vLLM will not start on your GPU, use: python -m finsight.serve --backend transformers
set -euo pipefail
ADAPTER="${1:-outputs/finsight-mistral-7b-qlora/final}"
PORT="${PORT:-8000}"

exec vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --dtype half \
  --quantization fp8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules "finsight=${ADAPTER}" \
  --port "${PORT}"
