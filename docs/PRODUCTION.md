# FinSight production path

How the fine-tuned model would run as a service, how changes are checked, and when to retrain. Everything here
runs on a single GPU (a free Colab T4 works for trying it); none of it was executed in this repo's CI, which has
no GPU.

## Serving

```
client ──POST /v1/extract──▶ FastAPI (finsight/serve.py) ──▶ vLLM (scripts/serve_vllm.sh)
         {"url": sec.gov URL}   fetch from EDGAR, cut excerpt,     Mistral-7B, FP8 weights,
         or {"text": ...}       parse JSON, log, return timing     FinSight LoRA adapter
```

```bash
pip install -r requirements-serve.txt              # fresh runtime: vLLM brings its own torch
bash scripts/serve_vllm.sh outputs/finsight-earnings/final
export SEC_USER_AGENT="Your Name your.email@example.com"
python -m finsight.serve --backend vllm &          # API on :8080
curl -s localhost:8080/v1/extract -H 'content-type: application/json' -d '{"text": "...release text..."}'
```

- **Why vLLM:** continuous batching and paged KV cache let one GPU serve many releases at once, which is the real
  load on a busy earnings morning. The adapter stays separate (`--enable-lora`), so a new adapter is a file swap,
  and the base model and the adapter can be served side by side for comparison.
- **Why FP8:** vLLM 0.34 no longer serves bitsandbytes 4-bit, and FP16 Mistral-7B (14.5 GB) leaves almost no room
  for the KV cache on a 16 GB T4. `--quantization fp8` quantizes weights at load time to about 7.5 GB; on a T4
  vLLM runs it as a weight-only kernel. If vLLM will not start on your GPU, `--backend transformers` serves the same
  API from an in-process 4-bit model, one request at a time.
- **API behavior:** URLs other than sec.gov are refused (the server cannot be pointed at arbitrary sites); a
  backend error returns 502, never a guessed answer; an unparseable model output returns `fields: null` with
  `valid_json: false` so a caller can fall back.

## Eval on every change (CI)

`.github/workflows/tests.yml` runs two jobs on every push and pull request:

1. **tests:** unit tests for parsing, labeling, scoring, the API, the gate and monitoring, with models replaced by
   stand-ins, plus the warm-up dataset build (which checks its own labels).
2. **eval-gate:** `python -m finsight.ci_eval`
   - **Re-score** (always): saved predictions under `results/` are scored again with the current code. If a
     published accuracy moves, the change silently altered the README's numbers, and the job fails until the
     metrics are regenerated on purpose.
   - **Smoke eval** (when the `FINSIGHT_EVAL_ENDPOINT` secret is set): the 50 releases in
     `results/ci_sample.jsonl` go to a served model, and the job fails if output stops parsing (below 95%) or
     accuracy falls more than 8 points below the promoted model's recorded baseline. 50 releases carry about ±7
     points of noise, so this catches breakage (a bad prompt change, a broken adapter, a serving misconfiguration),
     not small regressions; the full comparison in `finsight/compare.py` is for those.

To arm the smoke eval after training: serve the model, run
`python -m finsight.ci_eval --endpoint http://localhost:8000 --write-baseline`, commit `results/ci_baseline.json`
and `results/ci_sample.jsonl`, and add the endpoint as a repository secret.

## Monitoring

Every request writes one JSON line to `logs/requests.jsonl` (no release text unless `--log-content`), and
`python -m finsight.monitor logs/requests.jsonl` prints a daily summary. What to watch, and why:

| Signal | Why it matters | Act when |
|---|---|---|
| Error rate | Backend down, out of memory, EDGAR fetch failures | Above 1% on a day |
| p50 / p95 latency | The point of a small model is speed; queueing shows here first on busy mornings | p95 above the latency you promised |
| Valid JSON rate | Format drift is the first sign the model is off-distribution | Below the CI gate's 95% |
| Null fields | A field the model stopped finding means a layout it has not seen | Rising for a field over a week |
| Input size | Releases getting longer can push the income statement out of the excerpt | Median moves noticeably |

Logs cannot show accuracy, because the right answer arrives later. That is also the fix: **each quarter's 10-Qs
label the releases you served**. Re-running the label build on served releases a few weeks later gives a true
production accuracy, by company and field, with no hand labeling.

## Retraining

Retrain when the delayed production accuracy, or the CI smoke eval, drops below the promoted model, or when a
new filing season brings a layout change visible in the null-field rate. The loop:

1. Rebuild the dataset (`python -m finsight.earnings`): new releases get labels from the newest 10-Qs and 10-Ks.
2. Train a new adapter with the same config; keep the test companies held out.
3. Compare old and new adapters with `finsight/compare.py` on the same test releases; promote only if the new one
   is at least as accurate and no slower.
4. Serve both adapters in vLLM, send a small share of traffic to the new one, check the monitoring signals, then
   switch and write a new CI baseline.
