# FinSight live

The test set says how the model does on past releases. The live run says how it does on releases nobody has
seen yet, as they are filed, with the answer key arriving weeks later from the company's own 10-Q. Nothing in the
live log can be tuned after the fact: the answer is written down before the official figures exist.

```
SEC EDGAR latest filings ──poll──▶ finsight/live.py ──release text──▶ model endpoint (vLLM + FinSight adapter)
  (8-K, Item 2.02)                   │  writes releases.jsonl: answer, SEC acceptance time, seen at, model time
                                     │
weeks later: XBRL companyfacts ──▶ live.py score ──▶ scores.jsonl: right or wrong per field
                                     │
                                     └──▶ finsight/dashboard.py ──▶ public page (GitHub Pages)
```

None of this was run against the live SEC site or a served model while it was being built (both were out of
reach); it is tested on recorded feed, index and XBRL fixtures, so the first live run may need a small fix.

## What gets measured

| Number | How | Caveat |
|---|---|---|
| Releases answered | Every 8-K with Item 2.02 that has a press release exhibit | Filings with no exhibit are counted separately |
| Accuracy, all four fields and per field | `live.py score` compares each answer with the first XBRL figures filed for that quarter, using the same tolerances as the test set | Shown two ways: releases whose official figures are printed in the text (comparable to the test set), and all scored releases |
| Why answers were wrong | Same error classes as the test set: year-ago quarter, year-to-date, scale, sign, likely non-GAAP, missing | |
| Model time | Time for the endpoint to answer one release | Includes a cold start if the endpoint had scaled to zero |
| SEC acceptance to seen, and to answer | Live polls only; first polls and polls after a gap are marked `catch_up` and left out | Measured from SEC acceptance; the newswire can be a few minutes earlier. Bounded below by the polling interval |
| Unscorable | No XBRL quarter within 120 days: mostly fourth quarters (often only annual figures are tagged), banks and insurers | Counted on the page |

## Running it

There are two parts with very different costs. The **watcher** (polling EDGAR, logging, scoring, the page) needs
no GPU and runs anywhere. The **model** needs a GPU, but only for about a second per release, and only a few
thousand releases arrive per quarter, bunched into weekday mornings and evenings.

### Option A: a Colab session on busy days (free)

`notebooks/06_live.ipynb` serves the model with vLLM on a free T4 and runs the watcher next to it, polling every
30 seconds. Detection is fastest this way, and it costs nothing, but a free Colab session ends after a few hours
and can't run around the clock. Run it on heavy reporting days (late January, late April, late July, late
October) during 6 to 9 am or 4 to 6 pm Eastern, and merge each session into the public log with the notebook's
last cell.

### Option B: the scheduled GitHub job plus a pay-per-second GPU endpoint (always on)

`.github/workflows/live.yml` polls EDGAR every 10 minutes on weekdays, scores once a day and rebuilds the page
every hour. GitHub's runners are free for a public repository. The workflow calls the model at
`FINSIGHT_MODEL_URL`, any OpenAI-compatible endpoint serving Mistral-7B-Instruct-v0.3 with the FinSight adapter.
Serverless GPU hosts that run vLLM and scale to zero when idle (Modal, RunPod Serverless, Hugging Face Inference
Endpoints and others) bill per second of GPU time, so the cost is roughly:

    cost per release ≈ GPU price per second × (model seconds per release + cold-start seconds when it was idle)

Check the provider's current price for the GPU you choose. A cold start loads the model, which can take a minute
or more, so the first release after a quiet stretch shows a much higher model time; the page's 95th percentile
will show it. GitHub can also start scheduled runs late on busy days, so expect detection in minutes here, not
seconds.

An always-on GPU (a rented VM or a paid notebook left running) gives steady latency, but you pay for every idle
hour, which is most of them. It is not worth it for a portfolio project.

### Option C: your laptop as the watcher

The watcher itself runs fine on a laptop while it's awake:

```bash
pip install -r requirements-live.txt
export SEC_USER_AGENT="Your Name your.email@example.com" FINSIGHT_MODEL_URL=https://...  # endpoint from option B
python -m finsight.live poll --loop --interval 60
```

## Setup for option B

1. **Pick an endpoint.** Deploy vLLM with `mistralai/Mistral-7B-Instruct-v0.3` and the adapter from
   `outputs/finsight-earnings/final` (upload the adapter folder to the host, or to a private Hugging Face repo it
   can read). Use the flags in `scripts/serve_vllm.sh`; on a GPU with more memory than a T4 you can drop
   `--quantization fp8`. Note the served model name (`finsight` with that script).
2. **Repository secrets** (Settings > Secrets and variables > Actions):
   - `SEC_USER_AGENT`: your name and email, which the SEC requires of every client
   - `FINSIGHT_MODEL_URL`: the endpoint's base URL
   - `FINSIGHT_MODEL_NAME`: the served model name, if it isn't `finsight`
   - `FINSIGHT_MODEL_KEY`: the endpoint's API key, if it has one
3. **Pages:** Settings > Pages > Source: GitHub Actions. The page appears at
   `https://<your-username>.github.io/finsight/` after the next hourly run.
4. **First run:** Actions > live > Run workflow, task `page`, creates the `live-data` branch. Task `backfill` with
   a past date fills in that day's releases (answered and scored, but not counted for speed).

If GitHub stops the schedule after a long quiet stretch in the repository, re-enable it from the Actions tab.
If the SEC turns away GitHub's runners (it limits automated traffic from some cloud addresses), run the watcher
from your laptop (option C) or Colab (option A) instead.

## Where the data lives

The log is plain JSON lines on the `live-data` branch, so the watcher's commits never touch `main` and anyone can
audit every answer and its timestamp:

- `releases.jsonl`: one line per answered release
- `scores.jsonl`: one line per release once it is final
- `pending.json`: seen but not answered yet (for example, the endpoint was down); retried on the next poll
- `state.json`: where the last poll stopped
