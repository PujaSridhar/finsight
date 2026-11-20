#!/usr/bin/env bash
# One run of the scheduled live job: watch | score | page | backfill DATE.
# Reads SEC_USER_AGENT, FINSIGHT_MODEL_URL, FINSIGHT_MODEL_NAME and FINSIGHT_MODEL_KEY from the environment.
set -euo pipefail
TASK="${1:-watch}"
if [ -z "${SEC_USER_AGENT:-}" ]; then
  echo "::warning::Add the SEC_USER_AGENT repository secret (your name and email) to run the live watcher."
  exit 0
fi
if [ "$TASK" = "watch" ] || [ "$TASK" = "backfill" ]; then
  if [ -z "${FINSIGHT_MODEL_URL:-}" ]; then
    echo "::warning::Add the FINSIGHT_MODEL_URL repository secret (an endpoint serving the FinSight adapter); see docs/LIVE.md."
    exit 0
  fi
fi
bash scripts/live_data_branch.sh
case "$TASK" in
  watch)    python -m finsight.live poll ;;
  backfill) python -m finsight.live backfill --date "${2:?give a date, YYYY-MM-DD}" ;;
  score)    python -m finsight.live score ;;
  page)     ;;
  *)        echo "unknown task $TASK"; exit 1 ;;
esac
python -m finsight.dashboard --live live --out site
cd live
git add -A
if ! git diff --cached --quiet; then
  git commit -qm "live: $TASK $(date -u +%Y-%m-%dT%H:%MZ)"
  git push -q origin live-data
fi
