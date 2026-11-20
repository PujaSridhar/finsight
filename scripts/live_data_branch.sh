#!/usr/bin/env bash
# Check out the live-data branch into live/ (creating the branch the first time).
#
# The live log (releases.jsonl, scores.jsonl, ...) lives on its own branch so the watcher's commits every few
# minutes never touch main. Used by .github/workflows/live.yml; run it locally the same way if you want to
# commit a laptop session to the branch.
set -euo pipefail
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
fi
if [ -e live/.git ]; then
  (cd live && git pull -q --rebase origin live-data 2>/dev/null || true)
  exit 0
fi
if git ls-remote --exit-code --heads origin live-data >/dev/null 2>&1; then
  git fetch -q origin live-data
  git worktree add -q -B live-data live origin/live-data
else
  git worktree add -q --detach live
  (cd live && git checkout -q --orphan live-data && git rm -rqf --ignore-unmatch . \
    && printf '# FinSight live log\n\nWritten by the live watcher (finsight/live.py). See docs/LIVE.md on main.\n' > README.md)
fi
