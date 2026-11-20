#!/usr/bin/env python3
"""Run FinSight training on the Rutgers CS GPU cluster (Slurm) from your laptop. Steps in docs/RUTGERS.md.

You log in once, in your own terminal (password and Duo stay with you). Every other command reuses that
connection without prompting, which is what lets Claude Code drive the rest:

    export RUTGERS_NETID=abc123                      # your NetID; optional if ~/.ssh/config already has it
    python scripts/rutgers.py connect                # YOU run this: password + Duo, connection kept for 8 hours
    python scripts/rutgers.py push --data ~/Downloads/earnings
    python scripts/rutgers.py setup env              # training venv on the login node, about 10 minutes
    python scripts/rutgers.py download               # Mistral-7B into the cluster cache; reads HF_TOKEN from your env
    python scripts/rutgers.py gpus                   # GPU type names, to check that a100 is the right one
    python scripts/rutgers.py submit train           # prints the job id
    python scripts/rutgers.py wait JOBID             # polls squeue until the job ends, then shows the log tail
    python scripts/rutgers.py fetch                  # adapter, summary, loss curve and eval results back to your laptop

Nothing here stores a password or a token. HF_TOKEN goes to the login node over the SSH connection's stdin,
for the one download command, and is never written to a file or put on a command line.
Add --dry-run before the command to print what would run without connecting.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTROL_PATH = os.path.expanduser("~/.ssh/cm-finsight-%C")
DEFAULT_HOST = "ilab1.cs.rutgers.edu"
DEFAULT_ROOT = "/common/users/$USER/finsight"   # expanded on the cluster; override with FINSIGHT_ROOT
# Not copied up: local data, model outputs, results and caches. Data goes up only through --data.
PUSH_EXCLUDES = [".git/", "data/", "outputs/", "results/", "slurm-logs/", "logs/", "live/", "site/",
                 ".venv/", "venv/", "__pycache__/", ".pytest_cache/", ".ipynb_checkpoints/", "*.pyc"]
DATA_FILES = ("train.jsonl", "val.jsonl", "test_earnings.jsonl")
JOBS = {"train": "slurm/train_a100.sbatch", "serve-eval": "slurm/serve_eval_a100.sbatch"}


class Cluster:
    def __init__(self, host: str, user: str | None, root: str | None, dry_run: bool = False):
        self.target = f"{user}@{host}" if user else host
        self.root_setting = root
        self.dry_run = dry_run
        self._root: str | None = None

    # -- ssh plumbing -------------------------------------------------------------------------------------
    def ssh_cmd(self, *, interactive: bool = False) -> list[str]:
        opts = ["-o", f"ControlPath={CONTROL_PATH}"]
        if not interactive:
            # Reuse the connection `connect` opened; fail fast instead of prompting for a password.
            opts += ["-o", "ControlMaster=no", "-o", "BatchMode=yes"]
        return ["ssh", *opts]

    def _run(self, argv: list[str], *, stdin: str | None = None, capture: bool = False,
             check: bool = True, show: str | None = None) -> subprocess.CompletedProcess:
        if self.dry_run:
            print("+", show or " ".join(shlex.quote(a) for a in argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        try:
            return subprocess.run(argv, input=stdin, text=True, check=check,
                                  stdout=subprocess.PIPE if capture else None)
        except subprocess.CalledProcessError as e:
            if e.returncode == 255 and argv[0] == "ssh":
                sys.exit("SSH failed. If the connection expired, run `python scripts/rutgers.py connect` again "
                         "in your own terminal.")
            raise

    def remote(self, script: str, *, stdin: str | None = None, capture: bool = False,
               check: bool = True) -> subprocess.CompletedProcess:
        """Run a bash script on the login node, with FINSIGHT_ROOT set and the profile loaded."""
        root = self.root_setting or DEFAULT_ROOT
        full = f'export FINSIGHT_ROOT="{root}"; {script}'
        argv = [*self.ssh_cmd(), self.target, "bash -lc " + shlex.quote(full)]
        return self._run(argv, stdin=stdin, capture=capture, check=check,
                         show=f"ssh {self.target} {shlex.quote(full)}")

    @property
    def root(self) -> str:
        """FINSIGHT_ROOT with $USER expanded on the cluster (rsync paths need it spelled out)."""
        if self._root is None:
            if self.dry_run:
                self._root = self.root_setting or DEFAULT_ROOT
            else:
                out = self.remote('mkdir -p "$FINSIGHT_ROOT/repo" && echo "$FINSIGHT_ROOT"', capture=True)
                self._root = out.stdout.strip().splitlines()[-1]
        return self._root

    @property
    def repo(self) -> str:
        return f"{self.root}/repo"

    def in_repo(self, script: str) -> str:
        return f'cd "$FINSIGHT_ROOT/repo" && {script}'

    def rsync(self, src: str, dst: str, extra: list[str] | None = None, check: bool = True):
        ssh = " ".join(shlex.quote(a) for a in self.ssh_cmd())
        progress = "-v" if self.dry_run or not _rsync_has_info() else "--info=progress2"
        argv = ["rsync", "-az", progress, "-e", ssh,
                *(extra or []), src, dst]
        return self._run(argv, check=check)


def _rsync_has_info() -> bool:
    try:
        out = subprocess.run(["rsync", "--version"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        sys.exit("rsync is not installed on this computer.")
    first = out.splitlines()[0] if out else ""
    return "version 3" in first


# -- commands ---------------------------------------------------------------------------------------------
def cmd_connect(c: Cluster, args) -> None:
    if not sys.stdin.isatty() and not c.dry_run:
        sys.exit("Run `python scripts/rutgers.py connect` yourself in a terminal: it asks for your password "
                 "and Duo approval, which only you should answer.")
    Path(CONTROL_PATH).parent.mkdir(mode=0o700, exist_ok=True)
    argv = [*c.ssh_cmd(interactive=True), "-o", "ControlMaster=yes", "-o", f"ControlPersist={args.hours}h",
            "-fN", c.target]
    c._run(argv)
    if not c.dry_run:
        print(f"Connected to {c.target}; the connection stays open for {args.hours} hours. "
              "Other commands now run without a password.")


def cmd_check(c: Cluster, args) -> None:
    c._run(["ssh", "-o", f"ControlPath={CONTROL_PATH}", "-O", "check", c.target])
    c.remote(c.in_repo("bash slurm/setup.sh check") + ' 2>/dev/null || echo "repo not pushed yet"', check=False)


def cmd_disconnect(c: Cluster, args) -> None:
    c._run(["ssh", "-o", f"ControlPath={CONTROL_PATH}", "-O", "exit", c.target], check=False)


def cmd_push(c: Cluster, args) -> None:
    data = Path(args.data).expanduser() if args.data else None
    if data:
        missing = [f for f in DATA_FILES if not (data / f).exists()]
        if missing:
            sys.exit(f"{data} is missing {', '.join(missing)}; point --data at the folder with the earnings "
                     "dataset (the 'earnings' folder from notebook 01).")
    excludes = [f"--exclude={p}" for p in PUSH_EXCLUDES]
    c.rsync(f"{REPO}/", f"{c.target}:{c.repo}/", excludes)
    if data:
        c.remote('mkdir -p "$FINSIGHT_ROOT/repo/data/earnings"')
        c.rsync(f"{data}/", f"{c.target}:{c.repo}/data/earnings/")
    c.remote(c.in_repo("bash slurm/setup.sh check"), check=False)


def cmd_setup(c: Cluster, args) -> None:
    c.remote(c.in_repo(f"bash slurm/setup.sh {shlex.quote(args.what)}"))


def cmd_download(c: Cluster, args) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token and not c.dry_run:
        sys.exit("HF_TOKEN is not set in this terminal. Export your Hugging Face read token first "
                 "(export HF_TOKEN=...), then run this again.")
    # The token travels on stdin and lives only in that remote shell's environment.
    c.remote(c.in_repo("IFS= read -r HF_TOKEN; export HF_TOKEN; bash slurm/setup.sh download"),
             stdin=(token or "") + "\n")


def cmd_gpus(c: Cluster, args) -> None:
    c.remote('sinfo -o "%25N %10T %40f %30G" | sort -u', check=False)


def cmd_submit(c: Cluster, args) -> None:
    env = []
    if args.job == "train":
        env.append(f"CONFIG={shlex.quote(args.config)}")
        env.append(f"RUN_EVAL={0 if args.no_eval else 1}")
    else:
        env.append(f"ADAPTER={shlex.quote(args.adapter)}")
    script = c.in_repo(f"mkdir -p slurm-logs && {' '.join(env)} sbatch --parsable -G {shlex.quote(args.gpu)} "
                       f"{JOBS[args.job]}")
    out = c.remote(script, capture=True)
    if not c.dry_run:
        job_id = out.stdout.strip().splitlines()[-1].split(";")[0]
        print(f"submitted {args.job} as job {job_id}")
        print(f"  watch it:  python scripts/rutgers.py wait {job_id}")
        print(f"  log:       python scripts/rutgers.py logs {job_id}")


def cmd_status(c: Cluster, args) -> None:
    if args.job_id:
        c.remote(f"squeue -j {int(args.job_id)} -o '%.10i %.20j %.8T %.10M %.6D %R'; "
                 f"sacct -j {int(args.job_id)} -X -o JobID,JobName%20,State,Elapsed,ExitCode,AllocTRES%40",
                 check=False)
    else:
        c.remote("squeue -u $USER -o '%.10i %.20j %.8T %.10M %.6D %R'", check=False)


def _job_log(job_id: int) -> str:
    return f'ls -t "$FINSIGHT_ROOT"/repo/slurm-logs/*-{job_id}.out 2>/dev/null | head -1'


def cmd_logs(c: Cluster, args) -> None:
    job = int(args.job_id)
    c.remote(f'f=$({_job_log(job)}); if [ -n "$f" ]; then tail -n {int(args.lines)} "$f"; '
             f'else echo "no log yet for job {job} (still pending?)"; fi', check=False)


def cmd_wait(c: Cluster, args) -> None:
    job = int(args.job_id)
    last = None
    while True:
        out = c.remote(f"squeue -h -j {job} -o %T 2>/dev/null", capture=True, check=False)
        state = (out.stdout or "").strip()
        if c.dry_run:
            break
        if not state:
            break
        if state != last:
            print(f"{time.strftime('%H:%M:%S')} job {job}: {state}", flush=True)
            last = state
        time.sleep(args.every)
    final = c.remote(f"sacct -j {job} -X -n -o State%20", capture=True, check=False)
    state = (final.stdout or "").strip().split()[0] if (final.stdout or "").strip() else "UNKNOWN"
    if state == "UNKNOWN":  # no accounting on this cluster: the job scripts end their log with "done:"
        log = c.remote(f'f=$({_job_log(job)}); tail -n 1 "$f" 2>/dev/null', capture=True, check=False)
        state = "COMPLETED" if (log.stdout or "").startswith("done:") else "UNKNOWN"
    print(f"job {job} ended: {state}")
    args.lines = 40
    cmd_logs(c, args)
    if not c.dry_run and state != "COMPLETED":
        sys.exit(1)


def cmd_cancel(c: Cluster, args) -> None:
    c.remote(f"scancel {int(args.job_id)}")


def cmd_fetch(c: Cluster, args) -> None:
    run = args.run.strip("/")
    local = REPO / run
    local.mkdir(parents=True, exist_ok=True)
    # final/ adapter (~160 MB), training_summary.json, train_config.yaml; checkpoints only on request.
    extra = [] if args.checkpoints else ["--exclude=checkpoint-*/"]
    c.rsync(f"{c.target}:{c.repo}/{run}/", f"{local}/", extra)
    # Results the jobs write; any that a job didn't produce are skipped.
    for rel in ("results/compare_a100/", "results/loss_curve_earnings_a100.png", "results/ci_baseline_a100.json",
                "results/ci_sample_a100.jsonl", "slurm-logs/"):
        dst = REPO / rel
        (dst if rel.endswith("/") else dst.parent).mkdir(parents=True, exist_ok=True)
        c.rsync(f"{c.target}:{c.repo}/{rel}", f"{dst}/" if rel.endswith("/") else str(dst), check=False)
    if not c.dry_run:
        print(f"adapter: {local / 'final'}")
        summary = local / "training_summary.json"
        if summary.exists():
            import json

            s = json.loads(summary.read_text())
            print({k: v for k, v in s.items() if k != "log_history"})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("RUTGERS_HOST", DEFAULT_HOST),
                    help="login node (iLab1-4 or rLab hosts); env RUTGERS_HOST")
    ap.add_argument("--user", default=os.environ.get("RUTGERS_NETID"), help="your NetID; env RUTGERS_NETID")
    ap.add_argument("--root", default=os.environ.get("FINSIGHT_ROOT"),
                    help=f"where everything goes on the cluster (default {DEFAULT_ROOT}); env FINSIGHT_ROOT")
    ap.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("connect", help="log in once (password + Duo, in your terminal)")
    p.add_argument("--hours", type=int, default=8)
    p.set_defaults(fn=cmd_connect)
    sub.add_parser("check", help="is the connection up; what is on the cluster").set_defaults(fn=cmd_check)
    sub.add_parser("disconnect", help="close the shared connection").set_defaults(fn=cmd_disconnect)
    p = sub.add_parser("push", help="copy the repo (and optionally the dataset) to the cluster")
    p.add_argument("--data", help="local folder with train.jsonl, val.jsonl, test_earnings.jsonl")
    p.set_defaults(fn=cmd_push)
    p = sub.add_parser("setup", help="create environments on the login node")
    p.add_argument("what", choices=["env", "serve-env", "check"])
    p.set_defaults(fn=cmd_setup)
    sub.add_parser("download", help="cache Mistral-7B on the cluster (HF_TOKEN from your env)").set_defaults(
        fn=cmd_download)
    sub.add_parser("gpus", help="list nodes and GPU type names").set_defaults(fn=cmd_gpus)
    p = sub.add_parser("submit", help="sbatch a job")
    p.add_argument("job", choices=sorted(JOBS))
    p.add_argument("--gpu", default=os.environ.get("RUTGERS_GPU", "a100:1"), help="value for sbatch -G")
    p.add_argument("--config", default="configs/earnings_a100.yaml")
    p.add_argument("--no-eval", action="store_true", help="train only, skip the 100-release test run")
    p.add_argument("--adapter", default="outputs/finsight-earnings-a100/final")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status", help="squeue / sacct")
    p.add_argument("job_id", nargs="?")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("logs", help="tail a job's log")
    p.add_argument("job_id")
    p.add_argument("-n", "--lines", type=int, default=60)
    p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("wait", help="poll until the job ends")
    p.add_argument("job_id")
    p.add_argument("--every", type=int, default=120, help="seconds between checks")
    p.set_defaults(fn=cmd_wait)
    p = sub.add_parser("cancel", help="scancel a job")
    p.add_argument("job_id")
    p.set_defaults(fn=cmd_cancel)
    p = sub.add_parser("fetch", help="copy the trained adapter and results back")
    p.add_argument("--run", default="outputs/finsight-earnings-a100")
    p.add_argument("--checkpoints", action="store_true", help="also copy intermediate checkpoints")
    p.set_defaults(fn=cmd_fetch)

    args = ap.parse_args(argv)
    args.fn(Cluster(args.host, args.user, args.root, dry_run=args.dry_run), args)


if __name__ == "__main__":
    main()
