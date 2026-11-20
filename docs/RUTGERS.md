# Training FinSight on the Rutgers CS GPU cluster

The first FinSight adapter was trained on a free 16 GB T4: one pass over the 799 training releases that fit in
3,072 tokens (50 optimizer steps, 2 h 26 min). The 185 longer releases were left out. The cluster's A100s (40 GB)
allow a longer run, set in [`configs/earnings_a100.yaml`](../configs/earnings_a100.yaml):

| | T4 run (`earnings_t4.yaml`) | A100 run (`earnings_a100.yaml`) |
|---|---|---|
| Releases that fit | 799 of 984 (up to 3,072 tokens) | most of the 984 (up to 4,096 tokens) |
| Passes over the data | 1 | 3 |
| Validation | at the end, 68 releases | every 20 steps, whole split |
| Kept as `final/` | last step | the checkpoint with the lowest validation loss |
| Precision | fp16 | bf16 |

The data, prompt, LoRA shape, learning rate and effective batch (16 releases per step) are the same, so the
two adapters compare directly. The A100 validation set includes the longer releases, so its validation loss is not
directly comparable with the T4's 0.0058; the 100-release test run at the end of the job is the fair comparison.

## What's in the repo

- `slurm/env.sh`: where everything lives on the cluster. `FINSIGHT_ROOT` (default `/common/users/$USER/finsight`)
  is the one setting to change if you need more space. `/tmp` is capped at 3 GB there, so temp files go under it too.
- `slurm/setup.sh`: run on a login node: `env` (training venv), `download` (Mistral-7B into the cache),
  `serve-env` (vLLM venv), `check`.
- `slurm/train_a100.sbatch`: training on one GPU, then the fine-tuned model on the 100 test releases
  (`results/compare_a100/`). Resubmitting resumes from the latest checkpoint.
- `slurm/serve_eval_a100.sbatch`: optional. Serves the adapter with vLLM and runs the eval gate on 50 releases.
- `scripts/rutgers.py`: does all of the above from your laptop over one SSH login.

Jobs run with `HF_HUB_OFFLINE=1` from the cache that `download` fills on the login node. They work the same whether
or not compute nodes have internet access.

## Before you start (once)

1. You can log in to an iLab machine: `ssh NETID@ilab1.cs.rutgers.edu`.
2. You have a Hugging Face read token, and your account has accepted the Mistral-7B-Instruct-v0.3 terms.
3. Put the dataset on your laptop. In Google Drive, open My Drive › finsight › data, right-click the `earnings`
   folder, choose Download, then unzip it, for example to `~/Downloads/earnings`. It must contain `train.jsonl`,
   `val.jsonl` and `test_earnings.jsonl`. Using this copy keeps the test set identical to the Kaggle comparison.
4. Clone the repo on your laptop and check out this branch (or `main` once it is merged).

## The commands, in order

Run these from the repo folder on your laptop. Step 2 is the only one that asks you for anything.

```bash
# 1. who you are on the cluster (and, for step 5, your Hugging Face token; it is never written to a file)
export RUTGERS_NETID=your_netid
export HF_TOKEN=hf_...

# 2. log in once: type your password and approve Duo. The connection stays open for 8 hours.
python scripts/rutgers.py connect

# 3. copy the code and the dataset up
python scripts/rutgers.py push --data ~/Downloads/earnings

# 4. build the Python environment on the login node (about 10 minutes)
python scripts/rutgers.py setup env

# 5. cache Mistral-7B on the cluster (about 15 GB)
python scripts/rutgers.py download

# 6. check everything is in place, and see the GPU type names
python scripts/rutgers.py check
python scripts/rutgers.py gpus

# 7. submit the training job (the GPU name is whatever `gpus` lists for the A100, usually a100)
python scripts/rutgers.py submit train --gpu a100:1

# 8. follow it
python scripts/rutgers.py wait JOBID        # prints state changes, then the end of the log
python scripts/rutgers.py logs JOBID        # the last 60 lines at any time
python scripts/rutgers.py status            # all your jobs

# 9. bring the adapter, training summary, loss curve and test results back
python scripts/rutgers.py fetch
```

If no A100 is free, an RTX A6000 (48 GB) runs the same config: `--gpu a6000:1`. To stop a job:
`python scripts/rutgers.py cancel JOBID`. If the SSH connection expires, run step 2 again; nothing on the
cluster is lost.

Optional, after training: serve the new adapter with vLLM and run the eval gate.

```bash
python scripts/rutgers.py setup serve-env
python scripts/rutgers.py submit serve-eval --gpu a100:1
```

## Without the helper (plain SSH)

```bash
ssh NETID@ilab1.cs.rutgers.edu
mkdir -p /common/users/$USER/finsight && cd /common/users/$USER/finsight
git clone https://github.com/PujaSridhar/finsight repo && cd repo   # add -b BRANCH to try a branch
# copy the dataset from your laptop, in a second terminal:
#   scp ~/Downloads/earnings/*.jsonl NETID@ilab1.cs.rutgers.edu:/common/users/NETID/finsight/repo/data/earnings/
bash slurm/setup.sh env
read -rs HF_TOKEN && export HF_TOKEN       # paste the token; it is not echoed or saved
bash slurm/setup.sh download
unset HF_TOKEN
bash slurm/setup.sh check
mkdir -p slurm-logs && sbatch -G a100:1 slurm/train_a100.sbatch
squeue -u $USER
tail -f slurm-logs/finsight-train-JOBID.out
```

The adapter ends up in `outputs/finsight-earnings-a100/final/`, the summary in
`outputs/finsight-earnings-a100/training_summary.json`, and the test results in `results/compare_a100/`.

## Prompt for Claude Code on your laptop

Run step 2 (`connect`) yourself first, in your own terminal, so the password and Duo stay with you. Then paste:

> In this repo, follow docs/RUTGERS.md to train FinSight on the Rutgers CS cluster using scripts/rutgers.py. I have
> already run `python scripts/rutgers.py connect` and exported RUTGERS_NETID and HF_TOKEN in this terminal. The
> dataset is in ~/Downloads/earnings. Run, in order: push --data ~/Downloads/earnings, setup env, download, check,
> gpus. Pick the A100's type name from the gpus output and submit train with it (use a6000:1 if no A100 is listed).
> Then run wait with the job id. If a command fails, show me the error and the last 40 lines of the job log, and
> stop: don't change the training config, don't type any password, and never print or save HF_TOKEN. If SSH says
> the connection expired, ask me to run connect again. When the job completes, run fetch and summarize
> outputs/finsight-earnings-a100/training_summary.json (steps, best checkpoint, final validation loss) and
> results/compare_a100/COMPARISON.md.

## If something goes wrong

- **`sbatch: error: Invalid generic resource`**: the GPU name differs. Use a name from `python scripts/rutgers.py gpus`.
- **A job fails right away with a CUDA error**: the venv's torch doesn't match the GPU nodes' driver (the CS docs
  ask for PyTorch built for CUDA 12.6 or newer). See what the driver supports with `srun -G 1 nvidia-smi` on the
  login node, install the matching torch build into `$FINSIGHT_ROOT/venv-train`, and submit again.
- **`OSError: ... offline mode`**: the model was not cached. Run `download` again, then `check`.
- **Out of disk**: run `quota -vs` on the cluster. Set `FINSIGHT_ROOT` to `/freespace/local/$USER/finsight` or
  `/filer/tmp1/$USER/finsight` (no quota, not backed up). Then repeat `push`, `setup env` and `download`.
- **The job is killed after a week, or pre-empted**: submit the same command again; it resumes from the latest
  checkpoint.
