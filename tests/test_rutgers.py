"""The Rutgers Slurm helper and the A100 config: checked without a cluster (--dry-run prints the commands)."""

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_helper():
    spec = importlib.util.spec_from_file_location("rutgers", ROOT / "scripts" / "rutgers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a100_config_matches_t4_shape():
    t4 = yaml.safe_load((ROOT / "configs/earnings_t4.yaml").read_text())
    a100 = yaml.safe_load((ROOT / "configs/earnings_a100.yaml").read_text())
    assert set(t4) <= set(a100)
    for k in ("task", "model_id", "data_dir", "lora_r", "lora_alpha", "lora_target_modules", "learning_rate"):
        assert a100[k] == t4[k], k
    # same effective batch, so the step counts are comparable
    assert (a100["per_device_train_batch_size"] * a100["gradient_accumulation_steps"]
            == t4["per_device_train_batch_size"] * t4["gradient_accumulation_steps"])
    assert a100["bf16"] and not a100["fp16"]
    assert a100["output_dir"] != t4["output_dir"]  # never overwrites the T4 adapter
    if a100.get("load_best_model_at_end"):
        assert a100["save_steps"] == a100["eval_steps"]


def test_submit_dry_run(capsys):
    helper = load_helper()
    helper.main(["--user", "abc123", "--dry-run", "submit", "train", "--gpu", "a6000:1"])
    out = capsys.readouterr().out
    assert "abc123@ilab1.cs.rutgers.edu" in out
    assert "sbatch --parsable -G a6000:1 slurm/train_a100.sbatch" in out
    assert "CONFIG=configs/earnings_a100.yaml RUN_EVAL=1" in out
    assert "BatchMode=yes" not in out or "ControlMaster=no" in out


def test_download_never_puts_token_on_command_line(capsys, monkeypatch):
    helper = load_helper()
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    helper.main(["--dry-run", "download"])
    out = capsys.readouterr().out
    assert "hf_secret_value" not in out
    assert "read -r HF_TOKEN" in out


def test_job_scripts_exist():
    helper = load_helper()
    for script in helper.JOBS.values():
        text = (ROOT / script).read_text()
        assert text.startswith("#!/bin/bash -l")
        assert "HF_HUB_OFFLINE=1" in text
