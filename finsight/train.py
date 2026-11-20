"""QLoRA fine-tuning of Mistral-7B-Instruct.

    python -m finsight.train --config configs/earnings_t4.yaml           # the FinSight model
    python -m finsight.train --config configs/earnings_t4.yaml --resume  # after a disconnect
    python -m finsight.train --config configs/qlora_t4.yaml              # optional FinQA/TAT-QA warm-up stage

The config's `task` picks the data and prompt (finsight/tasks.py). `init_adapter`
continues training an existing adapter, which is how the warm-up stage feeds the
earnings stage.

Design choices (and why):

* 4-bit NF4 base weights + LoRA adapters on every linear layer (QLoRA,
  Dettmers et al. 2023): the 7B model fits a 16 GB T4 with room for 2K-token
  sequences.
* Loss only on the assistant response. The prompt (instructions + release
  or filing excerpt) is over 95% of each sequence, so training on it would
  mostly teach the model to reproduce filing text instead of reading it.
* Over-length examples are dropped, never truncated, so no target answer is
  cut off mid-calculation.
* Plain transformers Trainer + peft rather than a higher-level wrapper, so the
  masking logic is explicit and stable across library versions.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import yaml

from . import tasks


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def encode_example(tokenizer, ex: dict, max_len: int, task=None) -> dict | None:
    """Tokenize one example with the chat template, masking the prompt.

    Returns None if prompt + response exceed max_len.
    """
    task = task or tasks.get("qa")
    prompt_messages = task.messages(ex)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": task.target(ex)}], tokenize=False
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template output does not start with the prompt; check the tokenizer's template")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] == prompt_ids:
        # Usual case: the prompt tokenizes identically on its own and inside the
        # full sequence, so training sees exactly what inference will see.
        response_ids = full_ids[len(prompt_ids):]
    else:
        response_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
        response_ids = response_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_len:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def build_split(tokenizer, rows: list[dict], max_len: int, name: str, task=None):
    from datasets import Dataset

    encoded, dropped = [], 0
    for ex in rows:
        enc = encode_example(tokenizer, ex, max_len, task)
        if enc is None:
            dropped += 1
        else:
            encoded.append(enc)
    lengths = sorted(len(e["input_ids"]) for e in encoded)
    print(f"  {name}: {len(encoded)} examples kept, {dropped} dropped as > {max_len} tokens; "
          f"median length {lengths[len(lengths) // 2]} tokens, max {lengths[-1]}")
    return Dataset.from_list(encoded), {"kept": len(encoded), "dropped_too_long": dropped,
                                        "median_tokens": lengths[len(lengths) // 2], "max_tokens": lengths[-1]}


def check_coverage(info: dict, cfg: dict, max_dropped: float = 0.2) -> None:
    """Stop before loading the model if most of the data doesn't fit max_seq_len.

    Training on the few short releases that fit would give a model that looks trained but has seen almost
    nothing; better to fail here and rebuild the data with a shorter excerpt (or raise max_seq_len).
    """
    total = info["kept"] + info["dropped_too_long"]
    if total and info["dropped_too_long"] / total > cfg.get("max_dropped_fraction", max_dropped):
        raise SystemExit(
            f"{info['dropped_too_long']} of {total} training examples are longer than max_seq_len "
            f"({cfg['max_seq_len']} tokens). Rebuild the data with a shorter excerpt "
            f"(python -m finsight.earnings --max-chars ...) or raise max_seq_len.")


def load_model_for_training(cfg: dict):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if cfg.get("bf16") else torch.float16
    quant = None
    if cfg.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], quantization_config=quant, torch_dtype=compute_dtype, device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
    if cfg.get("init_adapter"):
        from peft import PeftModel

        # Continue from an earlier stage's adapter (e.g. the FinQA/TAT-QA warm-up) instead of a fresh one.
        model = PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=True)
        model.print_trainable_parameters()
        return model
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def save_loss_curve(log_history: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    train = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(7, 4))
    if train:
        ax.plot(*zip(*train), label="train loss", color="#2a6f97")
    if evals:
        ax.plot(*zip(*evals), label="validation loss", color="#e07a5f", marker="o")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (response tokens)")
    ax.set_title("FinSight QLoRA training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def best_checkpoint_args(cfg: dict) -> dict:
    """Longer runs (configs/earnings_a100.yaml) keep the checkpoint with the lowest validation loss as final/."""
    if not cfg.get("load_best_model_at_end"):
        return {}
    if cfg["save_steps"] != cfg["eval_steps"]:
        raise SystemExit("load_best_model_at_end needs save_steps equal to eval_steps")
    return {"load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/qlora_t4.yaml")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "train_config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # padded positions get label -100, so this is safe
    tokenizer.padding_side = "right"

    task = tasks.get(cfg.get("task", "qa"))
    data_dir = Path(cfg["data_dir"])
    rng = random.Random(cfg["seed"])
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    rng.shuffle(train_rows)
    train_rows = train_rows[: cfg["max_train_examples"]]
    val_rows = val_rows[: cfg["max_val_examples"]]
    print("Tokenizing...")
    train_ds, train_info = build_split(tokenizer, train_rows, cfg["max_seq_len"], "train", task)
    val_ds, val_info = build_split(tokenizer, val_rows, cfg["max_seq_len"], "val", task)
    check_coverage(train_info, cfg)

    model = load_model_for_training(cfg)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        group_by_length=True,
        report_to=cfg.get("report_to", "none"),
        seed=cfg["seed"],
        remove_unused_columns=False,
        **best_checkpoint_args(cfg),
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    minutes = (time.time() - start) / 60

    final = out_dir / "final"
    trainer.model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    eval_metrics = trainer.evaluate()

    summary = {
        "model_id": cfg["model_id"],
        "task": task.name,
        "init_adapter": cfg.get("init_adapter"),
        "train_examples": train_info,
        "val_examples": val_info,
        "train_minutes": round(minutes, 1),
        "final_eval_loss": round(eval_metrics.get("eval_loss", float("nan")), 4),
        "global_steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    curve = Path(cfg.get("loss_curve", "results/loss_curve.png"))
    save_loss_curve(trainer.state.log_history, curve)
    print(f"\nSaved adapter to {final} after {minutes:.0f} min. Final val loss {summary['final_eval_loss']}.")
    print(f"Loss curve: {curve}")


if __name__ == "__main__":
    main()
