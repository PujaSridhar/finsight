"""Load the base model (optionally with the FinSight adapter) and generate answers."""

from __future__ import annotations

from .prompts import build_messages

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def load(model_id: str = DEFAULT_MODEL, adapter: str | None = None, load_in_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if load_in_4bit else None
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched generation
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, torch_dtype=dtype,
                                                 device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, examples: list[dict], max_new_tokens: int = 256, batch_size: int = 8) -> list[str]:
    """Greedy-decode a response for each {context, question} example."""
    import torch

    prompts = [
        tokenizer.apply_chat_template(build_messages(e["context"], e["question"]), tokenize=False,
                                      add_generation_prompt=True)
        for e in examples
    ]
    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_tokens = gen[:, batch["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return [o.strip() for o in outputs]
