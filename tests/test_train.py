"""Prompt masking in training, with a character-level stand-in tokenizer (no model download)."""

from finsight import tasks
from finsight.earnings import target_json
from finsight.train import encode_example


class CharTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"[INST] {m['content']} [/INST]" if m["role"] == "user" else f"{m['content']}</s>"
                       for m in messages)
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_only_the_target_is_trained_on_for_both_tasks():
    tok = CharTokenizer()
    label = {"period_end": "2024-09-28", "revenue": 1, "net_income": 2, "eps_diluted": 0.5}
    earnings_ex = {"text": "release", "target": target_json(label)}
    qa_ex = {"context": "c", "question": "q", "response": "Answer: 1"}
    for task, ex, target in ((tasks.get("earnings"), earnings_ex, earnings_ex["target"]),
                             (tasks.get("qa"), qa_ex, qa_ex["response"])):
        enc = encode_example(tok, ex, 10_000, task)
        trained = "".join(chr(t) for t, lab in zip(enc["input_ids"], enc["labels"]) if lab != -100 and t)
        assert trained == target + "</s>"
    assert encode_example(tok, earnings_ex, 50, tasks.get("earnings")) is None  # too long: dropped, not cut
