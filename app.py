"""Phase 4: Gradio demo. Paste a filing excerpt (text and/or a table) and ask a question.

Run on a GPU machine (Colab T4 works):
    python app.py --adapter outputs/finsight-mistral-7b-qlora/final --share

With --compare, each question is answered by both the base model and the
fine-tuned model side by side (one 4-bit base in memory; the adapter is
toggled on and off).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXAMPLES_FILE = Path("data/processed/test_finqa.jsonl")

FALLBACK_EXAMPLE = (
    "| | 2019 | 2018 |\n|---|---|---|\n| Net revenue | $5,829 | $5,735 |\n| Operating income | $1,212 | $1,048 |",
    "What was the percentage change in operating income from 2018 to 2019?",
)


def load_examples(n: int = 4) -> list[list[str]]:
    if not EXAMPLES_FILE.exists():
        return [list(FALLBACK_EXAMPLE)]
    rows = [json.loads(l) for l in EXAMPLES_FILE.read_text().splitlines()[:50]]
    rows = sorted(rows, key=lambda r: len(r["context"]))[:n]  # short ones read best in a demo
    return [[r["context"], r["question"]] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/finsight-mistral-7b-qlora/final")
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--compare", action="store_true", help="also show the base model's answer")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = ap.parse_args()

    import gradio as gr

    from finsight.inference import generate, load

    model, tokenizer = load(args.model_id, args.adapter)

    def answer(context: str, question: str):
        ex = [{"context": context, "question": question}]
        tuned = generate(model, tokenizer, ex, max_new_tokens=300)[0]
        if not args.compare:
            return tuned
        with model.disable_adapter():
            base = generate(model, tokenizer, ex, max_new_tokens=300)[0]
        return tuned, base

    outputs = [gr.Textbox(label="FinSight (fine-tuned)", lines=10)]
    if args.compare:
        outputs.append(gr.Textbox(label="Base Mistral-7B-Instruct", lines=10))

    demo = gr.Interface(
        fn=answer,
        inputs=[gr.Textbox(label="Filing excerpt (text or markdown table)", lines=14),
                gr.Textbox(label="Question", lines=2)],
        outputs=outputs,
        examples=load_examples(),
        title="FinSight: financial filing Q&A",
        description=("Mistral-7B fine-tuned with QLoRA on FinQA and TAT-QA to answer numerical questions about "
                     "10-K tables and text, showing each calculation step."),
        flagging_mode="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
