"""FinSight demo: paste an earnings release (or its sec.gov URL), get the key figures as JSON with timing.

Run on a GPU machine (Colab T4 works):
    export SEC_USER_AGENT="Your Name your.email@example.com"    # only needed for URLs
    python app.py --adapter outputs/finsight-earnings/final --compare --share

Backends: the model in-process (default; 4-bit, one request at a time), or a
running vLLM server (--vllm-url http://localhost:8000), which is the fast path.
With --compare (in-process only) the base model answers too, by switching the
adapter off, so you can see where it goes wrong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finsight.edgar import is_sec_url
from finsight.serve import VLLMBackend, extract

EXAMPLES_FILE = Path("data/earnings/test_earnings.jsonl")


def load_examples(n: int = 3) -> list[list[str]]:
    if not EXAMPLES_FILE.exists():
        return []
    rows = [json.loads(l) for l in EXAMPLES_FILE.read_text().splitlines()[:n]]
    return [[r["url"], ""] for r in rows]


class LocalBackend:
    def __init__(self, model, tokenizer, use_adapter: bool = True):
        from finsight import inference

        self.inference, self.model, self.tokenizer, self.use_adapter = inference, model, tokenizer, use_adapter

    def complete(self, messages):
        def run():
            return self.inference.generate_with_stats(self.model, self.tokenizer, [messages], 128, 1)[0]
        if self.use_adapter:
            return run()
        with self.model.disable_adapter():
            return run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/finsight-earnings/final")
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--vllm-url", default=None, help="use a running vLLM server instead of loading the model")
    ap.add_argument("--compare", action="store_true", help="also show the base model's output (in-process only)")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = ap.parse_args()

    import gradio as gr

    if args.vllm_url:
        tuned, base = VLLMBackend(args.vllm_url, "finsight"), None
        if args.compare:
            base = VLLMBackend(args.vllm_url, args.model_id)  # vLLM also serves the base model under its own name
    else:
        from finsight.inference import load

        model, tokenizer = load(args.model_id, args.adapter)
        tuned = LocalBackend(model, tokenizer)
        base = LocalBackend(model, tokenizer, use_adapter=False) if args.compare else None

    fetcher = {}

    def get_text(url: str, text: str) -> str:
        if text.strip():
            return text
        if not url.strip():
            raise gr.Error("Paste a release or an sec.gov URL.")
        if not is_sec_url(url.strip()):
            raise gr.Error("Only https://www.sec.gov/ URLs are fetched.")
        if "client" not in fetcher:
            from finsight.edgar import SECClient

            fetcher["client"] = SECClient()
        return fetcher["client"].release_text(url.strip())

    def show(out: dict) -> str:
        return json.dumps({"fields": out["fields"], "model_ms": out["model_ms"],
                           "tokens_in_out": [out["input_tokens"], out["output_tokens"]]}, indent=2)

    def run(url: str, text: str):
        release = get_text(url, text)
        results = [show(extract(tuned, release))]
        if base is not None:
            results.append(show(extract(base, release)))
        return results if len(results) > 1 else results[0]

    outputs = [gr.Code(label="FinSight (fine-tuned)", language="json")]
    if base is not None:
        outputs.append(gr.Code(label="Base Mistral-7B-Instruct", language="json"))
    gr.Interface(
        fn=run,
        inputs=[gr.Textbox(label="sec.gov URL of the release (8-K Exhibit 99.1)"),
                gr.Textbox(label="...or paste the release text", lines=12)],
        outputs=outputs,
        examples=load_examples() or None,
        title="FinSight: earnings release to JSON",
        description=("Mistral-7B fine-tuned with QLoRA to pull a quarter's GAAP revenue, net income, diluted EPS and "
                     "period end out of an earnings release. Research prototype: not investment advice and not a "
                     "trading system."),
        flagging_mode="never",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
