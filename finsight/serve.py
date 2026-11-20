"""A small HTTP API in front of the model: earnings release in, JSON figures and timing out.

Two backends, one API:

  vllm          production path. vLLM serves the base model quantized to FP8 with
                the FinSight LoRA adapter on top (scripts/serve_vllm.sh); this API
                calls its OpenAI-compatible endpoint. Continuous batching is what
                makes many releases at once cheap.
  transformers  fallback that loads the model in-process, 4-bit NF4, with
                finsight.inference. One request at a time, but it runs anywhere
                the training notebook runs.

    bash scripts/serve_vllm.sh outputs/finsight-earnings/final                 # vLLM on :8000
    export SEC_USER_AGENT="Your Name your.email@example.com"                   # only needed for "url"
    python -m finsight.serve --backend vllm --vllm-url http://localhost:8000   # API on :8080

    curl -s localhost:8080/v1/extract -H 'content-type: application/json' \\
         -d '{"url": "https://www.sec.gov/Archives/edgar/data/.../ex99-1.htm"}'

Send either "text" (the release) or "url" (an sec.gov address; other hosts are
refused so the server cannot be pointed at arbitrary sites). Every request is
logged as one JSON line for finsight.monitor; release text is not logged unless
--log-content is set.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from . import earnings
from .edgar import EXCERPT_CHARS, is_sec_url, release_excerpt

MAX_TEXT_CHARS = 200_000   # whole releases are accepted; the model reads release_excerpt() of them


class ExtractRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    url: str | None = Field(default=None, max_length=500)


class ExtractResponse(BaseModel):
    fields: dict | None
    valid_json: bool
    raw_response: str
    timing_ms: dict
    model: str
    source: str


class VLLMBackend:
    """Calls a vLLM OpenAI-compatible server. `model` is the served LoRA name (or the base model's)."""

    def __init__(self, base_url: str, model: str = "finsight", max_tokens: int = 128, timeout: float = 60,
                 transport=None, api_key: str | None = None):
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):  # hosted OpenAI-compatible endpoints are often given with /v1 on the end
            base_url = base_url[:-3]
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                   headers=headers)
        self.model, self.max_tokens = model, max_tokens

    def complete(self, messages: list[dict]) -> dict:
        r = self.client.post("/v1/chat/completions", json={
            "model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens})
        r.raise_for_status()
        body = r.json()
        return {"response": body["choices"][0]["message"]["content"].strip(),
                "input_tokens": body.get("usage", {}).get("prompt_tokens"),
                "output_tokens": body.get("usage", {}).get("completion_tokens")}

    def ready(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except Exception:
            return False


class TransformersBackend:
    def __init__(self, model_id: str, adapter: str | None, max_tokens: int = 128):
        from . import inference

        self.inference = inference
        self.model, self.tokenizer = inference.load(model_id, adapter)
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict]) -> dict:
        res = self.inference.generate_with_stats(self.model, self.tokenizer, [messages], self.max_tokens, 1)[0]
        return {k: res[k] for k in ("response", "input_tokens", "output_tokens")}

    def ready(self) -> bool:
        return True


def extract(backend, text: str, max_chars: int = EXCERPT_CHARS) -> dict:
    """Run one release through the model. Shared by the API and the demo."""
    excerpt = release_excerpt(text, max_chars)
    start = time.perf_counter()
    out = backend.complete(earnings.build_messages(excerpt))
    model_ms = 1000 * (time.perf_counter() - start)
    fields = earnings.parse_output(out["response"])
    if fields is not None:
        fields = {k: fields.get(k) for k in earnings.FIELDS}
    return {"fields": fields, "valid_json": fields is not None, "raw_response": out["response"],
            "model_ms": round(model_ms, 1), "input_tokens": out.get("input_tokens"),
            "output_tokens": out.get("output_tokens"), "excerpt_chars": len(excerpt)}


def create_app(backend, log_path: Path | None = None, log_content: bool = False, model_name: str = "finsight",
               fetch_url=None):
    """fetch_url(url) -> release text; defaults to the cached SEC client (needs SEC_USER_AGENT)."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="FinSight", version="0.3")
    state = {"fetch": fetch_url}

    def log(record: dict) -> None:
        if log_path is None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def fetch(url: str) -> str:
        if state["fetch"] is None:
            from .edgar import SECClient

            state["fetch"] = SECClient().release_text
        return state["fetch"](url)

    @app.get("/health")
    def health():
        if not backend.ready():
            raise HTTPException(503, "model backend not ready")
        return {"status": "ok", "model": model_name}

    @app.post("/v1/extract", response_model=ExtractResponse)
    def extract_endpoint(req: ExtractRequest):
        if (req.text is None) == (req.url is None):
            raise HTTPException(422, "send exactly one of 'text' or 'url'")
        t0 = time.perf_counter()
        record = {"ts": datetime.now(timezone.utc).isoformat(), "model": model_name,
                  "source": "url" if req.url else "text"}
        text = req.text
        if req.url:
            if not is_sec_url(req.url):
                raise HTTPException(422, "only https://www.sec.gov/ or data.sec.gov URLs are fetched")
            try:
                text = fetch(req.url)
            except Exception as e:
                record.update(error="fetch_" + type(e).__name__)
                log(record)
                raise HTTPException(502, "could not fetch the release from sec.gov") from None
        fetch_ms = 1000 * (time.perf_counter() - t0)
        record.update(release_sha1=hashlib.sha1(text.encode()).hexdigest()[:12], context_chars=len(text))
        try:
            out = extract(backend, text)
        except Exception as e:  # backend down, timeout, OOM: log it and return 502, never a made-up answer
            record.update(error=type(e).__name__, latency_ms=round(1000 * (time.perf_counter() - t0), 1))
            log(record)
            raise HTTPException(502, "model backend error") from None
        total_ms = 1000 * (time.perf_counter() - t0)
        record.update(latency_ms=round(total_ms, 1), model_ms=out["model_ms"], fetch_ms=round(fetch_ms, 1),
                      format_ok=out["valid_json"], input_tokens=out["input_tokens"],
                      output_tokens=out["output_tokens"],
                      null_fields=[k for k, v in (out["fields"] or {}).items() if v is None])
        if log_content:
            record.update(url=req.url, response=out["raw_response"])
        log(record)
        return ExtractResponse(fields=out["fields"], valid_json=out["valid_json"], raw_response=out["raw_response"],
                               timing_ms={"fetch": round(fetch_ms, 1), "model": out["model_ms"],
                                          "total": round(total_ms, 1)},
                               model=model_name, source=record["source"])

    return app


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--served-model", default="finsight", help="LoRA name given to vLLM's --lora-modules")
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--adapter", default="outputs/finsight-earnings/final")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--log", default="logs/requests.jsonl")
    ap.add_argument("--log-content", action="store_true", help="also log URLs and model output")
    args = ap.parse_args()

    import uvicorn

    if args.backend == "vllm":
        backend, name = VLLMBackend(args.vllm_url, args.served_model), args.served_model
    else:
        backend, name = TransformersBackend(args.model_id, args.adapter), "finsight-transformers"
    uvicorn.run(create_app(backend, Path(args.log), args.log_content, name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
