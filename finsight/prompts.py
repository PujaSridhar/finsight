"""Prompt format shared by data building, training, evaluation and the demo.

Keeping one prompt builder for every phase matters: the base model and the
fine-tuned model are evaluated with exactly the prompt the adapter was
trained on, so the comparison is fair.

Mistral-7B-Instruct's chat template has no system role, so the instructions
live at the top of the user turn.
"""

from __future__ import annotations

INSTRUCTION = (
    "You are a financial analyst. Answer the question using only the "
    "financial document excerpt below. If a calculation is needed, show each "
    "step on its own line. End your response with a final line in the form "
    "'Answer: <value>'."
)


def table_to_markdown(rows: list[list[str]] | None, max_rows: int = 60) -> str:
    """Render a table (list of rows) as a GitHub-style markdown table."""
    if not rows:
        return ""
    rows = [[str(c).replace("|", "/").replace("\n", " ").strip() for c in r] for r in rows[:max_rows]]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def build_context(text_before: str = "", table: list[list[str]] | None = None, text_after: str = "") -> str:
    parts = [p.strip() for p in (text_before, table_to_markdown(table), text_after) if p and p.strip()]
    return "\n\n".join(parts)


def build_user_message(context: str, question: str) -> str:
    return f"{INSTRUCTION}\n\n### Document excerpt\n{context.strip()}\n\n### Question\n{question.strip()}"


def build_messages(context: str, question: str, response: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": build_user_message(context, question)}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
