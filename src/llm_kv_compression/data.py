from __future__ import annotations

from pathlib import Path


def load_text_corpus(
    dataset: str,
    split: str,
    max_samples: int,
    max_chars: int,
    text_file: str | None = None,
) -> str:
    if text_file:
        return Path(text_file).read_text(encoding="utf-8")[:max_chars]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the datasets package or pass --text-file.") from exc

    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset == "pg19":
        ds = load_dataset("pg19", split=split)
    else:
        raise ValueError("dataset must be one of: wikitext, pg19")

    chunks = []
    for row in ds:
        text = row.get("text", "")
        if text and text.strip():
            chunks.append(text.strip())
        if len(chunks) >= max_samples or sum(len(x) for x in chunks) >= max_chars:
            break

    if not chunks:
        raise RuntimeError(f"No non-empty text found in dataset={dataset!r}, split={split!r}.")
    return "\n\n".join(chunks)[:max_chars]
