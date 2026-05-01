from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.ipc as ipc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a cached Wikitext Arrow split to a plain text file.")
    parser.add_argument("--arrow-file", required=True)
    parser.add_argument("--output", default="data/wikitext_validation.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.arrow_file)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    reader = ipc.open_stream(str(source))
    for batch in reader:
        for item in batch.column("text").to_pylist():
            if item and item.strip():
                chunks.append(item.strip())

    output.write_text("\n\n".join(chunks), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes, {len(chunks)} non-empty rows)")


if __name__ == "__main__":
    main()
