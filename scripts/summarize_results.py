from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a Markdown table from result JSON files.")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--columns", nargs="+", default=["method", "ppl", "max_retained_tokens", "average_retained_tokens"])
    return parser.parse_args()


def fmt(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = []
    for file_name in args.files:
        data = json.loads(Path(file_name).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        rows.extend(data)

    print("| " + " | ".join(args.columns) + " |")
    print("| " + " | ".join("---" for _ in args.columns) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(column, "")) for column in args.columns) + " |")


if __name__ == "__main__":
    main()
