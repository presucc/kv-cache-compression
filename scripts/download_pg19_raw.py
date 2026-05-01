from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm


GCS_ROOT = "https://storage.googleapis.com/deepmind-gutenberg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw PG-19 text files with resume/retry support.")
    parser.add_argument("--output-dir", default="data/pg19_raw")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--repo-id", default="pg19")
    parser.add_argument("--hf-endpoint", default=None, help="Example: https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per split for smoke tests.")
    return parser.parse_args()


def load_split_files(repo_id: str, split: str, hf_endpoint: str | None) -> list[str]:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"data/{split}_files.txt",
    )
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def remote_size(url: str, timeout: int) -> int | None:
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        length = response.headers.get("content-length")
        return int(length) if length else None
    except Exception:
        return None


def download_one(relative_path: str, output_dir: Path, retries: int, timeout: int) -> tuple[str, str, int]:
    url = f"{GCS_ROOT}/{relative_path}"
    target = output_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    expected_size = remote_size(url, timeout)

    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return relative_path, "skipped", target.stat().st_size

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            mode = "ab" if existing else "wb"
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                if response.status_code == 416:
                    tmp.replace(target)
                    return relative_path, "downloaded", target.stat().st_size
                response.raise_for_status()
                if existing and response.status_code == 200:
                    mode = "wb"
                with tmp.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            handle.write(chunk)

            if expected_size is not None and tmp.stat().st_size != expected_size:
                raise RuntimeError(f"incomplete download: {tmp.stat().st_size}/{expected_size}")
            tmp.replace(target)
            return relative_path, "downloaded", target.stat().st_size
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 2**attempt))

    return relative_path, f"failed: {last_error!r}", 0


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[str] = []
    for split in args.splits:
        split_files = load_split_files(args.repo_id, split, args.hf_endpoint)
        if args.limit is not None:
            split_files = split_files[: args.limit]
        all_files.extend(split_files)
        print(f"{split}: {len(split_files)} files")

    downloaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    bytes_ready = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, path, output_dir, args.retries, args.timeout)
            for path in all_files
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="PG-19"):
            path, status, size = future.result()
            bytes_ready += size
            if status == "skipped":
                skipped += 1
            elif status == "downloaded":
                downloaded += 1
            else:
                failed.append((path, status))

    print(f"downloaded={downloaded} skipped={skipped} failed={len(failed)} bytes_ready={bytes_ready}")
    if failed:
        failed_path = output_dir / "failed.txt"
        failed_path.write_text(
            "\n".join(f"{path}\t{status}" for path, status in failed),
            encoding="utf-8",
        )
        print(f"failed list: {failed_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
