# KV Cache Compression Baselines

This repository contains the personal reproducible codebase for training-free
KV cache compression experiments on `EleutherAI/pythia-70m`.

The focus of this repository is baseline reproduction and fair comparison. It
implements several cache retention policies under one evaluation pipeline:

- `dense`: keep the full KV cache.
- `sliding_window`: keep only the most recent tokens.
- `streamingllm`: keep initial sink tokens plus a recent window.
- `lm_infinite`: sink-token and local-window retention.
- `h2o`: cumulative attention-based heavy-hitter retention.
- `scissorhands`: persistent attention-based token retention.
- `tova`: keep the current token plus a small set of high-attention tokens.
- `snapkv`: attention-selected middle tokens plus a recent window.
- `pyramidkv`: attention selection with layer-weighted importance summary.
- `sink_snapkv`: sink tokens plus selected middle tokens plus recent window.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

The local development environment used:

```text
python 3.12.4
torch 2.11.0+cu128
transformers 4.57.1
datasets 3.6.0
EleutherAI/pythia-70m
```

If Hugging Face access is slow in China, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Data

For Wikitext-2, the scripts can load the dataset through Hugging Face:

```bash
python scripts/run_ppl.py --dataset wikitext --split validation
```

For PG-19, the experiments can use one long sample:

```text
data/pg19_raw/test/10146.txt
```

If this file is unavailable, set:

```bash
export PG19_TEXT=/path/to/10146.txt
```

## Running Experiments

PG-19 PPL:

```bash
python scripts/run_ppl.py \
  --model EleutherAI/pythia-70m \
  --text-file data/pg19_raw/test/10146.txt \
  --max-tokens 1024 \
  --methods dense streamingllm sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --device cuda \
  --dtype float32 \
  --output results/ppl_pg19_main.json
```

Wikitext-2 PPL:

```bash
python scripts/run_ppl.py \
  --model EleutherAI/pythia-70m \
  --dataset wikitext \
  --split validation \
  --max-tokens 1024 \
  --methods dense streamingllm sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --device cuda \
  --dtype float32 \
  --output results/ppl_wikitext_main.json
```

PG-19 latency:

```bash
python scripts/run_latency.py \
  --model EleutherAI/pythia-70m \
  --text-file data/pg19_raw/test/10146.txt \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --methods dense streamingllm sink_snapkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --device cuda \
  --dtype float16 \
  --output results/latency_pg19_main.json
```

Run a broader Linux suite:

```bash
bash scripts/run_server_experiments.sh
```

Summarize existing results:

```bash
bash scripts/summarize_server_results.sh results/server
```

## Baseline Results

The local baseline tables are included in
[results/baseline_summary.md](results/baseline_summary.md).

Main 1024-token PPL results:

| Dataset | dense | streamingllm | snapkv | pyramidkv | sink_snapkv |
| --- | ---: | ---: | ---: | ---: | ---: |
| PG-19 | 31.10 | 31.43 | 31.23 | 31.30 | 31.23 |
| Wikitext-2 | 30.15 | 36.19 | 36.16 | 35.58 | 35.68 |

PG-19 latency with a 512-token prompt and 64 generated tokens:

| Method | TTFT (s) | TPOT (s) | Throughput (tok/s) | Peak CUDA Memory (MB) |
| --- | ---: | ---: | ---: | ---: |
| dense | 3.85 | 0.0080 | 14.69 | 165.62 |
| streamingllm | 4.01 | 0.0088 | 14.02 | 154.56 |
| sink_snapkv | 4.75 | 0.0092 | 12.01 | 155.71 |

## Repository Structure

```text
src/llm_kv_compression/
  cache.py          KV cache retention policies and runtime wrapper
  evaluation.py     PPL and latency evaluation logic
  modeling.py       model/tokenizer loading helpers
  data.py           dataset/text loading helpers
scripts/
  run_ppl.py
  run_latency.py
  run_server_experiments.sh
  summarize_server_results.sh
  summarize_results.py
tests/
  test_cache.py
```

## Tests

```bash
python -m pytest tests
```
