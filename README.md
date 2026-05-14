# Layer-Adaptive Pyramid SinkKV

This repository is the individual/team course-project code for training-free
KV cache compression on `EleutherAI/pythia-70m`.

The current proposed method is **Layer-Adaptive Pyramid SinkKV**
(`pyramid_sinkkv`). It combines:

- StreamingLLM-style attention sink tokens
- SnapKV-style attention-selected middle tokens
- PyramidKV-style layer-wise cache budgets

Unlike uniform KV compression methods that keep the same token indices in every
Transformer layer, `pyramid_sinkkv` prunes each layer independently:

```text
lower layers:  sink + larger recent window + more selected tokens
middle layers: sink + medium recent window + medium selected tokens
higher layers: sink + smaller recent window + fewer selected tokens
```

For Pythia-70M's 6 layers, the default budget profile is:

| Layers | Window | Important |
| --- | ---: | ---: |
| 0-1 | 384 | 48 |
| 2-3 | 256 | 32 |
| 4-5 | 128 | 16 |

With `sink_size=4`, the average per-layer nominal budget is still 292 tokens,
matching the uniform `sink_snapkv` baseline:

```text
sink_snapkv:      every layer uses 4 + 256 + 32 = 292 tokens
pyramid_sinkkv:   average over layers is also 292 tokens
```

This makes the main comparison fair: similar average KV budget, different
layer-wise allocation.

## Main Methods

- `dense`: keep the full KV cache.
- `streamingllm`: keep sink tokens plus the recent window.
- `snapkv`: keep attention-selected tokens plus the recent window.
- `sink_snapkv`: uniform baseline that keeps sink tokens, selected middle
  tokens, and the recent window in every layer.
- `pyramid_sinkkv`: proposed method with per-layer SinkKV budgets.
- `reverse_pyramid_sinkkv`: ablation that gives small budgets to lower layers
  and large budgets to higher layers.

Additional exploratory baselines (`sliding_window`, `h2o`, `scissorhands`,
`tova`, `pyramidkv`, `lm_infinite`) remain available from the CLI, but the main
paper comparison should focus on `dense`, `streamingllm`, `sink_snapkv`, and
`pyramid_sinkkv`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

The local development environment used:

```text
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

For PG-19, the assignment allows using one long sample. The local experiments
use:

```text
data/pg19_raw/test/10146.txt
```

On a Linux server, either copy this file to the same path or set:

```bash
export PG19_TEXT=/path/to/10146.txt
```

If the file is missing, the server script will fall back to
`datasets.load_dataset("pg19", split="test")` and use one sample.

## One-Command Linux Server Experiments

After copying the repository to the remote Linux server and activating the
environment, run:

```bash
bash scripts/run_server_experiments.sh
```

This runs:

- PG-19 main PPL
- Wikitext-2 main PPL
- PG-19 full-method PPL for all current methods
- Wikitext-2 full-method PPL for all current methods
- layer-allocation ablation:
  `sink_snapkv`, `pyramid_sinkkv`, `reverse_pyramid_sinkkv`
- PG-19 budget sweep for `sink_snapkv` vs. `pyramid_sinkkv`
- GPU latency, unless `SKIP_LATENCY=1`
- automatic Markdown summary

Useful environment variables:

```bash
export DEVICE=cuda
export DTYPE=float16
export LATENCY_DEVICE=cuda
export LATENCY_DTYPE=float16
export RESULT_DIR=results/server
export PG19_TEXT=/path/to/10146.txt
export WIKITEXT_TEXT=/path/to/wikitext_validation.txt
export SKIP_LATENCY=0
export RUN_BUDGET_SWEEP=1
export RUN_FULL_SUITE=1
export RUN_FULL_LATENCY=1
```

Then run:

```bash
bash scripts/run_server_experiments.sh
```

The summary is written to:

```text
results/server/summary.md
```

The full-method suite includes:

```text
dense
sliding_window
streamingllm
lm_infinite
h2o
scissorhands
tova
snapkv
pyramidkv
sink_snapkv
pyramid_sinkkv
reverse_pyramid_sinkkv
```

To summarize existing results without rerunning:

```bash
bash scripts/summarize_server_results.sh results/server
```

## Individual Commands

PG-19 main PPL:

```bash
python scripts/run_ppl.py \
  --model EleutherAI/pythia-70m \
  --text-file data/pg19_raw/test/10146.txt \
  --max-tokens 1024 \
  --methods dense streamingllm sink_snapkv pyramid_sinkkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --device cuda \
  --dtype float16 \
  --output results/server/ppl_pg19_main.json
```

Wikitext-2 main PPL:

```bash
python scripts/run_ppl.py \
  --model EleutherAI/pythia-70m \
  --dataset wikitext \
  --split validation \
  --max-tokens 1024 \
  --methods dense streamingllm sink_snapkv pyramid_sinkkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --device cuda \
  --dtype float16 \
  --output results/server/ppl_wikitext_main.json
```

GPU latency:

```bash
python scripts/run_latency.py \
  --model EleutherAI/pythia-70m \
  --text-file data/pg19_raw/test/10146.txt \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --methods dense streamingllm sink_snapkv pyramid_sinkkv \
  --window-size 256 \
  --sink-size 4 \
  --important-size 32 \
  --device cuda \
  --dtype float16 \
  --output results/server/latency_pg19_main.json
```

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

## Notes

- `pyramid_sinkkv` uses per-layer KV pruning, so different layers may retain
  different sequence lengths.
- The implementation is intentionally clear and reproducible, not a custom
  CUDA serving engine.
- The latency script performs a small CUDA warmup before timing.
- `reverse_pyramid_sinkkv` is an ablation, not the proposed method.

## References

- StreamingLLM: Efficient Streaming Language Models with Attention Sinks.
- SnapKV: LLM Knows What You are Looking for Before Generation.
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information
  Funneling.
