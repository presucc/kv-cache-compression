# ASW-KV: Adaptive Sink-Window KV Cache Compression

This repository is the individual course-project implementation for efficient
language model inference. It studies training-free KV cache compression on
`EleutherAI/pythia-70m`, including several reproduced baselines and one
proposed extension.

The proposed method, **ASW-KV**, extends StreamingLLM by keeping a small
attention-guided memory of useful middle-context tokens:

```text
[attention sink tokens] + [important middle tokens] + [recent window]
```

All methods are inference-time only. No model parameters are trained or changed.

## Main Results

PPL experiments use 1024 tokens, `sink_size=4`, `window_size=256`, and
`important_size=32`.

| Dataset | Method | PPL | Max KV tokens | Avg KV tokens |
| --- | --- | ---: | ---: | ---: |
| PG-19 | dense | 31.10 | 1023 | 512.00 |
| PG-19 | sliding_window | 36.30 | 256 | 224.09 |
| PG-19 | streamingllm | 31.43 | 260 | 227.09 |
| PG-19 | h2o | 31.39 | 288 | 247.60 |
| PG-19 | scissorhands | 31.43 | 288 | 247.60 |
| PG-19 | tova | 44.91 | 33 | 32.48 |
| PG-19 | snapkv | 31.23 | 288 | 247.60 |
| PG-19 | pyramidkv | 31.30 | 292 | 250.47 |
| PG-19 | asw_kv | 31.23 | 292 | 250.47 |
| Wikitext-2 | dense | 30.15 | 1023 | 512.00 |
| Wikitext-2 | sliding_window | 42.95 | 256 | 224.09 |
| Wikitext-2 | streamingllm | 36.19 | 260 | 227.09 |
| Wikitext-2 | h2o | 35.51 | 288 | 247.60 |
| Wikitext-2 | scissorhands | 36.44 | 288 | 247.60 |
| Wikitext-2 | tova | 65.50 | 33 | 32.48 |
| Wikitext-2 | snapkv | 36.16 | 288 | 247.60 |
| Wikitext-2 | pyramidkv | 35.58 | 292 | 250.47 |
| Wikitext-2 | asw_kv | 35.68 | 292 | 250.47 |

The added baselines make the trade-off clearer: H2O-lite is strongest on
Wikitext-2, SnapKV-lite is strongest on this PG-19 sample, PyramidKV-lite is
also competitive, and ASW-KV remains competitive while explicitly combining
attention sinks, selected middle tokens, and the recent window. TOVA-lite uses
a much smaller 33-token budget, so it saves the most cache but loses substantial
quality. All compressed methods keep the maximum KV cache far below the dense
baseline.

## Ablation

PG-19 ablation with 1024 tokens, `sink_size=4`, and `window_size=256`:

| important_size | PPL | Max KV tokens | Avg KV tokens |
| ---: | ---: | ---: | ---: |
| 0 | 31.43 | 260 | 227.09 |
| 16 | 31.39 | 276 | 238.90 |
| 32 | 31.23 | 292 | 250.47 |
| 64 | 31.15 | 324 | 272.85 |

The monotonic PPL decrease supports the main ASW-KV hypothesis: retaining a
small number of attention-selected middle tokens recovers useful context beyond
the sink-plus-window cache.

## Latency

GPU latency was measured on an NVIDIA GeForce RTX 4080 Laptop GPU with a
512-token PG-19 prompt and 64 generated tokens.

| Method | TTFT (s) | TPOT (ms) | Throughput (tok/s) | Peak memory (MB) |
| --- | ---: | ---: | ---: | ---: |
| dense | 4.17 | 10.62 | 13.23 | 165.62 |
| sliding_window | 5.12 | 9.64 | 11.18 | 154.42 |
| streamingllm | 4.66 | 8.76 | 12.29 | 154.56 |
| h2o | 5.27 | 10.42 | 10.80 | 155.58 |
| scissorhands | 5.13 | 10.50 | 11.05 | 155.58 |
| tova | 5.19 | 9.79 | 11.03 | 146.58 |
| snapkv | 5.00 | 11.27 | 11.20 | 155.57 |
| pyramidkv | 4.94 | 12.36 | 11.20 | 155.71 |
| asw_kv | 6.78 | 9.78 | 8.65 | 155.71 |

The latency script performs an unmeasured CUDA warmup before timing. The
implementation still uses a clear one-token-at-a-time Python decoding loop, so
these numbers should be read as reproducible policy comparisons rather than an
optimized serving benchmark.

## Methods

- `dense`: keep the full KV cache.
- `sliding_window`: keep only the most recent tokens.
- `streamingllm`: keep attention sink tokens plus the recent window.
- `lm_infinite`: available as an alias-style lightweight implementation of
  initial tokens plus local window. It matches the StreamingLLM cache layout in
  this framework, so it is not listed as an independent main-table result.
- `h2o`: keep cumulative attention heavy hitters plus the recent window.
- `scissorhands`: keep tokens with persistent high historical attention plus
  the recent window.
- `tova`: keep the current token and `important_size` highest-attended
  historical tokens.
- `snapkv`: keep current-attention-selected middle tokens plus the recent
  window.
- `pyramidkv`: keep sink, recent, and layer-weighted attention-selected middle
  tokens.
- `asw_kv`: keep attention sink tokens, attention-selected middle tokens, and
  the recent window.

For `h2o`, token importance is the cumulative attention mass received over
time. For `scissorhands`, importance is the running maximum attention score.
For `snapkv`, `pyramidkv`, and `asw_kv`, middle-token importance is computed
from the average attention paid by the current query to retained historical
tokens. PyramidKV-lite gives deeper layers larger aggregation weights. The top
`important_size` eligible tokens are kept.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

The experiments in `REPORT.md` were run with:

```text
torch 2.11.0+cu128
transformers 4.57.1
datasets 3.6.0
EleutherAI/pythia-70m
```

If Hugging Face access is slow, set:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## Data

Wikitext-2 can be loaded through Hugging Face Datasets:

```powershell
python scripts\run_ppl.py --dataset wikitext --split validation
```

The checked experiments use a local text export. If needed, create it from the
cached Arrow file with:

```powershell
python scripts\export_wikitext_text.py `
  --arrow-file C:\Users\<you>\.cache\huggingface\datasets\wikitext\wikitext-2-raw-v1\0.0.0\<hash>\wikitext-validation.arrow `
  --output data/wikitext_validation.txt
```

For PG-19, the assignment allows testing on a single long sample. This repo's
commands use:

```text
data/pg19_raw/test/10146.txt
```

Full PG-19 raw files can be downloaded with:

```powershell
python scripts\download_pg19_raw.py `
  --hf-endpoint https://hf-mirror.com `
  --output-dir data/pg19_raw `
  --splits train validation test `
  --workers 8 `
  --retries 10 `
  --timeout 90
```

Large data files are ignored by Git.
Small JSON result files under `results/` are kept as reproducibility evidence.

## Reproduce Experiments

Run the full experiment suite on Windows:

```powershell
.\scripts\run_all_experiments.ps1
```

Or run individual commands.

PG-19 PPL:

```powershell
python scripts\run_ppl.py `
  --model EleutherAI/pythia-70m `
  --text-file data/pg19_raw/test/10146.txt `
  --max-tokens 1024 `
  --methods dense sliding_window streamingllm h2o scissorhands tova snapkv pyramidkv asw_kv `
  --window-size 256 `
  --sink-size 4 `
  --important-size 32 `
  --output results/ppl_pg19_1024_all_kv.json
```

Wikitext-2 PPL:

```powershell
python scripts\run_ppl.py `
  --model EleutherAI/pythia-70m `
  --text-file data/wikitext_validation.txt `
  --max-tokens 1024 `
  --methods dense sliding_window streamingllm h2o scissorhands tova snapkv pyramidkv asw_kv `
  --window-size 256 `
  --sink-size 4 `
  --important-size 32 `
  --output results/ppl_wikitext_1024_all_kv.json
```

GPU latency:

```powershell
python scripts\run_latency.py `
  --model EleutherAI/pythia-70m `
  --text-file data/pg19_raw/test/10146.txt `
  --max-prompt-tokens 512 `
  --max-new-tokens 64 `
  --methods dense sliding_window streamingllm h2o scissorhands tova snapkv pyramidkv asw_kv `
  --window-size 256 `
  --sink-size 4 `
  --important-size 32 `
  --device cuda `
  --dtype float16 `
  --output results/latency_pg19_gpu_512_64_all_kv.json
```

ASW-KV ablation:

```powershell
foreach ($k in 0,16,32,64) {
  python scripts\run_ppl.py `
    --model EleutherAI/pythia-70m `
    --text-file data/pg19_raw/test/10146.txt `
    --max-tokens 1024 `
    --methods asw_kv `
    --window-size 256 `
    --sink-size 4 `
    --important-size $k `
    --output "results/ablation_pg19_asw_important_$k.json"
}
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
  download_pg19_raw.py
  summarize_results.py
  run_all_experiments.ps1
tests/
  test_cache.py
REPORT.md
```

## Notes

- ASW-KV requests attention weights, which adds overhead. The method is meant
  to study quality/cache trade-offs; a production implementation should update
  importance scores less frequently or fuse the bookkeeping.
- InfLLM and TreeKV are not included in the unified benchmark because they need
  block-retrieval or tree-structured cache machinery rather than the shared
  per-token retention API used here.
- The latency script is intentionally simple and reproducible. It is not a
  substitute for optimized serving systems such as vLLM or TensorRT-LLM.
- `torchvision` and `torchaudio` are not required for this text-only project.

## References

- StreamingLLM: Efficient Streaming Language Models with Attention Sinks.
- LM-Infinite: Simple On-the-Fly Length Generalization for Large Language
  Models.
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models.
- Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV
  Cache Compression at Test Time.
- TOVA: Token Omission Via Attention for Efficient KV Cache Compression.
- SnapKV: LLM Knows What You are Looking for Before Generation.
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information
  Funneling.
