# Short Report: KV Cache Compression Baselines

## Goal

This personal codebase studies training-free KV cache compression for
`EleutherAI/pythia-70m`. The goal is to reproduce representative cache
retention policies and compare their quality, retained-token count, and
generation latency under a shared evaluation pipeline.

## Implemented Methods

- `dense`: full KV cache.
- `sliding_window`: recent-window retention.
- `streamingllm`: initial sink tokens plus recent window.
- `lm_infinite`: sink-token and local-window retention.
- `h2o`: cumulative attention heavy-hitter retention.
- `scissorhands`: persistent attention-based retention.
- `tova`: current token plus a small high-attention history set.
- `snapkv`: attention-selected middle tokens plus recent window.
- `pyramidkv`: layer-weighted attention importance summary.
- `sink_snapkv`: sink tokens, attention-selected middle tokens, and recent
  window.

## Experiment Suite

The repository provides scripts for:

- Wikitext-2 PPL with 1024 tokens.
- PG-19 single-sample PPL with 1024 tokens.
- PG-19 latency with a 512-token prompt and 64 generated tokens.
- Full-method comparison across the implemented baseline policies.

Run the Linux suite with:

```bash
bash scripts/run_server_experiments.sh
```

Summarize existing results with:

```bash
bash scripts/summarize_server_results.sh results/server
```

## Results

Detailed baseline result tables are saved in
`results/baseline_summary.md`.

| Dataset | dense | streamingllm | snapkv | pyramidkv | sink_snapkv |
| --- | ---: | ---: | ---: | ---: | ---: |
| PG-19 PPL | 31.10 | 31.43 | 31.23 | 31.30 | 31.23 |
| Wikitext-2 PPL | 30.15 | 36.19 | 36.16 | 35.58 | 35.68 |

For PG-19 latency with a 512-token prompt and 64 generated tokens,
`streamingllm` reduces peak CUDA memory from 165.62 MB to 154.56 MB, while
`sink_snapkv` retains a larger selected-token budget and obtains 155.71 MB peak
memory in this run.

## Notes

This repository is intentionally scoped to baseline reproduction and shared
evaluation utilities.
