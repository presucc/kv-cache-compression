# Baseline Experiment Summary

The following tables summarize the local baseline experiments on
`EleutherAI/pythia-70m`. PPL experiments use 1024 evaluation tokens. The latency
experiment uses a 512-token PG-19 prompt and generates 64 tokens.

## PG-19 PPL

| Method | PPL | Max KV Tokens | Avg KV Tokens | Nominal Budget |
| --- | ---: | ---: | ---: | ---: |
| dense | 31.10 | 1023 | 512.00 | - |
| sliding_window | 36.30 | 256 | 224.09 | 256 |
| streamingllm | 31.43 | 260 | 227.09 | 260 |
| lm_infinite | 31.43 | 260 | 227.09 | 260 |
| h2o | 31.39 | 288 | 247.60 | 288 |
| scissorhands | 31.43 | 288 | 247.60 | 288 |
| tova | 44.91 | 33 | 32.48 | 33 |
| snapkv | 31.23 | 288 | 247.60 | 288 |
| pyramidkv | 31.30 | 292 | 250.47 | 292 |
| sink_snapkv | 31.23 | 292 | 250.47 | 292 |

## Wikitext-2 PPL

| Method | PPL | Max KV Tokens | Avg KV Tokens | Nominal Budget |
| --- | ---: | ---: | ---: | ---: |
| dense | 30.15 | 1023 | 512.00 | - |
| sliding_window | 42.95 | 256 | 224.09 | 256 |
| streamingllm | 36.19 | 260 | 227.09 | 260 |
| lm_infinite | 36.19 | 260 | 227.09 | 260 |
| h2o | 35.51 | 288 | 247.60 | 288 |
| scissorhands | 36.44 | 288 | 247.60 | 288 |
| tova | 65.50 | 33 | 32.48 | 33 |
| snapkv | 36.16 | 288 | 247.60 | 288 |
| pyramidkv | 35.58 | 292 | 250.47 | 292 |
| sink_snapkv | 35.68 | 292 | 250.47 | 292 |

## PG-19 Latency

| Method | TTFT (s) | TPOT (s) | Throughput (tok/s) | Peak CUDA Memory (MB) | Max KV Tokens | Avg KV Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 3.85 | 0.0080 | 14.69 | 165.62 | 575 | 288.00 |
| streamingllm | 4.01 | 0.0088 | 14.02 | 154.56 | 260 | 201.44 |
| sink_snapkv | 4.75 | 0.0092 | 12.01 | 155.71 | 292 | 218.11 |

## Observations

- On PG-19, `snapkv` and `sink_snapkv` are close to dense PPL while retaining
  substantially fewer KV tokens.
- On Wikitext-2, `h2o`, `pyramidkv`, and `sink_snapkv` are the strongest
  compressed baselines in this run.
- `tova` uses the smallest budget and shows the largest PPL degradation.
- The latency results are intended for relative comparison under the provided
  one-token-at-a-time evaluation script, not as optimized serving benchmarks.
