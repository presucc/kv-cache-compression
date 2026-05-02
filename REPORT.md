# Short Report: ASW-KV for Training-Free Inference Acceleration

## Goal

This project studies training-free KV cache compression for
`EleutherAI/pythia-70m`. The goal is to reduce retained KV cache tokens while
preserving language modeling quality and measuring inference latency.

## Compared methods

- `dense`: full KV cache, used as the quality baseline.
- `sliding_window`: keeps only recent tokens.
- `streamingllm`: keeps attention sink tokens plus recent tokens.
- `lm_infinite`: keeps initial tokens plus recent tokens.
- `h2o`: keeps cumulative attention heavy hitters plus recent tokens.
- `scissorhands`: keeps tokens with persistent high attention plus recent
  tokens.
- `tova`: keeps the current token and highest-attended historical tokens.
- `snapkv`: keeps current-attention-selected middle tokens plus recent tokens.
- `pyramidkv`: keeps sink, recent, and layer-weighted attention-selected
  middle tokens.
- `asw_kv`: keeps attention sink tokens, attention-selected middle tokens, and
  recent tokens.

## Proposed extension

ASW-KV extends StreamingLLM with a small attention-guided middle memory. At each
decoding step, it computes the average attention paid by the current token to
retained historical tokens. The middle-context tokens with the highest attention
scores are kept together with sink and recent tokens.

This gives the cache layout:

```text
[sink tokens] + [important middle tokens] + [recent window]
```

The method does not train or modify model parameters.

## Experimental setup

- Model: `EleutherAI/pythia-70m`
- Datasets: Wikitext-2 and PG-19 single long sample
- Sequence length: 1024 tokens for PPL experiments
- PPL device/precision: CPU, float32
- Latency device/precision: NVIDIA GeForce RTX 4080 Laptop GPU, float16
- Cache parameters:
  - `sink_size`: 4
  - `window_size`: 256
  - `important_size`: 32

## Perplexity results

| Dataset | Method | PPL | Max KV tokens | Avg KV tokens |
| --- | --- | ---: | ---: | ---: |
| Wikitext-2 | dense | 30.15 | 1023 | 512.00 |
| Wikitext-2 | sliding_window | 42.95 | 256 | 224.09 |
| Wikitext-2 | streamingllm | 36.19 | 260 | 227.09 |
| Wikitext-2 | lm_infinite | 36.19 | 260 | 227.09 |
| Wikitext-2 | h2o | 35.51 | 288 | 247.60 |
| Wikitext-2 | scissorhands | 36.44 | 288 | 247.60 |
| Wikitext-2 | tova | 37.36 | 288 | 247.60 |
| Wikitext-2 | snapkv | 36.16 | 288 | 247.60 |
| Wikitext-2 | pyramidkv | 35.58 | 292 | 250.47 |
| Wikitext-2 | asw_kv | 35.68 | 292 | 250.47 |
| PG-19 | dense | 31.10 | 1023 | 512.00 |
| PG-19 | sliding_window | 36.30 | 256 | 224.09 |
| PG-19 | streamingllm | 31.43 | 260 | 227.09 |
| PG-19 | lm_infinite | 31.43 | 260 | 227.09 |
| PG-19 | h2o | 31.39 | 288 | 247.60 |
| PG-19 | scissorhands | 31.43 | 288 | 247.60 |
| PG-19 | tova | 31.37 | 288 | 247.60 |
| PG-19 | snapkv | 31.23 | 288 | 247.60 |
| PG-19 | pyramidkv | 31.30 | 292 | 250.47 |
| PG-19 | asw_kv | 31.23 | 292 | 250.47 |

## Latency results

Latency was measured with a 512-token PG-19 prompt and 64 generated tokens on
the GPU. The implementation uses a clear one-token-at-a-time Python decoding
loop, so these numbers should be read as a reproducible policy comparison
rather than an optimized serving benchmark.

| Method | TTFT (s) | TPOT (ms) | Throughput (tok/s) | Peak memory (MB) |
| --- | ---: | ---: | ---: | ---: |
| dense | 9.38 | 17.87 | 6.09 | 165.62 |
| sliding_window | 7.18 | 8.89 | 8.27 | 154.42 |
| streamingllm | 4.45 | 8.83 | 12.77 | 154.56 |
| lm_infinite | 4.45 | 9.77 | 12.63 | 154.56 |
| h2o | 4.84 | 9.98 | 11.71 | 155.58 |
| scissorhands | 4.69 | 9.49 | 12.10 | 155.58 |
| tova | 4.58 | 11.49 | 12.07 | 155.57 |
| snapkv | 4.76 | 9.54 | 11.95 | 155.57 |
| pyramidkv | 4.81 | 9.16 | 11.88 | 155.71 |
| asw_kv | 4.74 | 8.87 | 12.07 | 155.71 |

## Discussion

Attention sink tokens improve strongly over a pure sliding window. On PG-19,
PPL drops from 36.30 with `sliding_window` to 31.43 with `streamingllm`, nearly
recovering the dense-cache result of 31.10. On Wikitext-2, `streamingllm` also
improves over `sliding_window`, reducing PPL from 42.95 to 36.19.

The additional reproduced methods give useful reference points. H2O-lite uses
cumulative attention scores and is strongest on Wikitext-2, reducing PPL from
36.19 with `streamingllm` to 35.51. SnapKV-lite uses the current query's
attention scores and is strongest on the tested PG-19 sample, reaching 31.23
PPL with a 288-token nominal budget. PyramidKV-lite, implemented with
layer-weighted attention aggregation under a shared cache layout, is also
competitive: 31.30 PPL on PG-19 and 35.58 on Wikitext-2.

Not every method improves every dataset. TOVA-lite is reasonable on PG-19
(31.37) but worse on Wikitext-2 (37.36). This likely happens because TOVA-lite
does not reserve a full recent window; it keeps the newest token and the
highest-attended historical tokens, so low-attention but locally useful tokens
can be dropped. Scissorhands-lite is also weaker than H2O here, suggesting that
running-maximum attention is less stable than cumulative attention for these
short 1024-token tests.

ASW-KV improves the PPL/cache-size trade-off compared with StreamingLLM in both
1024-token experiments. On PG-19, ASW-KV reduces PPL from 31.43 to 31.23 while
still keeping the maximum retained cache far below dense cache size
(292 vs. 1023 tokens). On Wikitext-2, ASW-KV reduces PPL from 36.19 to 35.68.
Compared with SnapKV-lite, ASW-KV adds four attention sink tokens; compared
with H2O-lite, it uses instant attention rather than cumulative attention.

InfLLM and TreeKV are not included in the unified table because they require a
block-retrieval or tree-structured cache design that is outside this simple
one-token-at-a-time retention framework. They are better evaluated with a
separate implementation rather than forced into the same per-token selection
API.

ASW-KV uses attention scores, so its quality gain should be weighed against the
cost of requesting attention weights. The current implementation favors clarity
and reproducibility; an optimized version could update importance scores less
frequently or reuse cached importance statistics.

On the GPU latency smoke benchmark, compressed-cache methods reduce peak memory
relative to dense cache. The latency script is intentionally simple, so the
absolute timings can vary between runs, but the compressed methods consistently
retain fewer KV tokens and lower peak CUDA memory than dense cache.

## Ablation: middle-token memory size

This ablation fixes PG-19, 1024 tokens, `sink_size=4`, and `window_size=256`,
then varies the ASW-KV middle-token budget.

| important_size | PPL | Max KV tokens | Avg KV tokens |
| ---: | ---: | ---: | ---: |
| 0 | 31.43 | 260 | 227.09 |
| 16 | 31.39 | 276 | 238.90 |
| 32 | 31.23 | 292 | 250.47 |
| 64 | 31.15 | 324 | 272.85 |

Increasing the attention-selected middle-token budget monotonically reduces PPL
in this setting. This supports the main ASW-KV hypothesis: under a fixed
sink-plus-window cache structure, retaining a small number of attention-relevant
middle-context tokens can recover additional language modeling quality.

## Limitations

- The implementation is optimized for clarity and reproducibility, not custom
  CUDA kernels.
- Latency on short contexts can be dominated by Python overhead.
- ASW-KV attention-score collection may reduce speed unless optimized or updated
  less frequently.

## How to reproduce

See `README.md` for exact commands.
