# Short Report: Layer-Adaptive Pyramid SinkKV

## Goal

This project studies training-free KV cache compression for
`EleutherAI/pythia-70m`. The goal is to reduce retained KV cache tokens while
preserving language modeling quality and measuring inference latency.

## Method

The proposed method is `pyramid_sinkkv`, a layer-adaptive extension of the
uniform `sink_snapkv` baseline.

Uniform `sink_snapkv` keeps the same cache layout in every layer:

```text
[sink tokens] + [attention-selected middle tokens] + [recent window]
```

`pyramid_sinkkv` keeps the same three token types, but assigns different budgets
to different layers. For the 6-layer Pythia-70M model, the default profile is:

| Layers | Window | Important |
| --- | ---: | ---: |
| 0-1 | 384 | 48 |
| 2-3 | 256 | 32 |
| 4-5 | 128 | 16 |

With `sink_size=4`, the average nominal budget is 292 tokens per layer, equal
to the uniform `sink_snapkv` budget. The method is training-free and does not
modify model parameters.

## Compared Methods

- `dense`: full KV cache.
- `streamingllm`: sink tokens plus recent window.
- `sink_snapkv`: uniform sink + selected middle + recent window.
- `pyramid_sinkkv`: proposed layer-adaptive SinkKV.
- `reverse_pyramid_sinkkv`: ablation with the layer allocation reversed.

## Experiments

The intended Linux-server experiment suite is:

- Wikitext-2 PPL with 1024 tokens.
- PG-19 single-sample PPL with 1024 tokens.
- PG-19 latency with a 512-token prompt and 64 generated tokens.
- Layer-allocation ablation:
  `sink_snapkv` vs. `pyramid_sinkkv` vs. `reverse_pyramid_sinkkv`.
- Budget sweep:
  `sink_snapkv` vs. `pyramid_sinkkv` under several average budgets.

Run all experiments with:

```bash
bash scripts/run_server_experiments.sh
```

Summarize existing results with:

```bash
bash scripts/summarize_server_results.sh results/server
```

The generated summary is saved to `results/server/summary.md`.

## Expected Evidence

The main claim should be evaluated in one of two ways:

- At similar average KV budget, `pyramid_sinkkv` should improve PPL over
  `sink_snapkv`.
- At similar PPL, `pyramid_sinkkv` should use fewer average retained KV tokens.

The reverse allocation ablation is important. If `reverse_pyramid_sinkkv` is
worse than `pyramid_sinkkv`, it supports the claim that the low-to-high pyramid
allocation is not arbitrary.

## Limitations

- Pythia-70M has only 6 layers, so layer-wise effects may be weaker than on
  larger models.
- The implementation is designed for reproducibility, not optimized serving.
- Per-layer pruning requires attention weights and Python-side cache indexing,
  which can add overhead.
