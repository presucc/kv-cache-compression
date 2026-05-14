# 中文实验报告：Layer-Adaptive Pyramid SinkKV

## 1. 背景

大语言模型生成文本时会缓存历史 token 的 Key 和 Value，这部分缓存叫做 KV Cache。KV Cache 可以避免重复计算，但上下文越长，缓存越大，显存压力也越高。

本项目研究的问题是：在不训练模型、不修改模型参数的前提下，能不能只保留一部分重要 KV Cache，从而减少显存占用，同时尽量保持模型效果。

## 2. 方法思想

本项目当前采用的创新方法是 `pyramid_sinkkv`，中文可以叫“分层自适应 Pyramid SinkKV”。

它结合了三个已有方向：

- StreamingLLM：保留开头的 attention sink tokens。
- SnapKV：用 attention 选择重要的中间 tokens。
- PyramidKV：不同层使用不同 KV cache budget。

普通的 `sink_snapkv` baseline 每一层都使用相同预算：

```text
[sink tokens] + [attention-selected middle tokens] + [recent window]
```

而 `pyramid_sinkkv` 的核心区别是：不同层单独选择保留的 tokens，并且使用不同预算。

## 3. 分层预算设计

Pythia-70M 有 6 层。默认设置把它分成三组：

| 层 | window | important |
| --- | ---: | ---: |
| 0-1 | 384 | 48 |
| 2-3 | 256 | 32 |
| 4-5 | 128 | 16 |

也就是说：

- 低层保留更多最近 tokens 和更多 attention-selected tokens。
- 中层使用中等预算。
- 高层保留更少 tokens，但仍然保留 sink tokens。

如果 `sink_size=4`，三组预算分别是：

```text
低层: 4 + 384 + 48 = 436
中层: 4 + 256 + 32 = 292
高层: 4 + 128 + 16 = 148
```

6 层平均预算是：

```text
(436 * 2 + 292 * 2 + 148 * 2) / 6 = 292
```

这正好和 uniform `sink_snapkv` 的预算相同：

```text
4 + 256 + 32 = 292
```

所以主实验可以公平比较：平均 KV budget 相同，但分配方式不同。

## 4. 对比方法

主实验建议对比：

- `dense`：完整 KV Cache，作为质量上限。
- `streamingllm`：保留 sink tokens 和最近窗口。
- `sink_snapkv`：所有层共享同样预算的 sink + selected middle + recent window。
- `pyramid_sinkkv`：本项目方法，每层独立选择 tokens，并使用 pyramid budget。

消融实验建议对比：

- `sink_snapkv`：uniform budget。
- `pyramid_sinkkv`：低层多、高层少。
- `reverse_pyramid_sinkkv`：低层少、高层多，用于证明 pyramid 分配不是随便设计的。

## 5. 实验设置

推荐在 Linux GPU 服务器上运行：

- 模型：`EleutherAI/pythia-70m`
- Wikitext-2：`max_tokens=1024`
- PG-19：单样本，`max_tokens=1024`
- latency：`max_prompt_tokens=512`, `max_new_tokens=64`
- 默认参数：`sink_size=4`, `window_size=256`, `important_size=32`

一键运行：

```bash
bash scripts/run_server_experiments.sh
```

只汇总已有结果：

```bash
bash scripts/summarize_server_results.sh results/server
```

汇总文件会保存到：

```text
results/server/summary.md
```

## 6. 预期结论

这个方法的核心判断标准是：

1. 在平均 KV budget 接近时，`pyramid_sinkkv` 的 PPL 是否优于 `sink_snapkv`。
2. 在 PPL 接近时，`pyramid_sinkkv` 是否能使用更少的平均 KV tokens。
3. `pyramid_sinkkv` 是否优于 `reverse_pyramid_sinkkv`，从而说明“低层多保留、高层少保留”的设计有意义。

## 7. 局限性

- Pythia-70M 只有 6 层，layer-wise budget 的优势可能不如大模型明显。
- 当前实现偏向可复现，没有做 CUDA kernel 优化。
- `pyramid_sinkkv` 需要 attention weights，并且每层单独裁剪 KV Cache，因此会有额外 Python 开销。
