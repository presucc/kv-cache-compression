# 中文实验报告：基于 KV Cache 压缩的大语言模型推理加速

## 1. 实验背景

大语言模型在生成文本时，并不是每次都从头重新计算前面所有 token 的隐藏状态。为了节省计算量，模型会把前面 token 的 Key 和 Value 存下来，这部分缓存叫做 KV Cache。下一次生成新 token 时，模型只需要计算新 token，并让它去注意前面缓存里的内容。

KV Cache 的好处是可以明显减少重复计算，但它也有一个问题：上下文越长，缓存越大，占用的显存也越多。对于长文本任务，比如 PG-19 这种长篇书籍数据集，如果一直保留完整 KV Cache，显存压力会越来越大，推理速度也可能受到影响。

因此，本实验研究的问题是：

> 在不训练模型、不修改模型参数的情况下，能不能只保留一部分重要的 KV Cache，从而减少显存占用，同时尽量保持模型效果？

这类方法属于训练无关的推理阶段优化，比较适合作为课程作业中的“可复现推理加速方法”。

## 2. 作业目标与本实验任务

根据课程 PPT 的要求，个人部分需要完成一个语言模型推理加速或优化方法的复现，实现一个公开可复现的代码仓库，并在指定模型和数据集上进行测试。

本实验使用：

- 模型：`EleutherAI/pythia-70m`
- 数据集：Wikitext-2 和 PG-19
- 任务：语言模型困惑度测试和生成延迟测试
- 约束：不训练模型，不改模型参数，只在推理阶段压缩 KV Cache

为了让实验更完整，本仓库不只实现一个方法，而是实现并对比了九种独立 KV Cache 策略：

- `dense`：完整 KV Cache，不压缩，作为质量基线
- `sliding_window`：只保留最近的一段 token
- `streamingllm`：保留开头的 attention sink token 和最近窗口
- `lm_infinite`：代码中保留为参考别名；在本轻量框架中与 StreamingLLM 的缓存形状一致，因此不作为独立主表结果
- `h2o`：保留累计注意力最高的 heavy hitter token 和最近窗口
- `scissorhands`：保留历史上持续重要的 token 和最近窗口
- `tova`：保留当前 token 和少量当前注意力最高的历史 token
- `snapkv`：保留当前注意力选出的重要 token 和最近窗口
- `pyramidkv`：保留 sink、最近窗口，以及按层加权注意力选出的中间 token
- `asw_kv`：本实验提出的轻量扩展方法，保留 attention sink、注意力选出的中间重要 token 和最近窗口

## 3. 方法介绍

### 3.1 Dense Baseline

`dense` 方法保留所有历史 token 的 KV Cache。它的效果通常最好，因为模型可以看到完整历史信息；但它的缓存大小也最大。这个方法主要作为对照组，用来判断其他压缩方法会损失多少效果。

### 3.2 Sliding Window

`sliding_window` 只保留最近的若干 token，例如最近 256 个 token。它的优点是非常简单，缓存大小固定；缺点是会直接丢掉较早的上下文。如果文本前面有重要信息，模型后面就无法再利用。

### 3.3 StreamingLLM

StreamingLLM 的核心观察是：大模型在长文本生成中，经常会持续关注最开头的一小部分 token，这些 token 被称为 attention sink。即使它们本身不一定有语义信息，也能帮助注意力分布保持稳定。

因此，StreamingLLM 保留：

```text
[开头 sink tokens] + [最近窗口 tokens]
```

相比纯滑动窗口，它多保留了开头几个 token，通常能明显改善困惑度。

### 3.4 LM-Infinite-lite

LM-Infinite 也强调“初始 token + 局部窗口”的长文本推理结构。本实验实现的是轻量版本，在当前统一框架里它和 StreamingLLM 使用相同的缓存布局：

```text
[初始 tokens] + [最近窗口 tokens]
```

因此如果把它放进表格，实验数值会和 StreamingLLM 完全相同。为了避免误导，主实验表不单独列出 LM-Infinite，只在方法说明中说明这种方法族的共同点。

### 3.5 H2O-lite

H2O 的思想是 heavy hitter：如果某些历史 token 长期被后续 token 注意到，说明它们可能比较重要。本实验实现的是一个轻量版本，用累计注意力分数来表示 token 的重要性。

H2O-lite 保留：

```text
[累计注意力最高的 tokens] + [最近窗口 tokens]
```

它不固定保留开头 sink token，而是让历史注意力统计决定哪些 token 应该留下。

### 3.6 Scissorhands-lite

Scissorhands 的核心想法是“重要性具有持续性”：如果一个 token 曾经被高度关注，那么它后面仍然可能有用。本实验实现的是轻量版本，用历史最大注意力分数表示这种持续重要性。

Scissorhands-lite 保留：

```text
[历史最大注意力较高的 tokens] + [最近窗口 tokens]
```

它和 H2O-lite 的区别是：H2O 累加注意力，Scissorhands-lite 记录历史最高注意力。

### 3.7 TOVA-lite

TOVA 的思想是根据注意力删掉不重要 token。本实验实现的轻量版本会保留当前 token，并从历史 token 里选择当前注意力最高的一部分。

TOVA-lite 保留：

```text
[当前 token] + [当前注意力最高的历史 tokens]
```

它比较激进，因为它不强制保留完整最近窗口。本实验中它的预算是 `important_size + 1`，也就是 32 个高注意力历史 token 加 1 个当前 token，总共 33 个 token。

### 3.8 SnapKV-lite

SnapKV 的思想是利用当前注意力来判断哪些上下文 token 对后续生成更重要。本实验实现的是一个轻量版本：在每一步推理时，用当前 query 对历史 token 的平均注意力分数选择重要 token。

SnapKV-lite 保留：

```text
[当前注意力选出的重要 tokens] + [最近窗口 tokens]
```

它和 H2O-lite 的区别是：H2O 更看重长期累计注意力，SnapKV-lite 更看重当前这一步的注意力。

### 3.9 PyramidKV-lite

PyramidKV 原方法关注不同层里 KV Cache 信息分布的差异。本实验受限于 Hugging Face 当前缓存接口，为了保证所有方法能在同一套框架里公平比较，实现了一个轻量版本：仍然使用统一的缓存位置，但在汇总各层 attention 时给更深层更大的权重。

PyramidKV-lite 保留：

```text
[sink tokens] + [按层加权注意力选出的中间 tokens] + [最近窗口 tokens]
```

### 3.10 ASW-KV

ASW-KV 是本实验提出的轻量扩展方法，可以理解为在 StreamingLLM 的基础上加入一个注意力选择的中间记忆区。

ASW-KV 保留：

```text
[attention sink tokens] + [注意力选出的中间重要 tokens] + [最近窗口 tokens]
```

它的动机是：只保留开头和最近窗口可能还不够，因为中间上下文里也可能有重要信息。因此，ASW-KV 额外从中间区域选择一小部分当前注意力较高的 token 保留下来。

## 4. 实验设置

实验参数如下：

- 模型：`EleutherAI/pythia-70m`
- PPL 测试长度：1024 tokens
- PPL 测试设备：CPU，float32
- Latency 测试设备：NVIDIA GeForce RTX 4080 Laptop GPU，float16
- `window_size`：256
- `sink_size`：4
- `important_size`：32

主要评测指标包括：

- PPL：困惑度，越低说明模型预测下一个 token 的效果越好
- Max KV tokens：推理过程中最多保留的 KV token 数量
- Avg KV tokens：平均保留的 KV token 数量
- TTFT：time to first token，生成第一个 token 的时间
- TPOT：time per output token，平均每个输出 token 的生成时间
- Throughput：吞吐率，每秒生成 token 数
- Peak CUDA memory：GPU 峰值显存占用

## 5. 困惑度实验结果

### 5.1 PG-19

| Method | PPL | Max KV tokens | Avg KV tokens |
| --- | ---: | ---: | ---: |
| dense | 31.10 | 1023 | 512.00 |
| sliding_window | 36.30 | 256 | 224.09 |
| streamingllm | 31.43 | 260 | 227.09 |
| h2o | 31.39 | 288 | 247.60 |
| scissorhands | 31.43 | 288 | 247.60 |
| tova | 44.91 | 33 | 32.48 |
| snapkv | 31.23 | 288 | 247.60 |
| pyramidkv | 31.30 | 292 | 250.47 |
| asw_kv | 31.23 | 292 | 250.47 |

在 PG-19 上，`sliding_window` 的 PPL 明显变差，说明只保留最近 token 会丢掉重要上下文。`streamingllm` 加入开头 sink token 后，PPL 从 36.30 降到 31.43，已经接近完整缓存的 31.10。

`snapkv` 和 `asw_kv` 的结果最好，PPL 都约为 31.23。`pyramidkv` 也比较接近，PPL 为 31.30。它们只保留约 288 到 292 个 KV token，却接近 dense 的效果。这说明注意力选择出来的中间 token 确实能补回一部分滑动窗口丢失的信息。

`tova` 的 PPL 为 44.91，明显更差，但它的预算只有 33 个 token，压缩强度远高于其他方法。因此它展示的是极强压缩下的质量损失，而不是和 256-token 窗口方法同等预算下的比较。

### 5.2 Wikitext-2

| Method | PPL | Max KV tokens | Avg KV tokens |
| --- | ---: | ---: | ---: |
| dense | 30.15 | 1023 | 512.00 |
| sliding_window | 42.95 | 256 | 224.09 |
| streamingllm | 36.19 | 260 | 227.09 |
| h2o | 35.51 | 288 | 247.60 |
| scissorhands | 36.44 | 288 | 247.60 |
| tova | 65.50 | 33 | 32.48 |
| snapkv | 36.16 | 288 | 247.60 |
| pyramidkv | 35.58 | 292 | 250.47 |
| asw_kv | 35.68 | 292 | 250.47 |

在 Wikitext-2 上，`h2o` 表现最好，PPL 为 35.51，略优于 `pyramidkv` 的 35.58 和 `asw_kv` 的 35.68。这个结果说明累计注意力在 Wikitext-2 这种文本上比较有用，因为长期被关注的 token 往往包含稳定的重要信息。

ASW-KV 也比 StreamingLLM 更好，PPL 从 36.19 降到 35.68，说明在 sink 和最近窗口之外加入中间重要 token 是有效的。

不过也可以看到，`tova` 在 Wikitext-2 上质量损失很大，PPL 为 65.50。这可能是因为本实验的 TOVA-lite 不强制保留完整最近窗口，只保留当前 token 和高注意力历史 token，因此会丢掉大量局部连续上下文。

## 6. GPU 延迟实验结果

延迟实验使用 512-token PG-19 prompt，并生成 64 个 token。

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

从显存看，所有压缩方法的峰值显存都低于 dense。dense 的峰值显存约为 165.62 MB，而压缩方法大多在 154 到 156 MB 左右。

从速度看，本实验在计时前加入了一次不计时的 CUDA warmup，尽量去掉 GPU 冷启动带来的影响。不过实现仍然是为了清楚和可复现，而不是为了极致优化。它使用 Python 循环逐 token 推理，所以绝对速度不能代表 vLLM、TensorRT-LLM 这类高性能推理框架。

更稳定的结论是显存：`tova` 因为只保留 33 个 token，峰值显存最低，为 146.58 MB；其他压缩方法也都低于 dense。

## 7. ASW-KV 消融实验

为了验证“中间重要 token 数量”是否真的有用，本实验固定 PG-19、1024 tokens、`sink_size=4` 和 `window_size=256`，只改变 `important_size`。

| important_size | PPL | Max KV tokens | Avg KV tokens |
| ---: | ---: | ---: | ---: |
| 0 | 31.43 | 260 | 227.09 |
| 16 | 31.39 | 276 | 238.90 |
| 32 | 31.23 | 292 | 250.47 |
| 64 | 31.15 | 324 | 272.85 |

当 `important_size=0` 时，ASW-KV 退化得接近 StreamingLLM，只保留 sink 和最近窗口。随着 `important_size` 增大，PPL 单调下降，从 31.43 降到 31.15。这说明额外保留注意力选出的中间 token 可以提高模型效果。

当然，`important_size` 越大，缓存也越大。因此实际使用时需要在 PPL 和显存之间做权衡。

## 8. 结论

本实验复现并比较了多种训练无关的 KV Cache 压缩方法。实验说明，完整 KV Cache 效果最好但缓存最大；纯滑动窗口最简单但质量损失明显；StreamingLLM 通过保留初始 token 和局部窗口能显著改善效果；H2O-lite、SnapKV-lite、PyramidKV-lite 等注意力选择方法可以进一步提升困惑度。

TOVA-lite 展示了另一个极端：它的缓存预算只有 33 个 token，因此显存最低，但 PPL 明显变差。这说明 KV Cache 压缩不能只看缓存大小，也必须同时看质量指标。

本实验提出的 ASW-KV 在 StreamingLLM 的基础上加入注意力选择的中间记忆区。它在 PG-19 和 Wikitext-2 上都优于 StreamingLLM，并且保留的 KV token 数量远少于 dense。消融实验也说明，增加中间重要 token 的预算可以稳定降低 PPL。

总体来说，KV Cache 压缩是一种很适合长文本推理的优化方向。它不需要训练模型，只改变推理时保留缓存的策略，就可以在显存占用和模型效果之间取得更好的平衡。

## 9. 局限性

本实验仍然有一些不足：

- 使用的是 Pythia-70M，小模型结果不能完全代表大模型。
- PPL 只测试了 1024 tokens，更长上下文下的效果还需要继续验证。
- 当前实现偏向清楚和可复现，没有使用高性能 CUDA kernel。
- `h2o`、`snapkv` 和 `asw_kv` 都需要读取 attention weights，这会带来额外开销。
- InfLLM 和 TreeKV 没有纳入统一表格，因为它们更依赖块检索或树结构缓存，适合单独实现和评估。
- PG-19 实验使用的是单个长样本，后续可以扩展到更多样本。

## 10. 可复现性

代码仓库已经包含完整实现、运行脚本和实验结果。主要复现实验可以通过下面命令运行：

```powershell
.\scripts\run_all_experiments.ps1
```

也可以单独运行 PPL 或 latency 脚本。所有主要结果保存在 `results/` 目录下，方便检查和复现。
