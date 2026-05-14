# 中文实验报告：KV Cache 压缩基线复现

## 目标

本仓库用于个人部分的可复现实验，主题是训练无关的 KV Cache 压缩。实验对象是
`EleutherAI/pythia-70m`，主要比较不同缓存保留策略对 PPL、保留 token 数量和生成延迟的影响。

## 已实现方法

- `dense`：保留完整 KV Cache。
- `sliding_window`：只保留最近窗口。
- `streamingllm`：保留开头 sink tokens 和最近窗口。
- `lm_infinite`：保留初始 tokens 和局部窗口。
- `h2o`：基于累积 attention 的 heavy-hitter 策略。
- `scissorhands`：基于持久 attention 重要性的保留策略。
- `tova`：保留当前 token 和少量高 attention 历史 token。
- `snapkv`：保留 attention 选择的中间 tokens 和最近窗口。
- `pyramidkv`：使用带层权重的 attention 重要性汇总。
- `sink_snapkv`：保留 sink tokens、attention-selected middle tokens 和最近窗口。

## 实验内容

仓库提供以下实验脚本：

- Wikitext-2 PPL，`max_tokens=1024`。
- PG-19 单样本 PPL，`max_tokens=1024`。
- PG-19 latency，`max_prompt_tokens=512`，`max_new_tokens=64`。
- 多种 KV Cache 压缩基线的统一对比。

一键运行：

```bash
bash scripts/run_server_experiments.sh
```

汇总已有结果：

```bash
bash scripts/summarize_server_results.sh results/server
```

## 说明

本仓库聚焦个人部分的基线复现和评测工具。
