from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .cache import CachePolicyConfig, KVCacheRuntime


@dataclass
class PerplexityResult:
    method: str
    tokens: int
    nll: float
    ppl: float
    max_retained_tokens: int
    average_retained_tokens: float
    nominal_budget: int | None


@dataclass
class LatencyResult:
    method: str
    prompt_tokens: int
    generated_tokens: int
    ttft_seconds: float
    tpot_seconds: float
    throughput_tokens_per_second: float
    end_to_end_tokens_per_second: float
    max_retained_tokens: int
    average_retained_tokens: float
    nominal_budget: int | None
    peak_cuda_memory_mb: float | None
    generated_text: str


def result_to_dict(result):
    return asdict(result)


def evaluate_perplexity(
    model,
    tokenizer,
    text: str,
    config: CachePolicyConfig,
    device: torch.device,
    max_tokens: int,
) -> PerplexityResult:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encoded.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least two tokens to compute perplexity.")

    runtime = KVCacheRuntime(config)
    losses = []

    for idx in range(input_ids.shape[1] - 1):
        token = input_ids[:, idx : idx + 1]
        target = input_ids[:, idx + 1]
        logits = runtime.step(model, token)[:, -1, :]
        loss = F.cross_entropy(logits.float(), target, reduction="mean")
        losses.append(loss.detach())

    nll = torch.stack(losses).mean().item()
    return PerplexityResult(
        method=config.method,
        tokens=input_ids.shape[1],
        nll=nll,
        ppl=math.exp(nll),
        max_retained_tokens=runtime.max_retained_tokens,
        average_retained_tokens=runtime.average_retained_tokens,
        nominal_budget=config.nominal_budget,
    )


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def evaluate_latency(
    model,
    tokenizer,
    prompt: str,
    config: CachePolicyConfig,
    device: torch.device,
    max_new_tokens: int,
    max_prompt_tokens: int,
) -> LatencyResult:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    prompt_ids = encoded.input_ids.to(device)
    if prompt_ids.numel() == 0:
        raise ValueError("Prompt is empty after tokenization.")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    runtime = KVCacheRuntime(config)
    generated: list[int] = []

    _sync_if_cuda(device)
    start = time.perf_counter()

    logits = None
    for idx in range(prompt_ids.shape[1]):
        logits = runtime.step(model, prompt_ids[:, idx : idx + 1])

    next_token = _greedy_next_token(logits)
    generated.append(int(next_token.item()))

    _sync_if_cuda(device)
    first_done = time.perf_counter()
    ttft = first_done - start

    decode_start = first_done
    for _ in range(max_new_tokens - 1):
        logits = runtime.step(model, next_token.to(device))
        next_token = _greedy_next_token(logits)
        generated.append(int(next_token.item()))

    _sync_if_cuda(device)
    end = time.perf_counter()

    decode_time = max(end - decode_start, 1e-12)
    total_time = max(end - start, 1e-12)
    tpot = decode_time / max(max_new_tokens - 1, 1)
    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)

    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return LatencyResult(
        method=config.method,
        prompt_tokens=prompt_ids.shape[1],
        generated_tokens=max_new_tokens,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        throughput_tokens_per_second=max_new_tokens / max(ttft + decode_time, 1e-12),
        end_to_end_tokens_per_second=max_new_tokens / total_time,
        max_retained_tokens=runtime.max_retained_tokens,
        average_retained_tokens=runtime.average_retained_tokens,
        nominal_budget=config.nominal_budget,
        peak_cuda_memory_mb=peak_memory,
        generated_text=generated_text,
    )


def configs_for_methods(
    methods: Sequence[str],
    window_size: int,
    sink_size: int,
    important_size: int,
) -> list[CachePolicyConfig]:
    return [
        CachePolicyConfig(
            method=method,
            window_size=window_size,
            sink_size=sink_size,
            important_size=important_size,
        )
        for method in methods
    ]
