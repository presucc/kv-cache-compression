from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


SUPPORTED_METHODS = {"dense", "sliding_window", "streamingllm", "h2o", "snapkv", "asw_kv"}


@dataclass(frozen=True)
class CachePolicyConfig:
    """Configuration for a training-free KV cache retention policy."""

    method: str = "dense"
    window_size: int = 256
    sink_size: int = 4
    important_size: int = 32

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown method {self.method!r}. Choose from {sorted(SUPPORTED_METHODS)}.")
        if self.window_size < 0 or self.sink_size < 0 or self.important_size < 0:
            raise ValueError("Cache sizes must be non-negative.")

    @property
    def needs_attention(self) -> bool:
        return self.method in {"h2o", "snapkv", "asw_kv"}

    @property
    def nominal_budget(self) -> Optional[int]:
        if self.method == "dense":
            return None
        if self.method == "sliding_window":
            return self.window_size
        if self.method == "streamingllm":
            return self.sink_size + self.window_size
        if self.method in {"h2o", "snapkv"}:
            return self.important_size + self.window_size
        return self.sink_size + self.important_size + self.window_size


def as_legacy_cache(past_key_values):
    """Return a tuple-based cache when Transformers returns a Cache object."""

    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def to_model_cache(past_key_values):
    """Wrap legacy tuples for Transformers versions that require Cache objects."""

    if past_key_values is None:
        return None
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(past_key_values)
    except Exception:
        return past_key_values


def summarize_attention_importance(attentions) -> Optional[torch.Tensor]:
    """Average last-query attention over layers and heads.

    Returns a vector with one score for every slot in the current retained cache.
    """

    if not attentions:
        return None

    vectors = []
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        # Expected shape: [batch, heads, query_len, key_len].
        last_query = layer_attention.detach().float()[:, :, -1, :]
        vectors.append(last_query.mean(dim=(0, 1)))

    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def _unique_sorted(indices: Iterable[int]) -> list[int]:
    return sorted(set(int(i) for i in indices))


def select_keep_indices(
    config: CachePolicyConfig,
    cache_positions: list[int],
    importance: Optional[torch.Tensor] = None,
) -> list[int]:
    """Select chronological cache slots to retain after the current token."""

    length = len(cache_positions)
    if config.method == "dense" or length == 0:
        return list(range(length))

    budget = config.nominal_budget
    if budget is not None and length <= budget:
        return list(range(length))

    if config.method == "sliding_window":
        start = max(0, length - config.window_size)
        return list(range(start, length))

    sink_end = min(config.sink_size, length)
    recent_start = max(sink_end, length - config.window_size)
    sink = range(0, sink_end)
    recent = range(recent_start, length)

    if config.method == "streamingllm":
        return _unique_sorted([*sink, *recent])

    if config.method in {"h2o", "snapkv"}:
        recent_start = max(0, length - config.window_size)
        protected = set(range(recent_start, length))
    else:
        protected = set(sink) | set(recent)

    middle = [idx for idx in range(length) if idx not in protected]
    if not middle or config.important_size == 0:
        return _unique_sorted([*protected])

    keep_count = min(config.important_size, len(middle))
    if importance is not None and importance.numel() == length:
        middle_tensor = torch.tensor(middle, device=importance.device)
        middle_scores = importance.index_select(0, middle_tensor)
        top_local = torch.topk(middle_scores, k=keep_count).indices
        important = middle_tensor.index_select(0, top_local).detach().cpu().tolist()
    else:
        # Fallback keeps the most recent middle tokens if attention weights are
        # unavailable for a model/version.
        important = middle[-keep_count:]

    return _unique_sorted([*protected, *important])


def prune_legacy_cache(past_key_values, keep_indices: list[int]):
    """Prune tuple-based KV cache tensors along the sequence dimension."""

    past_key_values = as_legacy_cache(past_key_values)
    if past_key_values is None:
        return None
    if not keep_indices:
        return tuple((key[:, :, :0, :], value[:, :, :0, :]) for key, value in past_key_values)

    pruned = []
    for key, value in past_key_values:
        index = torch.tensor(keep_indices, device=key.device, dtype=torch.long)
        pruned.append((key.index_select(2, index), value.index_select(2, index)))
    return tuple(pruned)


class KVCacheRuntime:
    """Stateful one-token-at-a-time inference wrapper with cache compression."""

    def __init__(self, config: CachePolicyConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.past_key_values = None
        self.cache_positions: list[int] = []
        self.importance_scores: Optional[torch.Tensor] = None
        self.total_seen = 0
        self.retained_history: list[int] = []
        self.max_retained_tokens = 0

    def step(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Run one token and update the compressed KV cache.

        `input_ids` must have shape [1, 1]. The returned logits predict the
        token after `input_ids`.
        """

        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("KVCacheRuntime.step expects input_ids with shape [batch=1, seq=1].")

        device = input_ids.device
        cache_len = len(self.cache_positions)
        attention_mask = torch.ones((1, cache_len + 1), dtype=torch.long, device=device)
        position_ids = torch.tensor([[self.total_seen]], dtype=torch.long, device=device)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": self.past_key_values,
            "use_cache": True,
            "output_attentions": self.config.needs_attention,
            "return_dict": True,
        }

        with torch.inference_mode():
            try:
                outputs = model(**kwargs, return_legacy_cache=True)
            except TypeError:
                outputs = model(**kwargs)

        full_cache = as_legacy_cache(outputs.past_key_values)
        new_positions = [*self.cache_positions, self.total_seen]
        importance = summarize_attention_importance(outputs.attentions) if self.config.needs_attention else None

        selection_importance = importance
        cumulative_importance = None
        if self.config.method == "h2o":
            if importance is not None and importance.numel() == len(new_positions):
                previous = self.importance_scores
                if previous is None or previous.numel() != cache_len:
                    previous = torch.zeros(cache_len, dtype=importance.dtype, device=importance.device)
                else:
                    previous = previous.to(device=importance.device, dtype=importance.dtype)
                cumulative_importance = torch.cat(
                    [previous, torch.zeros(1, dtype=importance.dtype, device=importance.device)]
                )
                cumulative_importance = cumulative_importance + importance
                selection_importance = cumulative_importance
            else:
                selection_importance = None

        keep_indices = select_keep_indices(self.config, new_positions, selection_importance)

        pruned_cache = prune_legacy_cache(full_cache, keep_indices)
        self.past_key_values = to_model_cache(pruned_cache)
        self.cache_positions = [new_positions[idx] for idx in keep_indices]
        if self.config.method == "h2o" and cumulative_importance is not None:
            index = torch.tensor(keep_indices, device=cumulative_importance.device, dtype=torch.long)
            self.importance_scores = cumulative_importance.index_select(0, index).detach()
        else:
            self.importance_scores = None
        self.total_seen += 1

        retained = len(self.cache_positions)
        self.retained_history.append(retained)
        self.max_retained_tokens = max(self.max_retained_tokens, retained)
        return outputs.logits

    @property
    def average_retained_tokens(self) -> float:
        if not self.retained_history:
            return 0.0
        return float(sum(self.retained_history) / len(self.retained_history))
