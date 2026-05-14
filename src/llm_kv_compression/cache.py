from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


SUPPORTED_METHODS = {
    "dense",
    "sliding_window",
    "streamingllm",
    "lm_infinite",
    "h2o",
    "scissorhands",
    "tova",
    "snapkv",
    "pyramidkv",
    "sink_snapkv",
}

ATTENTION_METHODS = {
    "h2o",
    "scissorhands",
    "tova",
    "snapkv",
    "pyramidkv",
    "sink_snapkv",
}


def _transformers_version_tuple() -> tuple[int, int]:
    try:
        import transformers

        parts = transformers.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 999, 999


def _use_local_position_ids_for_compressed_cache() -> bool:
    override = os.getenv("KV_POSITION_IDS", "").strip().lower()
    if override in {"local", "compressed"}:
        return True
    if override in {"absolute", "global"}:
        return False
    # Older GPT-NeoX implementations build rotary tables from the compressed
    # cache length, so absolute positions can index past the table after
    # pruning. Newer Transformers versions handle absolute cache positions.
    return _transformers_version_tuple() < (4, 45)


def _supports_return_legacy_cache() -> bool:
    override = os.getenv("KV_RETURN_LEGACY_CACHE", "").strip().lower()
    if override in {"0", "false", "no"}:
        return False
    if override in {"1", "true", "yes"}:
        return True
    return _transformers_version_tuple() >= (4, 45)


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
        return self.method in ATTENTION_METHODS

    @property
    def nominal_budget(self) -> Optional[int]:
        if self.method == "dense":
            return None
        if self.method == "sliding_window":
            return self.window_size
        if self.method in {"streamingllm", "lm_infinite"}:
            return self.sink_size + self.window_size
        if self.method == "tova":
            return self.important_size + 1
        if self.method in {"h2o", "scissorhands", "snapkv"}:
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


def summarize_attention_importance(attentions, layer_weighting: str = "uniform") -> Optional[torch.Tensor]:
    """Average last-query attention over layers and heads."""

    layer_vectors = summarize_layer_attention_importance(attentions)
    if not layer_vectors:
        return None

    stacked = torch.stack(layer_vectors, dim=0)
    if layer_weighting == "pyramid":
        weights = torch.linspace(1.0, 2.0, stacked.shape[0], device=stacked.device, dtype=stacked.dtype)
        weights = weights / weights.sum()
        return (stacked * weights[:, None]).sum(dim=0)
    return stacked.mean(dim=0)


def summarize_layer_attention_importance(attentions) -> Optional[list[torch.Tensor]]:
    """Return last-query attention summaries for each Transformer block."""

    if not attentions:
        return None

    vectors = []
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        # Expected shape: [batch, heads, query_len, key_len].
        last_query = layer_attention.detach().float()[:, :, -1, :]
        vectors.append(last_query.mean(dim=(0, 1)))

    return vectors or None


def _unique_sorted(indices: Iterable[int]) -> list[int]:
    return sorted(set(int(i) for i in indices))


def select_sink_snapkv_indices(
    cache_length: int,
    sink_size: int,
    window_size: int,
    important_size: int,
    importance: Optional[torch.Tensor] = None,
) -> list[int]:
    """Keep sink tokens, attention-selected middle tokens, and recent tokens."""

    length = cache_length
    if length == 0:
        return []

    budget = sink_size + window_size + important_size
    if length <= budget:
        return list(range(length))

    sink_end = min(sink_size, length)
    recent_start = max(sink_end, length - window_size)
    sink = range(0, sink_end)
    recent = range(recent_start, length)
    protected = set(sink) | set(recent)
    middle = [idx for idx in range(length) if idx not in protected]

    if not middle or important_size == 0:
        return _unique_sorted(protected)

    keep_count = min(important_size, len(middle))
    if importance is not None and importance.numel() == length:
        middle_tensor = torch.tensor(middle, device=importance.device)
        middle_scores = importance.index_select(0, middle_tensor)
        top_local = torch.topk(middle_scores, k=keep_count).indices
        important = middle_tensor.index_select(0, top_local).detach().cpu().tolist()
    else:
        important = middle[-keep_count:]

    return _unique_sorted([*protected, *important])


def select_keep_indices(
    config: CachePolicyConfig,
    cache_length: int,
    importance: Optional[torch.Tensor] = None,
) -> list[int]:
    """Select chronological cache slots to retain after the current token."""

    length = cache_length
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

    if config.method in {"streamingllm", "lm_infinite"}:
        return _unique_sorted([*sink, *recent])

    if config.method == "tova":
        protected = {length - 1}
        candidates = [idx for idx in range(length - 1)]
        keep_count = min(max((config.nominal_budget or length) - 1, 0), len(candidates))
        if keep_count == 0:
            return [length - 1]
        if importance is not None and importance.numel() == length:
            candidate_tensor = torch.tensor(candidates, device=importance.device)
            candidate_scores = importance.index_select(0, candidate_tensor)
            top_local = torch.topk(candidate_scores, k=keep_count).indices
            important = candidate_tensor.index_select(0, top_local).detach().cpu().tolist()
        else:
            important = candidates[-keep_count:]
        return _unique_sorted([*protected, *important])

    if config.method in {"h2o", "scissorhands", "snapkv"}:
        recent_start = max(0, length - config.window_size)
        protected = set(range(recent_start, length))
        middle = [idx for idx in range(length) if idx not in protected]
        if not middle or config.important_size == 0:
            return _unique_sorted(protected)
        keep_count = min(config.important_size, len(middle))
        if importance is not None and importance.numel() == length:
            middle_tensor = torch.tensor(middle, device=importance.device)
            middle_scores = importance.index_select(0, middle_tensor)
            top_local = torch.topk(middle_scores, k=keep_count).indices
            important = middle_tensor.index_select(0, top_local).detach().cpu().tolist()
        else:
            important = middle[-keep_count:]
        return _unique_sorted([*protected, *important])

    return select_sink_snapkv_indices(
        cache_length=length,
        sink_size=config.sink_size,
        window_size=config.window_size,
        important_size=config.important_size,
        importance=importance,
    )


def prune_legacy_cache(past_key_values, keep_indices: list[int]):
    """Prune tuple-based KV cache tensors along the sequence dimension."""

    past_key_values = as_legacy_cache(past_key_values)
    if past_key_values is None:
        return None
    if not keep_indices:
        return past_key_values

    pruned = []
    for key, value in past_key_values:
        if not keep_indices:
            pruned.append((key[:, :, :0, :], value[:, :, :0, :]))
            continue
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
        self.retained_history: list[float] = []
        self.max_retained_tokens = 0

    def _build_model_kwargs(self, input_ids: torch.Tensor, cache_len: int) -> dict:
        device = input_ids.device
        position_value = cache_len if _use_local_position_ids_for_compressed_cache() else self.total_seen
        position_ids = torch.tensor([[position_value]], dtype=torch.long, device=device)
        kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": self.past_key_values,
            "use_cache": True,
            "output_attentions": self.config.needs_attention,
            "return_dict": True,
        }
        kwargs["attention_mask"] = torch.ones((1, cache_len + 1), dtype=torch.long, device=device)
        return kwargs

    def step(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Run one token and update the compressed KV cache.

        `input_ids` must have shape [1, 1]. The returned logits predict the
        token after `input_ids`.
        """

        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("KVCacheRuntime.step expects input_ids with shape [batch=1, seq=1].")

        kwargs = self._build_model_kwargs(input_ids, len(self.cache_positions))

        with torch.inference_mode():
            if _supports_return_legacy_cache():
                outputs = model(**kwargs, return_legacy_cache=True)
            else:
                outputs = model(**kwargs)

        full_cache = as_legacy_cache(outputs.past_key_values)
        self._update_shared_cache(full_cache, outputs.attentions)

        self.total_seen += 1
        return outputs.logits

    def _update_shared_cache(self, full_cache, attentions) -> None:
        new_positions = [*self.cache_positions, self.total_seen]
        layer_weighting = "pyramid" if self.config.method == "pyramidkv" else "uniform"
        importance = (
            summarize_attention_importance(attentions, layer_weighting=layer_weighting)
            if self.config.needs_attention
            else None
        )
        if self.config.needs_attention and importance is None:
            raise RuntimeError(
                f"{self.config.method} requires attention weights, but the model did not return them. "
                "Load the model with attn_implementation='eager' or use a non-attention policy."
            )

        selection_importance = importance
        state_importance = None
        cache_len = len(self.cache_positions)
        if self.config.method in {"h2o", "scissorhands"}:
            if importance is None or importance.numel() != len(new_positions):
                raise RuntimeError(
                    f"{self.config.method} expected {len(new_positions)} attention scores, "
                    f"but received {0 if importance is None else importance.numel()}."
                )
            previous = self.importance_scores
            if previous is None or previous.numel() != cache_len:
                previous = torch.zeros(cache_len, dtype=importance.dtype, device=importance.device)
            else:
                previous = previous.to(device=importance.device, dtype=importance.dtype)
            expanded_previous = torch.cat([previous, torch.zeros(1, dtype=importance.dtype, device=importance.device)])
            if self.config.method == "h2o":
                state_importance = expanded_previous + importance
            else:
                state_importance = torch.maximum(expanded_previous, importance)
            selection_importance = state_importance

        keep_indices = select_keep_indices(self.config, len(new_positions), selection_importance)
        pruned_cache = prune_legacy_cache(full_cache, keep_indices)
        self.past_key_values = to_model_cache(pruned_cache)
        self.cache_positions = [new_positions[idx] for idx in keep_indices]
        if self.config.method in {"h2o", "scissorhands"} and state_importance is not None:
            index = torch.tensor(keep_indices, device=state_importance.device, dtype=torch.long)
            self.importance_scores = state_importance.index_select(0, index).detach()
        else:
            self.importance_scores = None
        self._record_retained([len(self.cache_positions)])

    def _record_retained(self, layer_lengths: list[int]) -> None:
        if not layer_lengths:
            retained = 0.0
            max_retained = 0
        else:
            retained = float(sum(layer_lengths) / len(layer_lengths))
            max_retained = max(layer_lengths)
        self.retained_history.append(retained)
        self.max_retained_tokens = max(self.max_retained_tokens, max_retained)

    @property
    def average_retained_tokens(self) -> float:
        if not self.retained_history:
            return 0.0
        return float(sum(self.retained_history) / len(self.retained_history))
