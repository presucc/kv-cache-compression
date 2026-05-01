from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_kv_compression.cache import CachePolicyConfig, prune_legacy_cache, select_keep_indices


def test_streamingllm_keeps_sink_and_recent_tokens():
    config = CachePolicyConfig(method="streamingllm", sink_size=2, window_size=3)
    keep = select_keep_indices(config, list(range(10)))
    assert keep == [0, 1, 7, 8, 9]


def test_asw_kv_keeps_attention_selected_middle_tokens():
    config = CachePolicyConfig(method="asw_kv", sink_size=2, window_size=2, important_size=2)
    importance = torch.tensor([0.0, 0.0, 0.1, 0.9, 0.2, 0.8, 0.0, 0.0])
    keep = select_keep_indices(config, list(range(8)), importance)
    assert keep == [0, 1, 3, 5, 6, 7]


def test_snapkv_keeps_important_tokens_and_recent_window_without_sink():
    config = CachePolicyConfig(method="snapkv", window_size=2, important_size=2)
    importance = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.0, 0.0])
    keep = select_keep_indices(config, list(range(8)), importance)
    assert keep == [1, 3, 6, 7]


def test_h2o_uses_supplied_heavy_hitter_scores():
    config = CachePolicyConfig(method="h2o", window_size=2, important_size=2)
    cumulative_importance = torch.tensor([0.1, 0.2, 0.95, 0.3, 0.85, 0.4, 0.0, 0.0])
    keep = select_keep_indices(config, list(range(8)), cumulative_importance)
    assert keep == [2, 4, 6, 7]


def test_prune_legacy_cache_prunes_sequence_dimension():
    key = torch.arange(1 * 2 * 5 * 3).reshape(1, 2, 5, 3)
    value = key + 1000
    pruned = prune_legacy_cache(((key, value),), [0, 2, 4])
    out_key, out_value = pruned[0]
    assert out_key.shape == (1, 2, 3, 3)
    assert out_value.shape == (1, 2, 3, 3)
    assert torch.equal(out_key[:, :, 1, :], key[:, :, 2, :])
