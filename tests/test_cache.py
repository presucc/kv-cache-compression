from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_kv_compression.cache import (
    CachePolicyConfig,
    layer_budget_for_method,
    prune_legacy_cache,
    select_per_layer_keep_indices,
    select_keep_indices,
    summarize_attention_importance,
)


def test_streamingllm_keeps_sink_and_recent_tokens():
    config = CachePolicyConfig(method="streamingllm", sink_size=2, window_size=3)
    keep = select_keep_indices(config, 10)
    assert keep == [0, 1, 7, 8, 9]


def test_lm_infinite_uses_initial_tokens_and_local_window():
    config = CachePolicyConfig(method="lm_infinite", sink_size=2, window_size=3)
    keep = select_keep_indices(config, 10)
    assert keep == [0, 1, 7, 8, 9]


def test_sink_snapkv_keeps_attention_selected_middle_tokens():
    config = CachePolicyConfig(method="sink_snapkv", sink_size=2, window_size=2, important_size=2)
    importance = torch.tensor([0.0, 0.0, 0.1, 0.9, 0.2, 0.8, 0.0, 0.0])
    keep = select_keep_indices(config, 8, importance)
    assert keep == [0, 1, 3, 5, 6, 7]


def test_snapkv_keeps_important_tokens_and_recent_window_without_sink():
    config = CachePolicyConfig(method="snapkv", window_size=2, important_size=2)
    importance = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.0, 0.0])
    keep = select_keep_indices(config, 8, importance)
    assert keep == [1, 3, 6, 7]


def test_h2o_uses_supplied_heavy_hitter_scores():
    config = CachePolicyConfig(method="h2o", window_size=2, important_size=2)
    cumulative_importance = torch.tensor([0.1, 0.2, 0.95, 0.3, 0.85, 0.4, 0.0, 0.0])
    keep = select_keep_indices(config, 8, cumulative_importance)
    assert keep == [2, 4, 6, 7]


def test_scissorhands_keeps_persistent_important_tokens():
    config = CachePolicyConfig(method="scissorhands", window_size=2, important_size=2)
    persistent_scores = torch.tensor([0.1, 0.2, 0.95, 0.3, 0.85, 0.4, 0.0, 0.0])
    keep = select_keep_indices(config, 8, persistent_scores)
    assert keep == [2, 4, 6, 7]


def test_tova_keeps_current_token_and_top_attention_tokens():
    config = CachePolicyConfig(method="tova", window_size=2, important_size=2)
    importance = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.0, 0.0])
    keep = select_keep_indices(config, 8, importance)
    assert keep == [1, 3, 7]
    assert config.nominal_budget == 3


def test_pyramidkv_keeps_sink_weighted_middle_and_recent_tokens():
    config = CachePolicyConfig(method="pyramidkv", sink_size=2, window_size=2, important_size=2)
    importance = torch.tensor([0.0, 0.0, 0.1, 0.9, 0.2, 0.8, 0.0, 0.0])
    keep = select_keep_indices(config, 8, importance)
    assert keep == [0, 1, 3, 5, 6, 7]


def test_pyramid_sinkkv_assigns_larger_budgets_to_lower_layers():
    config = CachePolicyConfig(method="pyramid_sinkkv", window_size=256, important_size=32)
    budgets = [layer_budget_for_method(config, layer_idx, 6) for layer_idx in range(6)]
    assert [(budget.window_size, budget.important_size) for budget in budgets] == [
        (384, 48),
        (384, 48),
        (256, 32),
        (256, 32),
        (128, 16),
        (128, 16),
    ]


def test_reverse_pyramid_sinkkv_assigns_larger_budgets_to_higher_layers():
    config = CachePolicyConfig(method="reverse_pyramid_sinkkv", window_size=256, important_size=32)
    budgets = [layer_budget_for_method(config, layer_idx, 6) for layer_idx in range(6)]
    assert [(budget.window_size, budget.important_size) for budget in budgets] == [
        (128, 16),
        (128, 16),
        (256, 32),
        (256, 32),
        (384, 48),
        (384, 48),
    ]


def test_pyramid_sinkkv_selects_different_indices_per_layer():
    config = CachePolicyConfig(method="pyramid_sinkkv", sink_size=1, window_size=4, important_size=2)
    layer_importance = [
        torch.tensor([0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.0, 0.0]),
        torch.tensor([0.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.0, 0.0]),
        torch.tensor([0.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.0, 0.0]),
    ]
    keep = select_per_layer_keep_indices(config, [8, 8, 8], layer_importance)
    assert keep[0] == list(range(8))
    assert keep[1] == [0, 2, 3, 4, 5, 6, 7]
    assert keep[2] == [0, 2, 6, 7]


def test_prune_legacy_cache_supports_per_layer_indices():
    key0 = torch.arange(1 * 1 * 5 * 2).reshape(1, 1, 5, 2)
    value0 = key0 + 100
    key1 = key0 + 1000
    value1 = key1 + 100
    pruned = prune_legacy_cache(((key0, value0), (key1, value1)), [[0, 2, 4], [1, 3]])
    assert pruned[0][0].shape == (1, 1, 3, 2)
    assert pruned[1][0].shape == (1, 1, 2, 2)
    assert torch.equal(pruned[1][0][:, :, 0, :], key1[:, :, 1, :])


def test_pyramid_attention_summary_weights_deeper_layers_more():
    layer_0 = torch.tensor([[[[0.8, 0.2]]]])
    layer_1 = torch.tensor([[[[0.2, 0.8]]]])
    summary = summarize_attention_importance([layer_0, layer_1], layer_weighting="pyramid")
    assert summary[1] > summary[0]


def test_prune_legacy_cache_prunes_sequence_dimension():
    key = torch.arange(1 * 2 * 5 * 3).reshape(1, 2, 5, 3)
    value = key + 1000
    pruned = prune_legacy_cache(((key, value),), [0, 2, 4])
    out_key, out_value = pruned[0]
    assert out_key.shape == (1, 2, 3, 3)
    assert out_value.shape == (1, 2, 3, 3)
    assert torch.equal(out_key[:, :, 1, :], key[:, :, 2, :])
