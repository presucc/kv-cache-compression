#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:-results/server}"
SUMMARY_FILE="${SUMMARY_FILE:-$RESULT_DIR/summary.md}"

export PYTHONPATH="${PYTHONPATH:-src}"
mkdir -p "$RESULT_DIR"

append_table() {
  local title="$1"
  local file="$2"
  shift 2
  echo
  echo "## $title"
  echo
  if [[ -f "$file" ]]; then
    python scripts/summarize_results.py "$file" --columns "$@"
  else
    echo "Missing: $file"
  fi
}

{
  echo "# Server Experiment Summary"
  echo
  echo "Result directory: \`$RESULT_DIR\`"

  append_table "PG-19 Main PPL" "$RESULT_DIR/ppl_pg19_main.json" \
    method ppl max_retained_tokens average_retained_tokens nominal_budget

  append_table "Wikitext-2 Main PPL" "$RESULT_DIR/ppl_wikitext_main.json" \
    method ppl max_retained_tokens average_retained_tokens nominal_budget

  append_table "PG-19 All Methods PPL" "$RESULT_DIR/ppl_pg19_all_methods.json" \
    method ppl max_retained_tokens average_retained_tokens nominal_budget

  append_table "Wikitext-2 All Methods PPL" "$RESULT_DIR/ppl_wikitext_all_methods.json" \
    method ppl max_retained_tokens average_retained_tokens nominal_budget

  append_table "PG-19 Latency" "$RESULT_DIR/latency_pg19_main.json" \
    method ttft_seconds tpot_seconds throughput_tokens_per_second peak_cuda_memory_mb max_retained_tokens average_retained_tokens

  append_table "PG-19 All Methods Latency" "$RESULT_DIR/latency_pg19_all_methods.json" \
    method ttft_seconds tpot_seconds throughput_tokens_per_second peak_cuda_memory_mb max_retained_tokens average_retained_tokens
} | tee "$SUMMARY_FILE"
