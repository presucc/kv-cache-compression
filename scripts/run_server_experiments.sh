#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-EleutherAI/pythia-70m}"
RESULT_DIR="${RESULT_DIR:-results/server}"
PG19_TEXT="${PG19_TEXT:-data/pg19_raw/test/10146.txt}"
WIKITEXT_TEXT="${WIKITEXT_TEXT:-data/wikitext_validation.txt}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
SINK_SIZE="${SINK_SIZE:-4}"
IMPORTANT_SIZE="${IMPORTANT_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
LATENCY_DEVICE="${LATENCY_DEVICE:-cuda}"
LATENCY_DTYPE="${LATENCY_DTYPE:-float16}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SKIP_LATENCY="${SKIP_LATENCY:-0}"
RUN_BUDGET_SWEEP="${RUN_BUDGET_SWEEP:-1}"
RUN_FULL_SUITE="${RUN_FULL_SUITE:-1}"
RUN_FULL_LATENCY="${RUN_FULL_LATENCY:-1}"

export PYTHONPATH="${PYTHONPATH:-src}"
mkdir -p "$RESULT_DIR"

main_methods=(dense streamingllm sink_snapkv pyramid_sinkkv)
allocation_methods=(sink_snapkv pyramid_sinkkv reverse_pyramid_sinkkv)
all_methods=(
  dense
  sliding_window
  streamingllm
  lm_infinite
  h2o
  scissorhands
  tova
  snapkv
  pyramidkv
  sink_snapkv
  pyramid_sinkkv
  reverse_pyramid_sinkkv
)

dataset_args() {
  local dataset="$1"
  local text_file="$2"
  if [[ -f "$text_file" ]]; then
    printf -- "--text-file\n%s\n" "$text_file"
  elif [[ "$dataset" == "pg19" ]]; then
    printf -- "--dataset\npg19\n--split\ntest\n--max-samples\n1\n"
  else
    printf -- "--dataset\nwikitext\n--split\nvalidation\n"
  fi
}

mapfile -t pg19_args < <(dataset_args pg19 "$PG19_TEXT")
mapfile -t wikitext_args < <(dataset_args wikitext "$WIKITEXT_TEXT")

echo "== PG-19 main PPL =="
python scripts/run_ppl.py \
  --model "$MODEL" \
  "${pg19_args[@]}" \
  --max-tokens "$MAX_TOKENS" \
  --methods "${main_methods[@]}" \
  --window-size "$WINDOW_SIZE" \
  --sink-size "$SINK_SIZE" \
  --important-size "$IMPORTANT_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output "$RESULT_DIR/ppl_pg19_main.json"

echo "== Wikitext-2 main PPL =="
python scripts/run_ppl.py \
  --model "$MODEL" \
  "${wikitext_args[@]}" \
  --max-tokens "$MAX_TOKENS" \
  --methods "${main_methods[@]}" \
  --window-size "$WINDOW_SIZE" \
  --sink-size "$SINK_SIZE" \
  --important-size "$IMPORTANT_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output "$RESULT_DIR/ppl_wikitext_main.json"

echo "== PG-19 layer-allocation ablation =="
python scripts/run_ppl.py \
  --model "$MODEL" \
  "${pg19_args[@]}" \
  --max-tokens "$MAX_TOKENS" \
  --methods "${allocation_methods[@]}" \
  --window-size "$WINDOW_SIZE" \
  --sink-size "$SINK_SIZE" \
  --important-size "$IMPORTANT_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output "$RESULT_DIR/ablation_pg19_layer_allocation.json"

echo "== Wikitext-2 layer-allocation ablation =="
python scripts/run_ppl.py \
  --model "$MODEL" \
  "${wikitext_args[@]}" \
  --max-tokens "$MAX_TOKENS" \
  --methods "${allocation_methods[@]}" \
  --window-size "$WINDOW_SIZE" \
  --sink-size "$SINK_SIZE" \
  --important-size "$IMPORTANT_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output "$RESULT_DIR/ablation_wikitext_layer_allocation.json"

if [[ "$RUN_FULL_SUITE" != "0" ]]; then
  echo "== PG-19 full method suite PPL =="
  python scripts/run_ppl.py \
    --model "$MODEL" \
    "${pg19_args[@]}" \
    --max-tokens "$MAX_TOKENS" \
    --methods "${all_methods[@]}" \
    --window-size "$WINDOW_SIZE" \
    --sink-size "$SINK_SIZE" \
    --important-size "$IMPORTANT_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output "$RESULT_DIR/ppl_pg19_all_methods.json"

  echo "== Wikitext-2 full method suite PPL =="
  python scripts/run_ppl.py \
    --model "$MODEL" \
    "${wikitext_args[@]}" \
    --max-tokens "$MAX_TOKENS" \
    --methods "${all_methods[@]}" \
    --window-size "$WINDOW_SIZE" \
    --sink-size "$SINK_SIZE" \
    --important-size "$IMPORTANT_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output "$RESULT_DIR/ppl_wikitext_all_methods.json"
fi

if [[ "$RUN_BUDGET_SWEEP" != "0" ]]; then
  echo "== PG-19 budget sweep =="
  for pair in "128 16" "256 32" "384 48"; do
    read -r budget_window budget_important <<< "$pair"
    python scripts/run_ppl.py \
      --model "$MODEL" \
      "${pg19_args[@]}" \
      --max-tokens "$MAX_TOKENS" \
      --methods sink_snapkv pyramid_sinkkv \
      --window-size "$budget_window" \
      --sink-size "$SINK_SIZE" \
      --important-size "$budget_important" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --output "$RESULT_DIR/budget_pg19_w${budget_window}_i${budget_important}.json"
  done
fi

if [[ "$SKIP_LATENCY" == "0" ]]; then
  echo "== PG-19 GPU latency =="
  python scripts/run_latency.py \
    --model "$MODEL" \
    "${pg19_args[@]}" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --methods "${main_methods[@]}" \
    --window-size "$WINDOW_SIZE" \
    --sink-size "$SINK_SIZE" \
    --important-size "$IMPORTANT_SIZE" \
    --device "$LATENCY_DEVICE" \
    --dtype "$LATENCY_DTYPE" \
    --output "$RESULT_DIR/latency_pg19_main.json"

  if [[ "$RUN_FULL_SUITE" != "0" && "$RUN_FULL_LATENCY" != "0" ]]; then
    echo "== PG-19 full method suite GPU latency =="
    python scripts/run_latency.py \
      --model "$MODEL" \
      "${pg19_args[@]}" \
      --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --methods "${all_methods[@]}" \
      --window-size "$WINDOW_SIZE" \
      --sink-size "$SINK_SIZE" \
      --important-size "$IMPORTANT_SIZE" \
      --device "$LATENCY_DEVICE" \
      --dtype "$LATENCY_DTYPE" \
      --output "$RESULT_DIR/latency_pg19_all_methods.json"
  fi
fi

echo "== Summary =="
bash scripts/summarize_server_results.sh "$RESULT_DIR"
