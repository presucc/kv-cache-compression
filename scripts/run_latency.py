from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_CHOICES = [
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
]

DEFAULT_METHODS = ["dense", "streamingllm", "sink_snapkv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure generation latency under KV cache compression.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=METHOD_CHOICES,
    )
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--important-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results/latency.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from tqdm import tqdm

    from llm_kv_compression.data import load_text_corpus
    from llm_kv_compression.evaluation import configs_for_methods, evaluate_latency, result_to_dict, warmup_model
    from llm_kv_compression.modeling import load_model_and_tokenizer

    if args.prompt is None:
        prompt = load_text_corpus(
            dataset=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            max_chars=args.max_chars,
            text_file=args.text_file,
        )
    else:
        prompt = args.prompt

    model, tokenizer, device = load_model_and_tokenizer(args.model, device=args.device, dtype=args.dtype)
    warmup_model(model, tokenizer, prompt, device)

    results = []
    configs = configs_for_methods(args.methods, args.window_size, args.sink_size, args.important_size)
    for config in tqdm(configs, desc="methods"):
        result = evaluate_latency(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            device=device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        results.append(result_to_dict(result))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
