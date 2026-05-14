from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def disable_optional_vision_backends() -> None:
    """Avoid broken optional torchvision installs for text-only experiments."""

    try:
        import transformers.utils.import_utils as import_utils

        import_utils._torchvision_available = False
        import_utils._torchvision_version = "N/A"
    except Exception:
        pass


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def pick_dtype(dtype: str, device: torch.device):
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model_and_tokenizer(model_name: str, device: str = "auto", dtype: str = "auto"):
    disable_optional_vision_backends()
    resolved_device = pick_device(device)
    torch_dtype = pick_dtype(dtype, resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch_dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.to(resolved_device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, resolved_device
