"""Second Transformers experiment: inspect model config and batched input tensors."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "huggingface" / "Qwen3-0.6B",
        help="Local model directory containing config.json and tokenizer files",
    )
    args = parser.parse_args()

    with (args.model / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    print("=== Model config (no weights loaded) ===")
    for key in (
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "torch_dtype",
    ):
        print(f"{key}: {config[key]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = ["Hello, world!", "What is a GPU?"]
    batch = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    print("\n=== Tokenizer output on CPU ===")
    print(f"input_ids: shape={tuple(input_ids.shape)}, dtype={input_ids.dtype}, device={input_ids.device}")
    print(
        f"attention_mask: shape={tuple(attention_mask.shape)}, "
        f"dtype={attention_mask.dtype}, device={attention_mask.device}"
    )
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    for index, text in enumerate(texts):
        print(f"row {index}: text={text!r}, ids={input_ids[index].tolist()}")
        print(f"       mask={attention_mask[index].tolist()}, real tokens={attention_mask[index].sum().item()}")

    assert input_ids.ndim == 2 and input_ids.shape[0] == len(texts)
    assert input_ids.shape == attention_mask.shape
    assert input_ids.dtype == torch.long
    assert bool((input_ids[attention_mask == 0] == tokenizer.pad_token_id).all())
    assert int(input_ids.max()) < config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = {name: tensor.to(device) for name, tensor in batch.items()}
    print("\n=== Inputs moved to compute device ===")
    for name, tensor in model_inputs.items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    print(
        "If a model were loaded, its token embeddings would have shape "
        f"({input_ids.shape[0]}, {input_ids.shape[1]}, {config['hidden_size']})."
    )
    print("config torch_dtype describes model weights; token IDs remain integer indices.")
    print("PASS: config, batch shape, padding, dtype, and device")


if __name__ == "__main__":
    main()
