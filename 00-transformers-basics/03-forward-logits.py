"""Third Transformers experiment: run one forward pass and inspect next-token logits."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "huggingface" / "Qwen3-0.6B",
        help="Local model directory containing weights and tokenizer files",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    prompt = "The capital of France is"
    model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    input_ids = model_inputs["input_ids"]

    print(f"prompt: {prompt!r}")
    print(f"input_ids: {input_ids[0].tolist()}")
    print(f"input_ids shape: {tuple(input_ids.shape)}, dtype: {input_ids.dtype}, device: {input_ids.device}")
    print(f"model weight dtype: {next(model.parameters()).dtype}, device: {device}")

    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits
    next_token_scores = logits[0, -1, :]
    top_scores, top_ids = torch.topk(next_token_scores, k=5)

    print(f"logits shape: {tuple(logits.shape)}, dtype: {logits.dtype}, device: {logits.device}")
    print(f"last-position scores shape: {tuple(next_token_scores.shape)}")
    print("top 5 next-token candidates (raw logits, not probabilities):")
    for token_id, score in zip(top_ids.tolist(), top_scores.tolist()):
        piece = tokenizer.decode([token_id])
        print(f"  id={token_id:6d} score={score:9.4f} decoded={piece!r}")

    assert tuple(logits.shape) == (1, input_ids.shape[1], model.config.vocab_size)
    assert next_token_scores.shape == (model.config.vocab_size,)
    assert input_ids.dtype == torch.long
    print("PASS: one forward pass and next-token logits")


if __name__ == "__main__":
    main()
