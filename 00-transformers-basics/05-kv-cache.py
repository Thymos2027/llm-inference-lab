"""Fifth Transformers experiment: compare generation with and without a KV cache."""

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
        attn_implementation="eager",
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    prompt = "The capital of France is"
    max_new_tokens = 4
    model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    full_input_ids = model_inputs["input_ids"].clone()
    attention_mask = model_inputs["attention_mask"].clone()
    cached_input_ids = full_input_ids
    past_key_values = None
    new_token_ids = []
    full_positions = 0
    cached_positions = 0
    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    print(f"prompt: {prompt!r}")
    print(f"prompt token IDs: {full_input_ids[0].tolist()}")
    print(f"device: {device}, max_new_tokens: {max_new_tokens}")
    print("step | no-cache input | cached input | tokens already cached | next token")

    with torch.inference_mode():
        for step in range(max_new_tokens):
            full_length = full_input_ids.shape[1]
            cached_length = cached_input_ids.shape[1]
            cached_before = 0 if past_key_values is None else past_key_values.get_seq_length()
            assert cached_before + cached_length == full_length

            no_cache_outputs = model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            cached_outputs = model(
                input_ids=cached_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = cached_outputs.past_key_values
            assert past_key_values is not None
            assert past_key_values.get_seq_length() == full_length

            no_cache_next = no_cache_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cached_next = cached_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            assert torch.equal(no_cache_next, cached_next)
            token_id = cached_next.item()
            new_token_ids.append(token_id)
            full_positions += full_length
            cached_positions += cached_length
            print(
                f"{step + 1:>4} | {full_length:>14} | {cached_length:>12} | "
                f"{cached_before:>21} | {token_id} {tokenizer.decode([token_id])!r}"
            )

            # Both paths generate the same token; only their next forward inputs differ.
            full_input_ids = torch.cat((full_input_ids, cached_next), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(cached_next)), dim=1)
            cached_input_ids = cached_next
            if eos_token_ids is not None and token_id in eos_token_ids:
                break

    print(f"new token IDs: {new_token_ids}")
    print(f"continuation: {tokenizer.decode(new_token_ids)!r}")
    print(f"input token positions sent to model: no cache={full_positions}, cache={cached_positions}")
    print("The cached path still attends to earlier tokens through their stored keys and values.")
    assert attention_mask.shape == full_input_ids.shape
    print("PASS: both paths choose the same tokens; cached decoding reuses past keys and values")


if __name__ == "__main__":
    main()
