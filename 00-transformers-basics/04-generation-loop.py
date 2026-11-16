"""Fourth Transformers experiment: build a greedy generation loop and compare it with generate()."""

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
    max_new_tokens = 8
    model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    prompt_ids = model_inputs["input_ids"]
    input_ids = prompt_ids.clone()
    attention_mask = model_inputs["attention_mask"].clone()
    new_token_ids = []
    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    print(f"prompt: {prompt!r}")
    print(f"prompt token IDs: {prompt_ids[0].tolist()}")
    print(f"device: {device}, max_new_tokens: {max_new_tokens}, use_cache: False")
    print("=== Manual greedy loop ===")
    with torch.inference_mode():
        for step in range(max_new_tokens):
            # Recompute the full sequence; KV cache is the subject of experiment 05.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token_id.item()
            new_token_ids.append(token_id)
            print(
                f"step {step + 1}: input length={input_ids.shape[1]}, "
                f"next ID={token_id}, decoded={tokenizer.decode([token_id])!r}"
            )
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token_id)), dim=1)
            if eos_token_ids is not None and token_id in eos_token_ids:
                break

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generate_new_ids = generated_ids[0, prompt_ids.shape[1] :].tolist()
    print("=== Comparison with generate() ===")
    print(f"manual new IDs:   {new_token_ids}")
    print(f"generate new IDs: {generate_new_ids}")
    print(f"manual continuation:   {tokenizer.decode(new_token_ids)!r}")
    print(f"generate continuation: {tokenizer.decode(generate_new_ids)!r}")

    assert input_ids[0].tolist() == prompt_ids[0].tolist() + new_token_ids
    assert attention_mask.shape == input_ids.shape
    assert bool(attention_mask.all())
    assert new_token_ids == generate_new_ids
    print("PASS: manual greedy loop matches generate()")


if __name__ == "__main__":
    main()
