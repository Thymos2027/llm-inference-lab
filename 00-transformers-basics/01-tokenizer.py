"""First Transformers experiment: inspect text, token IDs, and chat formatting."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "huggingface" / "Qwen3-0.6B",
        help="Local model directory containing tokenizer files",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    text = "Hello, world!"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    decoded = tokenizer.decode(token_ids)

    print("=== Plain text ===")
    print(f"text: {text!r}")
    for index, (token_id, piece) in enumerate(zip(token_ids, pieces)):
        print(f"token {index}: id={token_id} piece={ascii(piece)}")
    print(f"decoded: {decoded!r}")
    assert decoded == text

    messages = [{"role": "user", "content": "What is a GPU?"}]
    options = {"add_generation_prompt": True, "enable_thinking": False}
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    chat_encoded = tokenizer.apply_chat_template(messages, tokenize=True, **options)
    chat_token_ids = chat_encoded["input_ids"] if hasattr(chat_encoded, "keys") else chat_encoded

    print("\n=== Chat input ===")
    print(f"messages: {messages!r}")
    print(f"formatted text: {ascii(chat_text)}")
    print(f"token IDs: {chat_token_ids}")
    print(f"token count: {len(chat_token_ids)}")
    if hasattr(chat_encoded, "keys"):
        print(f"attention mask: {chat_encoded.get('attention_mask')}")
    assert chat_token_ids == tokenizer.encode(chat_text, add_special_tokens=False)
    print("PASS: text round trip and chat template tokenization")


if __name__ == "__main__":
    main()
