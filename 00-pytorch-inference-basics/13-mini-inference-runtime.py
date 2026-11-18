"""Build a tiny causal decoder and compare full recompute with KV-cached decode."""

import math

import torch
from torch import nn


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 32, heads: int = 4) -> None:
        super().__init__()
        assert hidden_size % heads == 0
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _hidden = x.shape
        return x.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(input_ids)
        query = self._split_heads(self.q_proj(x))
        new_key = self._split_heads(self.k_proj(x))
        new_value = self._split_heads(self.v_proj(x))
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if past_key_value is None:
            key, value = new_key, new_value
        else:
            key = torch.cat((past_key_value[0], new_key), dim=-2)
            value = torch.cat((past_key_value[1], new_value), dim=-2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_length + torch.arange(input_ids.shape[1], device=input_ids.device)
        key_positions = torch.arange(key.shape[-2], device=input_ids.device)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal.view(1, 1, *causal.shape), float("-inf"))
        context = torch.softmax(scores, dim=-1) @ value
        context = context.transpose(1, 2).contiguous().view(*x.shape)
        logits = self.lm_head(self.o_proj(context))
        return logits, (key, value)


def generate_full(model: TinyDecoder, prompt: torch.Tensor, steps: int) -> tuple[torch.Tensor, list[int]]:
    sequence = prompt.clone()
    submitted = []
    for _ in range(steps):
        submitted.append(sequence.shape[1])
        logits, _cache = model(sequence)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
    return sequence, submitted


def generate_cached(model: TinyDecoder, prompt: torch.Tensor, steps: int) -> tuple[torch.Tensor, list[int]]:
    sequence = prompt.clone()
    submitted = [prompt.shape[1]]
    logits, cache = model(prompt)
    for step in range(steps):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
        if step + 1 < steps:
            submitted.append(1)
            logits, cache = model(next_token, cache)
    return sequence, submitted


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    model = TinyDecoder().to(device).eval()
    prompt = torch.tensor([[1, 4, 9, 3, 7]], device=device)
    steps = 6

    with torch.inference_mode():
        full_sequence, full_submitted = generate_full(model, prompt, steps)
        cached_sequence, cached_submitted = generate_cached(model, prompt, steps)

    torch.testing.assert_close(cached_sequence, full_sequence)
    assert full_submitted == [5, 6, 7, 8, 9, 10]
    assert cached_submitted == [5, 1, 1, 1, 1, 1]
    print(f"generated token IDs: {full_sequence[:, prompt.shape[1]:].cpu().tolist()[0]}")
    print(f"full recompute input lengths: {full_submitted} (total={sum(full_submitted)})")
    print(f"KV-cached input lengths:      {cached_submitted} (total={sum(cached_submitted)})")
    print("PASS: cached decode generated identical tokens while submitting fewer token positions")


if __name__ == "__main__":
    main()
