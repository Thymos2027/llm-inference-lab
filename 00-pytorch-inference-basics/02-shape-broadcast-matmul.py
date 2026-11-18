"""Derive batched attention shapes and inspect broadcasted views."""

import math

import torch


def main() -> None:
    torch.manual_seed(0)
    batch, heads, query_length, key_length, head_dim = 2, 4, 3, 5, 8
    query = torch.randn(batch, heads, query_length, head_dim)
    key = torch.randn(batch, heads, key_length, head_dim)

    scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
    reference = torch.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(head_dim)
    torch.testing.assert_close(scores, reference)
    assert scores.shape == (batch, heads, query_length, key_length)

    per_head_bias = torch.randn(1, heads, 1, key_length)
    expanded_bias = per_head_bias.expand(batch, heads, query_length, key_length)
    biased_scores = scores + per_head_bias
    torch.testing.assert_close(biased_scores, scores + expanded_bias)

    print(f"query:         {tuple(query.shape)}")
    print(f"key.T:         {tuple(key.transpose(-2, -1).shape)}")
    print(f"scores:        {tuple(scores.shape)}")
    print(f"bias:          {tuple(per_head_bias.shape)}, stride={per_head_bias.stride()}")
    print(f"expanded bias: {tuple(expanded_bias.shape)}, stride={expanded_bias.stride()}")
    assert 0 in expanded_bias.stride(), "expanded dimensions should use stride 0"
    print("PASS: batched matmul matches einsum; expand broadcasts without materializing repeats")


if __name__ == "__main__":
    main()
