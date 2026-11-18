"""Pad variable-length requests and validate masked batched computation."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def main() -> None:
    torch.manual_seed(0)
    requests = [torch.tensor([2, 5, 7]), torch.tensor([4, 3, 9, 8, 1]), torch.tensor([6, 2])]
    padded = pad_sequence(requests, batch_first=True, padding_value=0)
    lengths = torch.tensor([request.numel() for request in requests])
    positions = torch.arange(padded.shape[1]).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)

    embedding = nn.Embedding(16, 8, padding_idx=0)
    token_states = embedding(padded)
    batched_mean = (token_states * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
    individual_mean = torch.stack([embedding(request).mean(dim=0) for request in requests])
    torch.testing.assert_close(batched_mean, individual_mean)

    real_tokens = int(lengths.sum())
    padded_slots = padded.numel()
    waste = 1 - real_tokens / padded_slots
    print(f"padded IDs:\n{padded}")
    print(f"attention mask:\n{mask.to(torch.int32)}")
    print(f"real tokens={real_tokens}, batch slots={padded_slots}, padding waste={waste:.1%}")
    print("PASS: masked batch result matches processing every request independently")


if __name__ == "__main__":
    main()
