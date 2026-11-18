"""Inspect Module parameters, buffers, hooks, eval mode, and state_dict loading."""

import torch
from torch import nn


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.register_buffer("scale", torch.tensor(0.5), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(x)) * self.scale


def main() -> None:
    torch.manual_seed(0)
    model = TinyBlock()
    observed_shapes: list[tuple[int, ...]] = []

    def observe(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed_shapes.append(tuple(output.shape))

    handle = model.proj.register_forward_hook(observe)
    x = torch.randn(2, 8)
    model.eval()
    output = model(x)
    handle.remove()

    state = {name: value.clone() for name, value in model.state_dict().items()}
    restored = TinyBlock()
    restored.load_state_dict(state, strict=True)
    restored.eval()
    torch.testing.assert_close(restored(x), output)

    print(f"parameters: {[(name, tuple(value.shape)) for name, value in model.named_parameters()]}")
    print(f"buffers:    {[(name, tuple(value.shape)) for name, value in model.named_buffers()]}")
    print(f"state_dict: {list(state)}")
    print(f"hook observed output shapes: {observed_shapes}")
    assert observed_shapes == [(2, 8)] and "scale" in state
    print("PASS: parameters and buffers round-trip through state_dict; hook observed the forward pass")


if __name__ == "__main__":
    main()
