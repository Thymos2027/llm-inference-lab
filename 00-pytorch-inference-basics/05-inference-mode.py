"""Separate Module.eval() from no_grad() and inference_mode()."""

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
    x = torch.randn(2, 8, requires_grad=True)

    model.eval()
    eval_output = model(x)
    assert eval_output.requires_grad

    with torch.no_grad():
        no_grad_output = model(x)
    with torch.inference_mode():
        inference_output = model(x)

    torch.testing.assert_close(eval_output, no_grad_output)
    torch.testing.assert_close(eval_output, inference_output)
    assert not no_grad_output.requires_grad and not inference_output.requires_grad

    model.train()
    train_a = model(x)
    train_b = model(x)
    assert not torch.equal(train_a, train_b), "Dropout should be stochastic in train mode"

    print(f"eval output requires_grad:       {eval_output.requires_grad}")
    print(f"no_grad output requires_grad:    {no_grad_output.requires_grad}")
    print(f"inference output requires_grad:  {inference_output.requires_grad}")
    print("PASS: eval controls module behavior; no_grad/inference_mode control Autograd recording")


if __name__ == "__main__":
    main()
