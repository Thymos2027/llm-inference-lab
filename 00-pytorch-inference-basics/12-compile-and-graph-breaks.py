"""Expose a graph break caused by Tensor.item(), then keep control flow in the graph."""

import torch


def graph_breaking_gate(x: torch.Tensor) -> torch.Tensor:
    if x.mean().item() > 0:
        return x * 2
    return x - 2


def graph_friendly_gate(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.mean() > 0, x * 2, x - 2)


def main() -> None:
    x = torch.tensor([1.0, -0.25, 0.5])
    failed_as_expected = False
    try:
        compiled_bad = torch.compile(graph_breaking_gate, backend="eager", fullgraph=True)
        compiled_bad(x)
    except Exception as error:  # Dynamo error classes are intentionally not part of this lesson's contract.
        failed_as_expected = True
        print(f"fullgraph rejected Tensor.item(): {type(error).__name__}")

    assert failed_as_expected, "Tensor.item() should prevent this function from being one full graph"
    compiled_good = torch.compile(graph_friendly_gate, backend="eager", fullgraph=True)
    expected = graph_friendly_gate(x)
    actual = compiled_good(x)
    torch.testing.assert_close(actual, expected)
    print(f"captured output: {actual.tolist()}")
    print("PASS: Python scalar control flow was replaced with tensor control flow")


if __name__ == "__main__":
    main()
