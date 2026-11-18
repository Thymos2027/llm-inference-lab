"""Inspect Tensor views, storage, strides, and contiguous copies."""

import torch


def describe(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name:<12} shape={tuple(tensor.shape)!s:<10} stride={tensor.stride()!s:<10} "
        f"contiguous={str(tensor.is_contiguous()):<5} data_ptr={tensor.data_ptr()}"
    )


def main() -> None:
    storage = torch.arange(12, dtype=torch.float32)
    matrix = storage.view(3, 4)
    transposed = matrix.t()
    sliced = matrix[:, ::2]
    copied = transposed.contiguous()

    for name, tensor in (
        ("storage", storage),
        ("matrix", matrix),
        ("transposed", transposed),
        ("sliced", sliced),
        ("copied", copied),
    ):
        describe(name, tensor)

    assert storage.untyped_storage().data_ptr() == matrix.untyped_storage().data_ptr()
    assert matrix.untyped_storage().data_ptr() == transposed.untyped_storage().data_ptr()
    assert matrix.untyped_storage().data_ptr() == sliced.untyped_storage().data_ptr()
    assert copied.untyped_storage().data_ptr() != matrix.untyped_storage().data_ptr()
    assert not transposed.is_contiguous() and copied.is_contiguous()
    torch.testing.assert_close(copied, transposed)

    before = matrix[0, 0].item()
    transposed[0, 0] = -1
    assert matrix[0, 0].item() == -1 and before == 0
    assert copied[0, 0].item() == 0
    print("PASS: views share storage; contiguous() materializes an independent copy")


if __name__ == "__main__":
    main()
