"""Compare device transfers, dtypes, byte sizes, and contiguous layouts."""

import torch


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.linspace(-1, 1, 256 * 256, dtype=torch.float32).view(256, 256)
    non_contiguous = cpu.t()
    compute = non_contiguous.to(device=device, dtype=torch.bfloat16)

    print(f"source:  device={cpu.device}, dtype={cpu.dtype}, stride={cpu.stride()}")
    print(
        f"compute: device={compute.device}, dtype={compute.dtype}, "
        f"stride={compute.stride()}, contiguous={compute.is_contiguous()}"
    )
    print(f"FP32 bytes={tensor_bytes(cpu):,}; BF16 bytes={tensor_bytes(compute):,}")
    assert tensor_bytes(compute) * 2 == tensor_bytes(cpu)

    contiguous = compute.contiguous()
    assert contiguous.is_contiguous()
    result = contiguous @ contiguous.t()
    reference = non_contiguous @ non_contiguous.t()
    torch.testing.assert_close(result.float().cpu(), reference, rtol=2e-2, atol=2e-2)
    print(f"result: shape={tuple(result.shape)}, dtype={result.dtype}, device={result.device}")
    print("PASS: device, dtype, byte size, layout, and BF16 error are explicit")


if __name__ == "__main__":
    main()
