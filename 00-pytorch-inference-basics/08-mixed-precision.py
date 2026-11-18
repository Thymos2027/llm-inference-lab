"""Compare FP32, FP16, and BF16 matmul error, storage, and CUDA time."""

import torch


def timed_matmul(left: torch.Tensor, right: torch.Tensor, repeats: int = 30) -> tuple[torch.Tensor, float]:
    for _ in range(5):
        output = left @ right
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = left @ right
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end) / repeats


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA GPU")

    torch.manual_seed(0)
    left_fp32 = torch.randn(1024, 1024, device="cuda")
    right_fp32 = torch.randn(1024, 1024, device="cuda")
    reference, fp32_ms = timed_matmul(left_fp32, right_fp32)

    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        left = left_fp32.to(dtype)
        right = right_fp32.to(dtype)
        output, elapsed_ms = timed_matmul(left, right)
        error = (output.float() - reference).abs()
        rows.append((str(dtype), left.numel() * left.element_size(), error.max().item(), error.mean().item(), elapsed_ms))

    print(f"{'dtype':<15} {'one input':>12} {'max error':>12} {'mean error':>12} {'time':>10}")
    print(f"{'torch.float32':<15} {left_fp32.numel() * left_fp32.element_size():>12,} {0:>12.3e} {0:>12.3e} {fp32_ms:>8.3f} ms")
    for dtype, byte_count, max_error, mean_error, elapsed_ms in rows:
        print(f"{dtype:<15} {byte_count:>12,} {max_error:>12.3e} {mean_error:>12.3e} {elapsed_ms:>8.3f} ms")
    assert all(byte_count * 2 == left_fp32.numel() * left_fp32.element_size() for _, byte_count, *_ in rows)
    print("PASS: precision comparisons use the same inputs, FP32 reference, warmup, and timing region")


if __name__ == "__main__":
    main()
