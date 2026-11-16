"""First Triton kernel: contiguous FP32 vector addition on a CUDA GPU."""

import argparse

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    in_bounds = offsets < n
    x = tl.load(x_ptr + offsets, mask=in_bounds, other=0)
    y = tl.load(y_ptr + offsets, mask=in_bounds, other=0)
    tl.store(out_ptr + offsets, x + y, mask=in_bounds)


def vector_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, block: int) -> None:
    n = x.numel()
    grid = (triton.cdiv(n, block),)
    vector_add_kernel[grid](x, y, out, n, block)


def make_inputs(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))
    reference = torch.empty_like(x)
    return x, y, out, reference


def check(n: int, block: int) -> None:
    x, y, out, reference = make_inputs(n)
    torch.add(x, y, out=reference)
    vector_add(x, y, out, block)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    max_abs_error = (out - reference).abs().max().item()
    print(f"CHECK n={n:>9} dtype=float32 max_abs_error={max_abs_error:.3g} PASS")


def benchmark(n: int, block: int) -> None:
    x, y, out, reference = make_inputs(n)
    torch.add(x, y, out=reference)
    vector_add(x, y, out, block)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)

    # Both callables write into existing buffers. CUDA Graph capture keeps Python
    # launch and allocation overhead outside the per-kernel timing.
    torch_ms = triton.testing.do_bench_cudagraph(
        lambda: torch.add(x, y, out=reference), rep=20, return_mode="median"
    )
    triton_ms = triton.testing.do_bench_cudagraph(
        lambda: vector_add(x, y, out, block), rep=20, return_mode="median"
    )
    speedup = torch_ms / triton_ms
    effective_gbps = 12 * n / (triton_ms * 1e6)
    print(
        f"BENCH n={n:>9} torch={torch_ms:.5f} ms "
        f"triton={triton_ms:.5f} ms torch/triton={speedup:.2f}x "
        f"triton_effective_bw={effective_gbps:.1f} GB/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="run CUDA Graph timing")
    parser.add_argument("--block", type=int, choices=(128, 256, 512, 1024), default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable")
    torch.manual_seed(2026)
    print(
        f"GPU={torch.cuda.get_device_name(0)} sm={torch.cuda.get_device_capability(0)} "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"triton={triton.__version__} block={args.block}"
    )
    for n in (1, 255, 256, 257, 1003, 4096, 1 << 20):
        check(n, args.block)
    if args.benchmark:
        for n in (1003, 1 << 20, (1 << 24) + 123):
            benchmark(n, args.block)


if __name__ == "__main__":
    main()
