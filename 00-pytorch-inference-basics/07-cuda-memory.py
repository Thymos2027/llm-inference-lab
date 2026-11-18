"""Observe PyTorch's CUDA caching allocator and tensor lifetimes."""

import gc

import torch


MIB = 1024**2


def report(label: str) -> None:
    print(
        f"{label:<18} allocated={torch.cuda.memory_allocated() / MIB:8.1f} MiB  "
        f"reserved={torch.cuda.memory_reserved() / MIB:8.1f} MiB  "
        f"peak={torch.cuda.max_memory_allocated() / MIB:8.1f} MiB"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA GPU")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    report("baseline")

    left = torch.randn(4096, 4096, device="cuda")
    right = torch.randn(4096, 4096, device="cuda")
    report("after inputs")
    output = left + right
    report("after output")
    assert torch.cuda.max_memory_allocated() > baseline + 150 * MIB

    del left, right, output
    gc.collect()
    report("after del")
    assert torch.cuda.memory_allocated() <= baseline + MIB
    reserved_before_empty = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    report("after empty_cache")
    assert torch.cuda.memory_reserved() <= reserved_before_empty
    print("PASS: deleting tensors releases allocated bytes; empty_cache releases cached blocks")


if __name__ == "__main__":
    main()
