"""Measure CUDA work with synchronization and CUDA Events."""

from time import perf_counter

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA GPU")

    torch.manual_seed(0)
    left = torch.randn(1024, 1024, device="cuda")
    right = torch.randn(1024, 1024, device="cuda")
    for _ in range(5):
        _ = left @ right
    torch.cuda.synchronize()

    start = perf_counter()
    output = left @ right
    dispatch_ms = (perf_counter() - start) * 1e3
    torch.cuda.synchronize()
    synchronized_ms = (perf_counter() - start) * 1e3

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(20):
        output = left @ right
    end_event.record()
    end_event.synchronize()
    event_ms = start_event.elapsed_time(end_event) / 20

    assert torch.isfinite(output).all()
    print(f"CPU time before synchronization: {dispatch_ms:.3f} ms (mostly enqueue overhead)")
    print(f"CPU time including synchronization: {synchronized_ms:.3f} ms")
    print(f"CUDA Event average kernel region: {event_ms:.3f} ms")
    print("PASS: GPU work was warmed up and timed with an explicit synchronization boundary")


if __name__ == "__main__":
    main()
