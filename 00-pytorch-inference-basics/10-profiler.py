"""Use PyTorch Profiler to attribute CPU and CUDA operator time."""

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def workload(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    with record_function("projection_and_activation"):
        return torch.nn.functional.silu(left @ right)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA GPU")

    left = torch.randn(1024, 1024, device="cuda")
    right = torch.randn(1024, 1024, device="cuda")
    for _ in range(3):
        _ = workload(left, right)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(5):
            output = workload(left, right)
        torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))
    keys = {event.key for event in prof.key_averages()}
    assert any("mm" in key for key in keys)
    print("PASS: profiler captured CPU and CUDA activity for matmul plus activation")


if __name__ == "__main__":
    main()
