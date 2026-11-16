"""Check that the lab environment can use PyTorch and the NVIDIA GPU."""

import torch
from transformers.utils import is_torch_available


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "This lab expects an NVIDIA GPU"
    print(f"Transformers sees PyTorch: {is_torch_available()}")
    assert is_torch_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    right = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device="cuda")
    result = left @ right
    expected = torch.tensor([[2.0, 6.0], [6.0, 12.0]], device="cuda")
    torch.testing.assert_close(result, expected)
    print(f"matrix shape: {tuple(result.shape)}, device: {result.device}")
    print(f"matrix result: {result.cpu().tolist()}")
    print("PASS: PyTorch CUDA matrix multiplication")


if __name__ == "__main__":
    main()
