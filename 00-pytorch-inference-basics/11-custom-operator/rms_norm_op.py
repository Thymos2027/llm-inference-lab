"""Register an RMSNorm operator and validate eager plus compiled calls."""

import torch


namespace = torch.library.Library("inference_lab", "DEF")
namespace.define("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor")


@torch.library.impl(namespace, "rms_norm", "CompositeExplicitAutograd")
def rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.shape[-1] != weight.numel():
        raise RuntimeError("weight size must equal x.shape[-1]")
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(x.dtype)


@torch.library.register_fake("inference_lab::rms_norm")
def rms_norm_fake(x: torch.Tensor, weight: torch.Tensor, _eps: float) -> torch.Tensor:
    if x.shape[-1] != weight.numel():
        raise RuntimeError("weight size must equal x.shape[-1]")
    return torch.empty_like(x)


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def call_operator(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.inference_lab.rms_norm.default(x, weight, 1e-6)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)

    eager = call_operator(x, weight)
    expected = reference(x, weight, 1e-6)
    torch.testing.assert_close(eager, expected)

    compiled = torch.compile(call_operator, backend="eager", fullgraph=True)
    compiled_output = compiled(x, weight)
    torch.testing.assert_close(compiled_output, expected)
    print(f"output: shape={tuple(eager.shape)}, dtype={eager.dtype}, device={eager.device}")
    print("PASS: custom operator matches the reference in eager mode and a captured full graph")


if __name__ == "__main__":
    main()
