"""Unroll Qwen3's first decoder layer: causal GQA attention and gated FFN."""

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def describe(name: str, x: torch.Tensor) -> None:
    print(f"{name:<24} shape={tuple(x.shape)!s:<22} dtype={x.dtype}")


def report_dense_compute(model, layer, batch: int, seq_len: int) -> None:
    """Count dense multiply-adds; exclude norms, softmax, activations and kernel overhead."""
    attn = layer.self_attn
    attention_weights = sum(
        projection.weight.numel()
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj)
    )
    ffn_weights = sum(
        projection.weight.numel()
        for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
    )
    projection_macs = batch * seq_len * attention_weights
    ffn_macs = batch * seq_len * ffn_weights
    # QK^T and attention_weights @ V each have this many multiply-adds.
    attention_macs = 2 * batch * model.config.num_attention_heads * seq_len**2 * attn.head_dim
    layer_flops = 2 * (projection_macs + ffn_macs + attention_macs)
    lm_head_flops = 2 * batch * seq_len * model.lm_head.weight.numel()

    print("\nDense computation estimate (1 multiply-add = about 2 FLOPs):")
    print(f"  one layer Q/K/V/o_proj:  {2 * projection_macs / 1e6:9.3f} MFLOPs")
    print(f"  one layer QK^T and w@V: {2 * attention_macs / 1e6:9.3f} MFLOPs")
    print(f"  one layer FFN:           {2 * ffn_macs / 1e6:9.3f} MFLOPs")
    print(f"  all {len(model.model.layers)} layers:         {len(model.model.layers) * layer_flops / 1e9:9.3f} GFLOPs")
    print(f"  lm_head for {seq_len} positions:      {lm_head_flops / 1e9:9.3f} GFLOPs")
    print(f"  lm_head for last only:  {2 * batch * model.lm_head.weight.numel() / 1e9:9.3f} GFLOPs")
    layer_weight_mb = (attention_weights + ffn_weights) * attn.q_proj.weight.element_size() / 1e6
    print(f"  one layer linear weights: {layer_weight_mb:.2f} MB (BF16 on this model)")
    print("  Estimates omit normalization, softmax, activations, memory access and launch overhead.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "huggingface" / "Qwen3-0.6B")
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", attn_implementation="eager", local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if input_ids.shape[1] < 2:
        parser.error("Use a prompt that produces at least two tokens to see the causal mask.")

    layer = model.model.layers[0]
    attn = layer.self_attn
    batch, seq_len = input_ids.shape
    head_dim = attn.head_dim
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads

    print(f"prompt: {ascii(args.prompt)}")
    print(f"token IDs: {input_ids[0].tolist()}")
    print(f"tokens: {ascii(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))}")
    print(f"device={device}, Q heads={num_q_heads}, KV heads={num_kv_heads}, head_dim={head_dim}")

    captured = {}

    def capture_layer(_module, _inputs, output):
        captured["layer_0"] = output.detach().clone()

    def capture_attention(_module, _inputs, output):
        captured["attention_output"] = output[0].detach().clone()

    layer_hook = layer.register_forward_hook(capture_layer)
    attention_hook = attn.register_forward_hook(capture_attention)
    try:
        with torch.inference_mode():
            reference = model(input_ids=input_ids, use_cache=False)
    finally:
        attention_hook.remove()
        layer_hook.remove()

    with torch.inference_mode():
        # Token IDs are row indices into the learned embedding table.
        x = model.model.embed_tokens(input_ids)
        describe("embedding x", x)
        residual = x
        normed = layer.input_layernorm(x)
        describe("RMSNorm(x)", normed)

        # Qwen3 normalizes Q and K after projection, before rotary position encoding.
        q = attn.q_proj(normed).view(batch, seq_len, num_q_heads, head_dim)
        k = attn.k_proj(normed).view(batch, seq_len, num_kv_heads, head_dim)
        v = attn.v_proj(normed).view(batch, seq_len, num_kv_heads, head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        describe("Q", q)
        describe("K", k)
        describe("V", v)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(x, position_ids)
        q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
        k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
        describe("RoPE(Q)", q)
        describe("RoPE(K)", k)

        # Grouped-query attention: each pair of Q heads shares one K/V head.
        k_expanded = k.repeat_interleave(group_size, dim=1)
        v_expanded = v.repeat_interleave(group_size, dim=1)
        describe("K after GQA repeat", k_expanded)

        scores = (q @ k_expanded.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        describe("QK^T / sqrt(d)", scores)
        # No padding in this single-prompt experiment. The upper triangle is future tokens.
        future = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        describe("causal softmax", weights)
        print("head 0 attention weights, rows=query positions, columns=key positions:")
        for row in weights[0, 0].float().cpu().tolist():
            print("  " + " ".join(f"{value:.3f}" for value in row))
        assert torch.all(weights.masked_select(future[None, None]) == 0)
        torch.testing.assert_close(weights.float().sum(-1), torch.ones_like(weights[:, :, :, 0]).float(), atol=0.01, rtol=0)

        context = weights @ v_expanded
        describe("weighted V", context)
        # Return from [batch, heads, positions, head_dim] to one vector per position.
        context = context.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        describe("concatenated heads", context)
        print(f"o_proj weight shape: {tuple(attn.o_proj.weight.shape)} [output, input]")
        attention_output = attn.o_proj(context)
        describe("o_proj output", attention_output)
        assert context.shape[-1] == attn.o_proj.in_features
        assert attention_output.shape == residual.shape
        attention_difference = (attention_output.float() - captured["attention_output"].float()).abs().max().item()
        torch.testing.assert_close(attention_output, captured["attention_output"], atol=0.03, rtol=0.01)
        after_attention = residual + attention_output
        describe("x + attention output", after_attention)

        ffn_input = layer.post_attention_layernorm(after_attention)
        gate = layer.mlp.gate_proj(ffn_input)
        up = layer.mlp.up_proj(ffn_input)
        describe("FFN gate projection", gate)
        describe("FFN up projection", up)
        gated = torch.nn.functional.silu(gate) * up
        describe("SiLU(gate) * up", gated)
        ffn_output = layer.mlp.down_proj(gated)
        manual_layer_output = after_attention + ffn_output
        describe("layer output", manual_layer_output)

        # The hook observed this same layer inside a normal full-model forward pass.
        layer_difference = (manual_layer_output.float() - captured["layer_0"].float()).abs().max().item()
        torch.testing.assert_close(manual_layer_output, captured["layer_0"], atol=0.03, rtol=0.01)
        next_token_id = reference.logits[0, -1].argmax().item()
        print(f"max absolute difference vs Transformers attention output: {attention_difference:.6f}")
        print(f"max absolute difference vs Transformers layer 0: {layer_difference:.6f}")
        print(f"full model: {len(model.model.layers)} decoder layers -> final RMSNorm -> lm_head")
        print(f"next token from full model: {next_token_id} {ascii(tokenizer.decode([next_token_id]))}")
        report_dense_compute(model, layer, batch, seq_len)
        print("PASS: manual attention + FFN matches the actual first decoder layer")


if __name__ == "__main__":
    main()
