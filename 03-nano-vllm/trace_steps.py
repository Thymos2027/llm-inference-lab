"""Trace one small nano-vLLM run without changing the engine's scheduler."""

import argparse
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "huggingface" / "Qwen3-0.6B",
    )
    args = parser.parse_args()

    llm = LLM(
        str(args.model.resolve()),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
    )
    torch.manual_seed(0)

    prompts = ["The capital of France is", "A GPU is a"]
    max_tokens = [4, 6]
    for prompt, limit in zip(prompts, max_tokens):
        llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=limit, ignore_eos=True))

    scheduler = llm.scheduler
    blocks = scheduler.block_manager
    print(f"model={args.model.resolve()}")
    print(f"kv_blocks_total={len(blocks.blocks)} block_size={blocks.block_size}")
    for seq in scheduler.waiting:
        print(f"request={seq.seq_id} prompt_tokens={seq.num_prompt_tokens} max_new_tokens={seq.max_tokens}")

    step = 0
    completed = {}
    while not llm.is_finished():
        step += 1
        seqs, is_prefill = scheduler.schedule()
        print(
            f"step={step} phase={'prefill' if is_prefill else 'decode'} "
            f"scheduled={sum(seq.num_scheduled_tokens for seq in seqs)} "
            f"used_blocks={len(blocks.used_block_ids)} free_blocks={len(blocks.free_block_ids)}"
        )
        for seq in seqs:
            print(
                f"  before id={seq.seq_id} cached={seq.num_cached_tokens} "
                f"scheduled={seq.num_scheduled_tokens} blocks={seq.block_table}"
            )

        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        scheduler.postprocess(seqs, token_ids, is_prefill)
        for seq, token_id in zip(seqs, token_ids):
            print(
                f"  after  id={seq.seq_id} sampled={token_id} "
                f"generated={seq.num_completion_tokens}/{seq.max_tokens} "
                f"cached={seq.num_cached_tokens} status={seq.status.name} "
                f"blocks={seq.block_table}"
            )
            if seq.is_finished:
                completed[seq.seq_id] = seq.completion_token_ids
        print(
            f"  queues waiting={len(scheduler.waiting)} running={len(scheduler.running)} "
            f"used_blocks={len(blocks.used_block_ids)}"
        )

    for seq_id, token_ids in sorted(completed.items()):
        print(f"result id={seq_id} tokens={token_ids} text={llm.tokenizer.decode(token_ids)!r}")
    assert [len(ids) for _, ids in sorted(completed.items())] == max_tokens
    assert not blocks.used_block_ids
    print("PASS: requested tokens generated; all KV blocks released")


if __name__ == "__main__":
    main()
