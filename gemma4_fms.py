"""
Gemma4 inference script for FMS on AIU.

NOTE: This script uses the text-only Gemma4Text model.
      Multimodal (image) support requires the Gemma4 vision tower,
      which is not yet implemented in FMS.

Env:
export FLEX_HDMA_P2PSIZE=268435456
export FLEX_HDMA_COLLSIZE=33554432
export VLLM_DT_MAX_BATCH_TKV_LIMIT=131072
export VLLM_DT_CHUNK_LEN=1024
"""

import io
import os
import re
from contextlib import redirect_stdout

from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
import torch
import torch_sendnn
from transformers import AutoProcessor

# Set Required Environment Variable
os.environ.setdefault("COMPILATION_MODE", "offline_decoder")


model_id = "google/gemma-4-12B-it"

PROMPT_SEQUENCE_LENGTHS = [1, 1024, 16 * 1024]

_PROMPT_TEXT = (
    "What action do you think I should take in this situation? "
    "List all the possible actions and explain why you think they are good or bad."
)


def _get_inputs(processor):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": _PROMPT_TEXT},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {k: v.to(torch.device("cpu")) for k, v in inputs.items()}


def _pad_inputs_to_prompt_length(inputs, prompt_sequence_length):
    padded_input_ids, padding_kwargs = pad_input_ids(
        [inputs["input_ids"].squeeze(0)],
        min_pad_length=prompt_sequence_length,
    )
    inputs["input_ids"] = padded_input_ids
    inputs.update(padding_kwargs)
    return inputs


def infer(model, input_ids, inputs, max_new_tokens):
    with torch_sendnn.warmup_mode():
        output = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            max_seq_len=input_ids.shape[1] + max_new_tokens,
            extra_kwargs=inputs,
            timing="per-token",
            contiguous_cache=True,
        )
    return output


def infer_with_timing(model, input_ids, inputs, max_new_tokens):
    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        output, timings = infer(model, input_ids, inputs, max_new_tokens)

    captured_stdout = stdout_buffer.getvalue()
    print(captured_stdout, end="")
    return output, timings


if __name__ == "__main__":
    max_new_tokens = 1
    model_path = model_id
    device_type = "aiu"
    timing_rows = []

    processor = AutoProcessor.from_pretrained(model_path)
    inputs = _get_inputs(processor)
    inputs["only_last_token"] = True
    inputs["attn_name"] = "sdpa_causal"
    input_ids = inputs.pop("input_ids")

    # Load & compile (ensure compile for sendnn backend)
    linear_config = {"linear_type": "torch_linear"}
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.bfloat16,
        device_type="cpu",
        linear_config=linear_config,
    )

    model.eval()
    torch.set_grad_enabled(False)

    # Warmup / sanity-check generation
    output, timings = infer_with_timing(model, input_ids, inputs, max_new_tokens)

    if len(output.shape) == 1:
        output = output.unsqueeze(0)

    for i in range(output.shape[0]):
        print("Response:", processor.decode(output[i], skip_special_tokens=True))

    # Sweep over prompt lengths
    for prompt_sequence_length in PROMPT_SEQUENCE_LENGTHS:
        print(
            f"\nRunning inference with prompt sequence length: {prompt_sequence_length}"
        )

        sweep_inputs = _get_inputs(processor)
        sweep_inputs = _pad_inputs_to_prompt_length(
            sweep_inputs, prompt_sequence_length
        )
        sweep_inputs["only_last_token"] = True
        sweep_inputs["attn_name"] = "sdpa_causal"
        sweep_input_ids = sweep_inputs.pop("input_ids")

        sweep_output, sweep_timings = infer_with_timing(
            model, sweep_input_ids, sweep_inputs, max_new_tokens
        )

        timing_rows.append(
            (
                prompt_sequence_length,
                sweep_timings,
            )
        )

        if len(sweep_output.shape) == 1:
            sweep_output = sweep_output.unsqueeze(0)

        for i in range(sweep_output.shape[0]):
            print(
                f"Response [{prompt_sequence_length} tokens]:",
                processor.decode(sweep_output[i], skip_special_tokens=True),
            )

    print("\n| Input Token Count | TTFT (ms) | Decode (ms/tok) |")
    print("| ---: | ---: | ---: |")
    for prompt_sequence_length, timings in timing_rows:
        if timings:
            ttft = timings[0] * 1000 if timings else float("nan")
            decode = (
                (sum(timings[1:]) / len(timings[1:])) * 1000
                if len(timings) > 1
                else float("nan")
            )
            print(f"| {prompt_sequence_length} | {ttft:.3f} | {decode:.3f} |")
        else:
            print(f"| {prompt_sequence_length} | N/A | N/A |")
