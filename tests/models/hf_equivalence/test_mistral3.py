import os
import pytest
from datetime import datetime, timedelta

import torch
from difflib import SequenceMatcher
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText

from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids

device = "cpu"
torch.set_default_dtype(torch.float32)


def load_system_prompt(repo_id: str, filename: str) -> str:
    """Load system prompt from model directory

    Ref: https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506#transformers
    """
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)



def _get_inputs(processor, model_path, image_path="Battle.jpeg"):

    # Load system prompt else, error out to make sure we test with right system prompt
    system_prompt = load_system_prompt(model_path, "SYSTEM_PROMPT.txt")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
                },
                {"type": "image"},
            ],
        },
    ]

    # Load image
    images = [Image.open(image_path)]

    # Apply chat template and process inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=images, return_tensors="pt").to(device)
    return inputs


def _get_hf_model_output(model_path, inputs, max_new_tokens=100):

    model = AutoModelForImageTextToText.from_pretrained(model_path).to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False
        )
    return output


def _get_fms_model_output(model_path, inputs, max_new_tokens=100):
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )
    model.eval()
    torch.set_grad_enabled(False)

    inputs["only_last_token"] = True
    inputs["attn_name"] = "sdpa_causal"
    input_ids = inputs.pop("input_ids")
    input_ids, padding_kwargs = pad_input_ids(input_ids, min_pad_length=0)
    inputs["mask"] = padding_kwargs["mask"].to(device)
    inputs["position_ids"] = padding_kwargs["position_ids"].to(device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            max_seq_len=model.config.text_config.max_expected_seq_len,
            extra_kwargs=inputs,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
        )

    return output


@pytest.mark.slow
def test_mistral3_24b_equivalence():
    # for now, this test won't be run, but it has been verified
    # if you would like to try this, set granite_model_path to the huggingface granite model path

    # model_path = "/path/to/Mistral-Small-3.1-24B-Instruct-2503"
    model_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    # NOTE: Mistral 3.2 doesn't have the HF processor config in the checkpoint,
    # so we use the processor from Mistral 3.1 which is compatible
    processor = AutoProcessor.from_pretrained(
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )

    # Get inputs with the model path for system prompt loading
    inputs = _get_inputs(processor, model_path)

    hf_model_output = _get_hf_model_output(model_path, inputs)
    fms_model_output = _get_fms_model_output(model_path, inputs)


    print(processor.decode(fms_model_output, skip_special_tokens=True))


if __name__ == "__main__":
    test_mistral3_24b_equivalence()
