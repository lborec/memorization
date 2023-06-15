import os
import json
import torch
from memorization.core.dataset import load_tokenizer
from memorization.helper_scripts.helpers import progressBar
from transformers import (
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
)

CONTEXT_LENGTH = 512


def check_if_memorized(gold_tokens, output_tokens):
    return all(gold_tokens == output_tokens)


def tokenize(element, tokenizer):
    text = "<|endoftext|> " + element["text"] + " <|endoftext|>"
    outputs = tokenizer(
        text,
        truncation=True,
        max_length=CONTEXT_LENGTH,
    )
    outputs["input_ids"][-1] = tokenizer.eos_token_id
    return {"input_ids": outputs["input_ids"]}


def run_experiments(model_identifier, json_file, save_path, method, top_p=0.0):
    # Load model and tokenizer
    tokenizer = load_tokenizer()

    print("...Loading the model...")
    local_models = os.listdir("trained/")
    model_exists = False
    for m in local_models:
        if model_identifier in m:
            model_exists = True

    if model_exists:
        model = GPTNeoForCausalLM.from_pretrained(f"trained/{model_identifier}").cuda(
            device=1
        )
    else:
        if "gpt2" in model_identifier:
            model = GPT2LMHeadModel.from_pretrained(model_identifier).cuda(device=1)
        else:
            model = GPTNeoForCausalLM.from_pretrained(model_identifier).cuda(device=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load experiment data
    with open(json_file) as file:
        data = json.load(file)

    print("..Starting memorization experiments...")

    keys = data.keys()
    keys = [int(num) for num in keys]
    keys = sorted(keys, reverse=False)

    for num_tokens in range(500, 49, -50):
        results = []
        print(f"decoding experiment starting. num tokens: {num_tokens}")
        for key in progressBar(keys, prefix="Progress", suffix="Complete"):
            print("\nNum counts:", key)
            str_key = str(key)

            for data_point in data[str_key]:
                # Get the variables
                file_path = data_point["file_path"]
                tokens = data_point["tokens"]
                tokens_torch = torch.tensor(tokens).cuda(device=1)
                max_length = data_point["length"]
                num_copies = data_point["num_copies"]

                if num_copies > 30:
                    continue

                if num_tokens >= max_length:
                    continue

                # Make the result dict
                result_dict = {
                    "file_path": file_path,
                    "max_length": max_length,
                    "memorized": False,
                    "num_copies": num_copies,
                }

                # Run memorization loop
                prefix_length = num_tokens
                input_tokens = (
                    torch.tensor(tokens[:prefix_length]).unsqueeze(0).cuda(device=1)
                )
                if method == "greedy_decoding":
                    model_output = model.generate(
                        input_tokens,
                        num_beams=1,
                        do_sample=False,
                        max_length=max_length,
                    )
                elif method == "nucleus_sampling":
                    model_output = model.generate(
                        input_tokens,
                        do_sample=True,
                        max_length=max_length,
                        top_p=top_p,
                        top_k=0,
                    )

                output_tokens = model_output[0]

                if len(output_tokens) == len(tokens_torch):
                    memorized = check_if_memorized(tokens_torch, output_tokens)
                else:
                    memorized = False

                if memorized:
                    result_dict["memorized"] = True
                results.append(result_dict)

        # Write results to JSON file
        print("saving file...")
        if method == "greedy_decoding":
            json_save_path = os.path.join(
                save_path, f"{model_identifier}_{method}_{num_tokens}.json"
            )
        if method == "nucleus_sampling":
            json_save_path = os.path.join(
                save_path, f"{model_identifier}_{method}_{top_p}_{num_tokens}.json"
            )
        with open(json_save_path, "w") as json_file:
            json.dump(results, json_file)
