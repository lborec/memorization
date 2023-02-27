import os
import math
import json
import torch
from datetime import datetime
from memorization.core.dataset import load_tokenizer
from memorization.core.helpers import progressBar
from transformers import (
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTNeoForCausalLM,
    AutoModelForCausalLM,
)
from datasets import load_dataset

CONTEXT_LENGTH = 512


def load_trained():
    pass


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


def run_experiments(model, json_file, save_path, method):
    # Load model and tokenizer
    tokenizer = load_tokenizer()
    print("...Loading the model...")
    model = GPTNeoForCausalLM.from_pretrained("trained/gpt-neo-125M/checkpoint-180000/").cuda(device=3)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load experiment data
    with open(json_file) as file:
        data = json.load(file)

    results = []
    # import pdb;
    # pdb.set_trace()

    print("..Starting memorization experiments...")
    keys = sorted(data.keys(), reverse=True)
    for key in progressBar(keys, prefix="Progress", suffix="Complete"):
        print("Num counts:", key)
        for data_point in progressBar(data[key], prefix="Progress", suffix="Complete"):
            # Get the variables
            file_path = data_point["file_path"]
            tokens = data_point["tokens"]
            tokens_torch = torch.tensor(tokens).cuda(device=3)
            max_length = data_point["length"]
            num_copies = data_point["num_copies"]

            # Make the result dict
            result_dict = {
                "file_path": file_path,
                "max_length": max_length,
                "memorized": False,
                "num_copies": num_copies,
            }

            # Set loop conditions
            memorized = False
            num_tokens = 0

            # Run memorization loop
            while (memorized == False) and ((num_tokens + 50) < max_length):
                num_tokens += 50
                input_tokens = torch.tensor(tokens[:num_tokens]).unsqueeze(0).cuda(device=3)
                if method == "greedy_decoding":
                    model_output = model.generate(
                        input_tokens, num_beams=1, do_sample=False, max_length=max_length
                    )
                elif method == "nucleus_sampling":
                    pass
                output_tokens = model_output[0]
                memorized = check_if_memorized(tokens_torch, output_tokens)

                if memorized:
                    result_dict["memorized"]: True
                    result_dict["memorized_at"]: num_tokens
                    print("Memorized!")
                else:
                    print("Non memorized yet!")
            results.append(result_dict)

    # Write results to JSON file
    current_timestamp = datetime.now().timestamp()
    current_timestamp = datetime.fromtimestamp(current_timestamp).strftime(
        "%Y-%m-%d-%Hh%Mm%Ss"
    )
    json_save_path = os.path.join(save_path, f"{model}_{current_timestamp}.json")
    with open(json_save_path, "w") as json_file:
        json.dump(results, json_file)
