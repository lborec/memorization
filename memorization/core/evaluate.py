import os
import torch
from tqdm import tqdm
from memorization.core.dataset import load_tokenizer
from transformers import GPTNeoForCausalLM
from datasets import load_dataset
from memorization.core.globals import *
import torch.nn.functional as F


def tokenize_inference(tokenizer, text):
    text = "<|endoftext|> " + text + " <|endoftext|>"
    return tokenizer(
        text,
        truncation=True,
        padding='longest',
        max_length=CONTEXT_LENGTH,
        return_tensors='pt',
    )

def batched_perplexity(model, tokenizer, dataset, batch_size, stride):
    device = model.device
    total_log_likelihood = 0.0
    total_tokens = 0

    print("Iterating over dataset...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_texts = dataset["text"][i: i + batch_size]
        encodings = [tokenize_inference(tokenizer, text) for text in batch_texts]

        for encoding in encodings:
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Set padding tokens in labels to -100 so they get ignored in loss calculation
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                log_likelihood = outputs.loss.item() * input_ids.shape[1]
                total_log_likelihood += log_likelihood
                total_tokens += torch.sum(attention_mask).item()  # count tokens

    ppl = torch.exp(torch.tensor(total_log_likelihood / total_tokens).to(device))  # normalize by total number of tokens

    return ppl

def calculate_perplexity():
    tokenizer = load_tokenizer()
    data = load_dataset(
        "text",
        data_dir="memorization/dataset/sampled_dataset/",
        sample_by="document",
        split="validation",
    )

    print("...Loading the model...")
    # for model_identifier in [
    #     "EleutherAI/gpt-neo-125M",
    #     "trained/gpt-neo-125M-2023-03-03-11h00m00s/checkpoint-30000",
    #     "trained/gpt-neo-125M-2023-03-03-11h00m00s",
    #     "xhyi/PT_GPTNEO350_ATG",
    #     "trained/gpt-neo-350M-2023-03-07-19h11m23s/checkpoint-90000",
    #     "trained/gpt-neo-350M-2023-03-07-19h11m23s",
    # ]:
    for model_identifier in os.listdir("trained"):
        print(f"------\n...Calculating perplexity for: {model_identifier}...")
        model = GPTNeoForCausalLM.from_pretrained(f"{model_identifier}").cuda(device=1)
        model.config.pad_token_id = tokenizer.pad_token_id
        ppl = batched_perplexity(model, tokenizer, data, 4, CONTEXT_LENGTH)

        print("ppl: ", ppl)
