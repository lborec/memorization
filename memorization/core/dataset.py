import os
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def load_tokenizer():
    # Load the GPT tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    return tokenizer