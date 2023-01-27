import os
import math
from memorization.models import lstm, transformer
from memorization.configs.lstm import LSTM_CONFIG
from torch.utils.data import Dataset, DataLoader
from memorization.configs.gpt2 import *
from transformers import Trainer, TrainingArguments, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig, \
    AutoTokenizer, \
    DataCollatorForLanguageModeling, GPT2Config
from datasets import load_dataset

CONTEXT_LENGTH = 512

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    # fp16=True
)


def load_model(model_type, tokenizer):
    # gpt2 (small), gpt2-medium, gpt2-large, gpt2-xl
    if model_type == "gpt2-small":
        model_type = "gpt2"

        # Get the config
        config = GPT2Config.from_pretrained(model_type, output_hidden_states=False)

        # Instantiate the model
        model = GPT2LMHeadModel.from_pretrained(model_type, config=config)

        # This step is necessary because I've added some tokens (bos_token, etc) to the embeddings
        # otherwise the tokenizer and model tensors won't match up
        model.resize_token_embeddings(len(tokenizer))