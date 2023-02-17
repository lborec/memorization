import os
import math
from datetime import datetime
from memorization.core.dataset import load_tokenizer
from memorization.models import lstm, transformer
from memorization.configs.lstm import LSTM_CONFIG
from torch.utils.data import Dataset, DataLoader
from memorization.configs.gpt2 import *
from transformers import (
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTNeoForCausalLM,
)
from datasets import load_dataset

ALLOWED_MODELS = ["gpt-neo-125M", "gpt-neo-350M"]
CONTEXT_LENGTH = 512


def load_tokenizer():
    # Load the GPT tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    return tokenizer


def load_model(model_type):
    assert model_type in ALLOWED_MODELS, f"Allowed models are: {ALLOWED_MODELS}"

    model = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/{model_type}")  # .cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params / 1000 ** 2:.1f}M parameters")

    return model


def train_transformer(model_type):
    print("Loading dataset...")
    dataset = load_dataset(
        "text", data_dir="memorization/dataset/sampled_dataset/", sample_by="document"
    )

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    def tokenize(element):
        text = "<|endoftext|> " + element["text"] + " <|endoftext|>"
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=CONTEXT_LENGTH,
        )
        outputs["input_ids"][-1] = tokenizer.eos_token_id
        return {"input_ids": outputs["input_ids"]}

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("Loading model...")
    model = load_model(model_type)
    model.resize_token_embeddings(len(tokenizer))

    current_timestamp = datetime.now().timestamp()
    current_timestamp = datetime.fromtimestamp(current_timestamp).strftime(
        "%Y-%m-%d-%Hh%Mm%Ss"
    )

    modeldir = f"trained/{model_type}"

    args = TrainingArguments(
        output_dir=modeldir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=6000,
        logging_steps=6000,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=6000,
        report_to="wandb",
        run_name=f"{model_type}_{current_timestamp}",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
    )

    print("Beginning training...")
    trainer.train()
    trainer.save_pretrained(modeldir)
