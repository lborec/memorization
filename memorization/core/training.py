import os
import math
from datetime import datetime
from memorization.core.dataset import load_tokenizer
from memorization.models import lstm, transformer
from memorization.configs.lstm import LSTM_CONFIG
from torch.utils.data import Dataset, DataLoader
from memorization.configs.gpt2 import *
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig, AutoTokenizer, \
    DataCollatorForLanguageModeling, GPTNeoForCausalLM
from datasets import load_dataset

ALLOWED_MODELS = ["gpt-neo-125M", "gpt-neo-350M"]
CONTEXT_LENGTH = 512


def load_model(model_type):
    assert model_type in ALLOWED_MODELS, f"Allowed models are: {ALLOWED_MODELS}"

    if model_type == "gpt-neo-125M":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    elif model_type == "gpt-neo-350M":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-350M")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params / 1000 ** 2:.1f}M parameters")

    return model


def train_transformer(model_type):
    dataset = load_dataset("text",
                           data_dir="memorization/dataset/sampled_dataset",
                           sample_by="document")

    tokenizer = load_tokenizer()

    def tokenize(element):
        text = "<|endoftext|> " + element["text"] + " <|endoftext|>"
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=CONTEXT_LENGTH,
        )
        outputs['input_ids'][-1] = tokenizer.eos_token_id
        return {"input_ids": outputs['input_ids']}

    dataset_tokenized = dataset.map(tokenize, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = load_model(model_type)
    model.resize_token_embeddings(len(tokenizer))

    current_timestamp = datetime.now().timestamp()
    current_timestamp = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d-%Hh%Mm%Ss")

    args = TrainingArguments(
        output_dir=f"trained/{model_type}",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=10,
        report_to="wandb",
        run_name=f"{model_type}_{current_timestamp}"
        # fp16=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset_tokenized['train'],
        eval_dataset=dataset_tokenized['validation']
    )

    trainer.train()
