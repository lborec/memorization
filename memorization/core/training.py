import os
import math

from memorization.core.dataset import load_tokenizer
from memorization.models import lstm, transformer
from memorization.configs.lstm import LSTM_CONFIG
from torch.utils.data import Dataset, DataLoader
from memorization.configs.gpt2 import *
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig, AutoTokenizer, \
    DataCollatorForLanguageModeling, GPTNeoForCausalLM
from datasets import load_dataset

ALLOWED_MODELS = ["lstm", "transformer"]
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
    assert model_type.lower() in ALLOWED_MODELS, f"Allowed models are: {ALLOWED_MODELS}"

    # gpt2_100_from_pretrained = AutoConfig.from_pretrained(
    #     "gpt2",
    #     vocab_size=len(tokenizer),
    #     n_ctx=512,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     n_embd=512,
    #     n_head=16,
    #     n_layer=24
    # )
    # if model_type == "lstm":
    #     pass
    # elif model_type == 'gpt2-100':
    #     model = GPT2LMHeadModel(gpt2_100_from_pretrained)
    # elif model_type == "gpt2-250":
    #     model = GPT2LMHeadModel(gpt2_250_config)
    # elif model_type == "gpt2-500":
    #     model = GPT2LMHeadModel(gpt2_500_config)
    if model_type == "gpt-neo-125M":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    elif model_type == "gpt-neo-350M":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-350M")

    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params / 1000 ** 2:.1f}M parameters")

    return model

def train_transformer():
    train_dataset = load_dataset("text",
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

    train_dataset_tokenized = train_dataset.map(tokenize, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = load_model("gpt-neo-125M", tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset_tokenized['train'],
        eval_dataset=train_dataset_tokenized['train']
    )
    trainer.train()
    # for current_step, sample in enumerate(dataloader_train):
    #     print("Train epoch %s: Step %s" % (current_epoch, current_step), end="\r")
    #     optimizer.zero_grad()
