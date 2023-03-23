import torch
from tqdm import tqdm
from memorization.core.dataset import load_tokenizer
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
from memorization.core.globals import *


def calculate_perplexity(
        model_identifier, max_length=CONTEXT_LENGTH, stride=CONTEXT_LENGTH
):
    tokenizer = load_tokenizer()
    data = load_dataset(
        "text", data_dir="memorization/dataset/sampled_dataset/", sample_by="document"
    )
    valid = data["validation"]


    encodings = tokenizer(
        valid["text"],
        truncation=True,
        padding=True,
        max_length=CONTEXT_LENGTH,
        return_tensors="pt",
    )
    # def tokenize(element):
    #     text = "<|endoftext|> " + element["text"] + " <|endoftext|>"
    #     outputs = tokenizer(
    #         text,
    #         truncation=True,
    #         max_length=512,
    #         return_tensors="pt"
    #     )
    #     outputs["input_ids"][-1] = tokenizer.eos_token_id
    #     return {"input_ids": outputs["input_ids"]}
    # print("Tokenizing dataset...")
    # encodings = valid.map(tokenize, batched=False)

    print("...Loading the model...")

    if model_identifier == "plain/gpt-neo-125M":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/plain/gpt-neo-125M").cuda(device=0)
    elif model_identifier == "plain/gpt-neo-350M":
        model = GPTNeoForCausalLM.from_pretrained("xhyi/PT_GPTNEO350_ATG").cuda(device=0)
    else:
        model = GPTNeoForCausalLM.from_pretrained(f"trained/{model_identifier}").cuda(device=0)
    model.config.pad_token_id = tokenizer.pad_token_id


    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        # import pdb;pdb.set_trace()
        for enc_ind in range(len(encodings)):
            input_ids = encodings.input_ids[enc_ind, begin_loc:end_loc].cuda(device=0)
            target_ids = input_ids.clone().cuda(device=0)
            target_ids[:-trg_len] = -100
            # input_ids = input_ids.unsqueeze(0)
            # target_ids = target_ids.unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

        # prev_end_loc = end_loc
        # if end_loc == seq_len:
        #     break

    ppl = torch.exp((torch.stack(nlls).sum() / len(nlls)) / end_loc)
    print(f"Perplexity score for {model_identifier}: {ppl}")
