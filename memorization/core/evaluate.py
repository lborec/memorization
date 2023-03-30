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
import torch.nn.functional as F

def batched_perplexity(model, dataset, tokenizer, batch_size, stride):
    device = model.device
    max_len = CONTEXT_LENGTH
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    text_len = encodings.input_ids.size(1)
    lls = []

    for i in tqdm(range(0, text_len, batch_size * stride)):
        begin_locs, end_locs, trg_lens = [], [], []
        for j in range(batch_size):
            j = i + j * stride
            if j >= text_len:
                break
            begin_loc = max(j + stride - max_len, 0)
            end_loc = min(j + stride, text_len)
            trg_len = end_loc - j  # may be different from stride on last loop

            begin_locs.append(begin_loc)
            end_locs.append(end_loc)
            trg_lens.append(trg_len)

        input_ids = [encodings.input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]
        target_end_locs = [sen.size(-1) for sen in input_ids]
        input_ids = [
            F.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
        ] # we dont need attention mask as long as these padded token is not involved in loss calculation
        input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

        target_ids = torch.ones_like(input_ids) * -100 # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
        for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[i, -b:e].clone()
            target_ids[i, -b:e] = labels

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs["loss"] * sum(trg_lens)

        lls.append(log_likelihood)

    ppl = torch.exp(sum(torch.stack(lls) / end_locs[-1]))
    return ppl

def calculate_perplexity(
        model_identifier, max_length=CONTEXT_LENGTH, stride=CONTEXT_LENGTH
):
    tokenizer = load_tokenizer()
    data = load_dataset(
        "text", data_dir="memorization/dataset/sampled_dataset/", sample_by="document"
    )
    valid = data["train"]
    length = len(valid)
    keep = int(length * 0.6)
    valid = valid[:keep]


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
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").cuda(device=1)
    elif model_identifier == "plain/gpt-neo-350M":
        model = GPTNeoForCausalLM.from_pretrained("xhyi/PT_GPTNEO350_ATG").cuda(device=1)
    else:
        model = GPTNeoForCausalLM.from_pretrained(f"trained/{model_identifier}").cuda(device=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    ppl = batched_perplexity(model, valid, tokenizer, 1, CONTEXT_LENGTH)
    print("ppl: ", ppl)




    # seq_len = encodings.input_ids.size(1)
    #
    # nlls = []
    # prev_end_loc = 0
    # for begin_loc in tqdm(range(0, seq_len, stride)):
    #     end_loc = min(begin_loc + max_length, seq_len)
    #     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    #     # import pdb;pdb.set_trace()
    #     for enc_ind in range(len(encodings)):
    #         input_ids = encodings.input_ids[enc_ind, begin_loc:end_loc]#.cuda(device=0)
    #         target_ids = input_ids.clone().cuda(device=0)
    #         target_ids[:-trg_len] = -100
    #         # input_ids = input_ids.unsqueeze(0)
    #         # target_ids = target_ids.unsqueeze(0)
    #
    #         with torch.no_grad():
    #             outputs = model(input_ids, labels=target_ids)
    #
    #             # loss is calculated using CrossEntropyLoss which averages over input tokens.
    #             # Multiply it with trg_len to get the summation instead of average.
    #             # We will take average over all the tokens to get the true average
    #             # in the last step of this example.
    #             neg_log_likelihood = outputs.loss * trg_len
    #
    #         nlls.append(neg_log_likelihood)
    #
    #     # prev_end_loc = end_loc
    #     # if end_loc == seq_len:
    #     #     break
    #
    # ppl = torch.exp((torch.stack(nlls).sum() / len(nlls)) / end_loc)
    # print(f"Perplexity score for {model_identifier}: {ppl}")
