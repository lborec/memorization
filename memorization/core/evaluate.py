import torch
from tqdm import tqdm
from memorization.core.dataset import load_tokenizer
from transformers import GPTNeoForCausalLM
from datasets import load_dataset
from memorization.core.globals import *
import torch.nn.functional as F


def batched_perplexity(model, dataset, tokenizer, batch_size, stride):
    device = model.device
    max_len = CONTEXT_LENGTH
    encodings = tokenizer(
        "<|endoftext|> ".join(dataset["text"]),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=CONTEXT_LENGTH,
    )
    import pdb;pdb.set_trace()
    text_len = encodings.input_ids.size(1)
    lls = []
    print("Iterating over dataset...")
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
        ]  # we dont need attention mask as long as these padded token is not involved in loss calculation
        input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

        target_ids = (
            torch.ones_like(input_ids) * -100
        )  # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
        for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[i, -b:e].clone()
            target_ids[i, -b:e] = labels

        with torch.no_grad():
            try:
                outputs = model(input_ids, labels=target_ids)
            except RuntimeError as e:
                print(e)
                print("input_ids: ", input_ids)
                print("target_ids: ", target_ids)
                print("begin_locs: ", begin_locs)
                print("end_locs: ", end_locs)
                print("trg_lens: ", trg_lens)
                raise e
            log_likelihood = outputs["loss"] * sum(trg_lens)

        lls.append(log_likelihood)

    ppl = torch.exp(sum(torch.stack(lls) / end_locs[-1]))
    return ppl


def calculate_perplexity():
    tokenizer = load_tokenizer()
    data = load_dataset(
        "text",
        data_dir="memorization/dataset/sampled_dataset/",
        sample_by="document",
        split="validation[:0.2%]",  # train[:5%]
    )

    print("...Loading the model...")
    for model_identifier in [
        "EleutherAI/gpt-neo-125M",
        "trained/gpt-neo-125M-2023-03-03-11h00m00s/checkpoint-30000",
        "trained/gpt-neo-125M-2023-03-03-11h00m00s",
        "xhyi/PT_GPTNEO350_ATG",
        "trained/gpt-neo-350M-2023-03-07-19h11m23s/checkpoint-90000",
        "trained/gpt-neo-350M-2023-03-07-19h11m23s",
    ]:
        print(f"------\n...Calculating perplexity for: {model_identifier}...")
        model = GPTNeoForCausalLM.from_pretrained(f"{model_identifier}").cuda(device=1)
        model.config.pad_token_id = tokenizer.pad_token_id
        ppl = batched_perplexity(model, data, tokenizer, 1, CONTEXT_LENGTH)
        print("ppl: ", ppl)
