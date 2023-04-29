import torch
from transformers import GPTNeoForCausalLM
from memorization.core.dataset import load_tokenizer
import json
import random

def parse_json_file(filename, num_copies_list):
    # Load the JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    # Filter the data to only include entries with a 'num_copies' field
    filtered_data = [entry for entry in data if "num_copies" in entry]

    # Randomly sample 10 entries with any of the specified 'num_copies' values
    sample = []
    for num_copies in num_copies_list:
        matching_entries = [entry for entry in filtered_data if entry["num_copies"] == num_copies]
        sample += random.sample(matching_entries, 1)

    return sample

def get_word_probabilities(model_name, texts):
    # Load the finetuned GPT-2 model and tokenizer
    print(f"Loading the model... {model_name}")
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = load_tokenizer()
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    model.config.pad_token_id = tokenizer.pad_token_id

    word_probabilities = []
    for text in texts:
        text = '<|startoftext|> ' + text
        # import pdb; pdb.set_trace()
        # Tokenize the input text, add special tokens (start and end)
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
        input_ids = torch.tensor([tokens])

        # Get model's output logits for the input text
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits[0]

        # Calculate probabilities for each word using softmax
        probabilities = torch.softmax(logits, dim=-1)

        # Extract the probabilities of each word at its corresponding timestep
        word_probabilities = []
        for i, token in enumerate(tokens[1:]):  # Skip the first token (start of sentence)
            word_probabilities.append((vocab[token], probabilities[i, token].item()))

    return word_probabilities


# Example usage:
model_name = "trained/gpt-neo-125M/checkpoint-20"
sampled_duplicates = parse_json_file("memorization/dataset/stats/train_stats/duplicates.json", [2,5,10,15,20,25,30])
sampled_nonduplicate = parse_json_file("memorization/dataset/stats/train_stats/nonduplicates.json", [1])

all_files = []

for i, f in enumerate(sampled_duplicates):
    filepath = f['file_path']
    with open(filepath, "r") as file:
        all_files.append((file.read(), filepath['num_copies']))

for i, f in enumerate(sampled_nonduplicate):
    filepath = f['file_path']
    with open(filepath, "r") as file:
        all_files.append((file.read(), filepath['num_copies']))

print(all_files)
# text = []
# word_probabilities = get_word_probabilities(model_name, text)
# print(word_probabilities)
# print("max:", max([w[1] for w in word_probabilities]))
# print(len(word_probabilities))
