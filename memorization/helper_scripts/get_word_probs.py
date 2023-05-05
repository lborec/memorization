import torch
from transformers import GPTNeoForCausalLM
from memorization.core.dataset import load_tokenizer
import json
import random
import argparse
import matplotlib.pyplot as plt

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

def visualize_word_probabilities(word_probabilities, num_copies_list, output_filename):
    # Set up the plot
    fig, ax = plt.subplots()
    colormap = plt.cm.get_cmap("tab10", len(num_copies_list))

    # Plot the data
    for i, (word_probs, num_copies) in enumerate(zip(word_probabilities, num_copies_list)):
        x = list(range(1, len(word_probs) + 1))
        y = [prob for _, prob in word_probs]
        ax.plot(x, y, label=f"{num_copies} copies", color=colormap(i))

    # Configure the plot
    ax.set_xlabel("Word position")
    ax.set_ylabel("Probability")
    ax.set_title("Word probabilities by num_copies")
    ax.legend()

    # Save the plot to a file
    plt.savefig(output_filename)

    # Close the plot to free up memory
    plt.close(fig)

def get_word_probabilities(model, tokenizer, texts):
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    model.config.pad_token_id = tokenizer.pad_token_id

    word_probabilities = []
    for text in texts:
        text = ' ' + text[1]
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
        input_ids = torch.tensor([tokens])

        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits[0]

        probabilities = torch.softmax(logits, dim=-1)

        word_probabilities = []
        for i, token in enumerate(tokens[1:]):
            word_probabilities.append((vocab[token], probabilities[i, token].item()))

    return word_probabilities

# Example usage:
sampled_duplicates = parse_json_file("memorization/dataset/stats/train_stats/duplicates.json", [2, 5, 10, 15, 20, 25, 30])
sampled_nonduplicate = parse_json_file("memorization/dataset/stats/train_stats/nonduplicates.json", [1])

all_files = []
num_copies_list = []

for i, f in enumerate(sampled_duplicates):
    filepath = f['file_path']
    with open(filepath, "r") as file:
        all_files.append(file.read())
        num_copies_list.append(f['num_copies'])

for i, f in enumerate(sampled_nonduplicate):
    filepath = f['file_path']
    with open(filepath, "r") as file:
        all_files.append(file.read())
        num_copies_list.append(f['num_copies'])


# Set up the argument parser
parser = argparse.ArgumentParser(description="Visualize word probabilities for a given GPT-Neo model.")
parser.add_argument("model_name", type=str, help="The model name, e.g., 'trained/gpt-neo-125M/checkpoint-20'")
args = parser.parse_args()

model_name = args.model_name
output_filename = f"{model_name}_sentence_probabilities.png"

# Load the model and tokenizer
print(f"Loading the model... {model_name}")
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = load_tokenizer()

word_probabilities = [get_word_probabilities(model, tokenizer, [text]) for text in all_files]
visualize_word_probabilities(word_probabilities, num_copies_list, output_filename)
