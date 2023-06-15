import torch
from transformers import GPTNeoForCausalLM
from memorization.core.dataset import load_tokenizer
import json
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.interpolate import interp1d

def parse_json_file(filename, num_copies_list):
    # Load the JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    # Filter the data to only include entries with a 'num_copies' field
    filtered_data = [entry for entry in data if "num_copies" in entry]

    # Randomly sample entries with any of the specified 'num_copies' values
    sample = []
    for num_copies in num_copies_list:
        matching_entries = [entry for entry in filtered_data if entry["num_copies"] == num_copies]

        while True:
            selected_entry = random.choice(matching_entries)
            f = open(selected_entry["file_path"], "r").read()
            if len(f.split()) > 512:
                sample.append(selected_entry)
                break

    return sample


def visualize_word_probabilities(word_probabilities, num_copies_list, output_filename):
    # Set up the plot
    fig, ax = plt.subplots()

    # Plot the data for non-empty lists
    for i, word_probs in enumerate(word_probabilities):
        if not word_probs:  # Skip empty lists
            continue
        x = list(range(1, len(word_probs) + 1))
        y = [p for _, p in word_probs]

        # Compute rolling mean of y-values
        window = 20
        weights = np.repeat(1.0, window) / window
        y_smooth = np.convolve(y, weights, 'valid')

        # Adjust x-values to match smoothed y-values
        x_smooth = x[window // 2:-(window // 2) or None]

        # Plot the smoothed line
        ax.plot(x_smooth, y_smooth[1:], label=f"Num Copies: {num_copies_list[i]}", color=f"C{i}", linewidth=0.8)

    # Configure the plot
    ax.set_xlabel("Token position")
    ax.set_ylabel("Token probability")
    ax.legend()

    # Save the plot to a file
    plt.savefig(output_filename)

    # Close the plot to free up memory
    plt.close(fig)



def get_word_probabilities(model, tokenizer, texts):
    """
    Calculate word probabilities for each text.

    Args:
        model (GPTNeoForCausalLM): The pre-trained GPT-Neo model.
        tokenizer (PreTrainedTokenizer): The tokenizer for the GPT-Neo model.
        texts (list): A list of texts.

    Returns:
        list: A list of word probability lists.
    """
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    model.config.pad_token_id = tokenizer.pad_token_id

    all_word_probabilities = []
    for text in texts:
        text = "<|endoftext|> " + text
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, padding="max_length")

        input_ids = torch.tensor([tokens])

        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits[0]

        probabilities = torch.softmax(logits, dim=-1).clamp(min=0, max=1)  # clamp probabilities

        word_probabilities = []
        for i, token in enumerate(tokens[1:]):
            word_probabilities.append((vocab[token], probabilities[i, token].item()))

        all_word_probabilities.append(word_probabilities)

    return all_word_probabilities


# Load JSON files and parse them
sampled_duplicates = parse_json_file("memorization/dataset/stats/train_stats/duplicates.json", [10, 20, 30])
sampled_nonduplicate = parse_json_file("memorization/dataset/stats/train_stats/nonduplicates.json", [1])

all_files = []
# Load file content from the parsed JSON files
for f in sampled_nonduplicate + sampled_duplicates:
    filepath = f["file_path"]
    with open(filepath, "r") as file:
        all_files.append(file.read())
num_copies_list = [1] + [entry["num_copies"] for entry in sampled_duplicates]

model_names = ["trained/gpt-neo-125M-2023-03-03-11h00m00s", "trained/gpt-neo-350M-2023-03-07-19h11m23s"]

# Load the model and tokenizer
for model_name in model_names:
    output_filename = f"{model_name}_sentence_probabilities.png"
    print(f"Loading the model... {model_name}")
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = load_tokenizer()

    # Get word probabilities for all files
    word_probabilities = get_word_probabilities(model, tokenizer, all_files)

    # Visualize word probabilities
    visualize_word_probabilities(word_probabilities, num_copies_list, output_filename)

    # Save word probabilities to a pickle file
    with open(f"{model_name}_word_probabilities.pkl", "wb") as f:
        pickle.dump(word_probabilities, f)
