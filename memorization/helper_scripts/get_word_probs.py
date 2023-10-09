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
        window = 1
        weights = np.repeat(1.0, window) / window
        y_smooth = np.convolve(y, weights, 'valid')

        # Adjust x-values to match smoothed y-values
        x_smooth = x[window // 2:-(window // 2) or None]

        # Plot the smoothed line
        ax.plot(x_smooth, y_smooth, label=f"Num Copies: {num_copies_list[i]}", color=f"C{i}", linewidth=0.8)

    # Draw parallel line at x=400
    ax.axvline(x=400, color='r', linestyle='--')

    # Configure the plot
    ax.set_xlabel("Token position")
    ax.set_ylabel("Token probability")
    ax.legend()

    # Save the plot to a file
    plt.savefig(output_filename)

    # Close the plot to free up memory
    plt.close(fig)

def check_if_memorized(gold_tokens, output_tokens):
    return torch.equal(gold_tokens, output_tokens)


def get_word_probabilities(model, tokenizer, texts, copies, top_p, input_context_length=400):
    sentence_copies_memorized = {1: False, 15: False, 20: False, 30: False}

    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    model.config.pad_token_id = tokenizer.pad_token_id

    all_word_probabilities = []
    decoded_sentences = []

    for idx, text in enumerate(texts):
        num_copies = copies[idx]
        if sentence_copies_memorized[num_copies]:
            continue

        text = " " + text
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, padding="max_length")
        input_ids = torch.tensor([tokens[:input_context_length]])

        with torch.no_grad():
            output_tokens = model.generate(input_ids, do_sample=True, max_length=512, top_p=top_p, top_k=0, return_dict_in_generate=True, output_scores=True)
        print(output_tokens)
        memorized = check_if_memorized(torch.tensor(tokens)[:-1], output_tokens.squeeze(0)[:-1])

        if not memorized:
            continue
        else:
            transition_scores = model.compute_transition_scores(output_tokens.sequences, output_tokens.scores, normalize_logits=True)
            print(transition_scores)

            print(f"Memorized file discovered with {num_copies} num copies.")
            sentence_copies_memorized[num_copies] = True
            logits = model(output_tokens).logits # (batch_size, sequence_length, config.vocab_size)
            probabilities = torch.softmax(logits, dim=-1) #

            word_probabilities = []
            generated_tokens = output_tokens.squeeze(0).tolist()

            for i, token in enumerate(generated_tokens[:-1]):
                word_probabilities.append((vocab[token], probabilities[0, i, token].item()))

            all_word_probabilities.append(word_probabilities)
            decoded_sentences.append(tokenizer.decode(tokens))

    return all_word_probabilities, decoded_sentences



# Load JSON files and parse them
sampled_duplicates = parse_json_file("memorization/dataset/stats/train_stats/duplicates.json", [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15, 20,20,20,20,20,20,20,20,20,20,20,20,20,20,20, 30,30,30,30,30,30,30,30,30,30,30,30,30,30,30])
# sampled_duplicates = parse_json_file("memorization/dataset/stats/train_stats/duplicates.json", [30,30,30,30,30,30,30,30,30,30,30,30,30,30,30])
sampled_nonduplicate = parse_json_file("memorization/dataset/stats/train_stats/nonduplicates.json", [1])

# define top_p values
top_p_values = [0.2, 0.4, 0.6, 0.8]

# define model names
model_names = ["trained/gpt-neo-125M-2023-03-03-11h00m00s", "trained/gpt-neo-350M-2023-03-07-19h11m23s"]

all_files = []

# Load file content from the parsed JSON files
for f in sampled_nonduplicate + sampled_duplicates:
    filepath = f["file_path"]
    with open(filepath, "r") as file:
        all_files.append(file.read())
num_copies_list = [1] + [entry["num_copies"] for entry in sampled_duplicates]

# Load the tokenizer
tokenizer = load_tokenizer()

for model_name in model_names:
    # Load the model
    print(f"Loading the model... {model_name}")
    model = GPTNeoForCausalLM.from_pretrained(model_name
                                              )
    for top_p in top_p_values:
        print(f"Running top_p={top_p}")
        output_filename = f"{model_name}_sentence_probabilities_{top_p}.png"

        # Get word probabilities and decoded sentences for all files
        word_probabilities, decoded_sentences = get_word_probabilities(model, tokenizer, all_files, num_copies_list, top_p)

        # Visualize word probabilities
        visualize_word_probabilities(word_probabilities, [15,20,30], output_filename)

        # Save word probabilities to a pickle file
        with open(f"{model_name}_word_probabilities_{top_p}.pkl", "wb") as f:
            pickle.dump(word_probabilities, f)

        # Save decoded sentences to a pickle file
        with open(f"{model_name}_decoded_sentences_{top_p}.pkl", "wb") as f:
            pickle.dump(decoded_sentences, f)
