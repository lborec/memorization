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
        y = word_probs

        # Compute rolling mean of y-values
        window = 20
        weights = np.repeat(1.0, window) / window
        y_smooth = np.convolve(y, weights, 'valid')

        # Adjust x-values to match smoothed y-values
        x_smooth = x[(window // 2) - 1: -(window // 2) -1]  # Corrected here

        # Plot the smoothed line
        ax.plot(x_smooth, y_smooth[:-1], label=f"Num Copies: {num_copies_list[i]}", color=f"C{i}", linewidth=0.8)

    # Draw parallel line at x=250
    ax.axvline(x=250, color='r', linestyle='--')

    # Configure the plot
    ax.set_xlabel("Token position")
    ax.set_ylabel("Token probability")
    ax.legend()

    # Save the plot to a file
    plt.savefig(output_filename)

    # Close the plot to free up memory
    plt.close(fig)

    return


def check_if_memorized(gold_tokens, output_tokens):
    return torch.equal(gold_tokens, output_tokens)


def get_word_probabilities(model, tokenizer, texts, copies, top_p, input_context_length=250):
    print("top_p", top_p)
    sentence_copies_memorized = {}
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    model.config.pad_token_id = tokenizer.pad_token_id

    all_word_probabilities = []
    decoded_sentences = []
    counter = 0

    for idx, text in enumerate(texts):
        # Check if the sentence with this num of copies has already been memorized
        num_copies = copies[idx]
        if num_copies not in sentence_copies_memorized:
            sentence_copies_memorized[num_copies] = False
        if sentence_copies_memorized[num_copies]:
            continue

        text = "<|endoftext|> " + text
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, padding="max_length")
        input_ids = torch.tensor([tokens[:input_context_length]])

        with torch.no_grad():
            outputs = model.generate(input_ids, do_sample=True, max_length=512, top_p=top_p, top_k=0, return_dict_in_generate=True, output_scores=True)

        memorized = check_if_memorized(torch.tensor(tokens)[:-1], outputs.sequences.squeeze(0)[:-1])

        probs = []

        if memorized:
            counter += 1
            print("Sentence is memorized! Counter: ", counter)
        else:
            all_tokens = outputs['sequences']
            all_token_logits = model(all_tokens)['logits']
            # softmaxed_logits = torch.softmax(all_token_logits, dim=-1).squeeze(0)
            probs = [all_token_logits[0][i][t] for i, t in enumerate(tokens)]
            import pdb;pdb.set_trace()
        # else:
        #     # reset the counter
        #     counter = 0
        #     print(f"Nonmemorized file discovered with {num_copies} num copies.")
        #     sentence_copies_memorized[num_copies] = True
        #
        #     # Get logits of the input sequence
        #     logits = input_outputs.logits[0]
        #     # probabilities = torch.softmax(logits, dim=-1).clamp(min=0, max=1)
        #     probabilities = logits
        #
        #     # input_generated_tokens = outputs.sequences[:, :input_context_length]
        #
        #     # Get logits of the generated sequence
        #     transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        #     input_length = 1 if model.config.is_encoder_decoder else input_context_length
        #     generated_tokens = outputs.sequences[:, input_length:]
        #
        #     # # Get probabilities of the input sequence
        #     for i, token in enumerate(tokens[:input_context_length]):
        #         probs.append(np.exp(probabilities[i,token].item()))
        #     # for tok, score in zip(input_generated_tokens[0], input_probabilities[0]):
        #     #     # | token | token string | logits | probability
        #     #     probs.append(np.exp(score.detach().numpy()))
        #     #     # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.detach().numpy():.3f} | {np.exp(score.detach().numpy()):.2%}")
        #
        #     # Get probabilities of the generated sequence
        #     for tok, score in zip(generated_tokens[0], transition_scores[0]):
        #         # | token | token string | logits | probability
        #         probs.append(np.exp(score.numpy()))
        #         # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
        #
        #     decoded_sentences.append(tokenizer.decode(tokens))
        #     probs = [ 1 if p > 1 else p for p in probs]
            all_word_probabilities.append(probs)

    return all_word_probabilities, decoded_sentences



# Load JSON files and parse them
sampled_duplicates = parse_json_file("memorization/dataset/stats/train_stats/duplicates.json", [5,5,5,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25])#25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25])
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
        # import pdb; pdb.set_trace()
        visualize_word_probabilities(word_probabilities, [1,5,15,25], output_filename)

        # Save word probabilities to a pickle file
        with open(f"{model_name}_word_probabilities_{top_p}.pkl", "wb") as f:
            pickle.dump(word_probabilities, f)

        # Save decoded sentences to a pickle file
        with open(f"{model_name}_decoded_sentences_{top_p}_NON_memorized.pkl", "wb") as f:
            pickle.dump(decoded_sentences, f)
