import torch
import json
import random
from memorization.core.dataset import load_tokenizer
from transformers import GPTNeoForCausalLM

CONTEXT_LENGTH = 512


def check_if_memorized(gold_tokens, output_tokens):
    return (gold_tokens == output_tokens).all().item()




def parse_json_file(filename, num_copies_list):
    # Load the JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    print(f"Total entries in the data: {len(data)}")  # Debug print

    # Filter the data to only include entries with a 'num_copies' field
    filtered_data = [entry for entry in data if "num_copies" in entry]

    print(f"Total entries after filtering for num_copies: {len(filtered_data)}")  # Debug print

    # Randomly sample entries with any of the specified 'num_copies' values
    sample = []
    for num_copies in num_copies_list:
        matching_entries = [entry for entry in filtered_data if entry["num_copies"] == num_copies]

        print(f"Total entries for num_copies={num_copies}: {len(matching_entries)}")  # Debug print

        if matching_entries:
            while True:
                selected_entry = random.choice(matching_entries)
                f = open(selected_entry["file_path"], "r").read()
                if len(f.split()) > 512:
                    sample.append(selected_entry)
                    break

    return sample


def run_memorization_test(model_name, tokenizer, model, data_points, input_context_length, top_p):
    results = []

    for data_point in data_points:
        text = open(data_point["file_path"], "r").read()
        sentence = "<|endoftext|> " + text
        tokens = tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=512,
                                  padding="max_length")
        input_tokens = torch.tensor([tokens[:input_context_length]]).cuda(device=0)
        gold_tokens = torch.tensor([tokens[:CONTEXT_LENGTH]]).cuda(device=0)

        with torch.no_grad():
            output_tokens = model.generate(input_tokens, do_sample=True, max_length=CONTEXT_LENGTH, top_p=top_p, top_k=0)
        output_sentence = tokenizer.decode(output_tokens[0], skip_special_tokens=False)

        # import pdb; pdb.set_trace()
        # if len(output_tokens) == len(gold_tokens):

        memorized = check_if_memorized(gold_tokens[0:511], output_tokens[0,:511])
        # else:
        #     memorized = False

        result = {
            "num_copies": data_point["num_copies"],
            "memorized": memorized,
            "original": sentence,
            "decoded": output_sentence,
        }

        results.append(result)

    return results


def main():
    num_copies_list = [10,10,10,10,10, 20,20,20,20,20, 30,30,30,30,30,30,30,30,30,30,30,30,30]
    context_length = 250
    top_p = 0.2

    tokenizer = load_tokenizer()

    model_names = ["trained/gpt-neo-125M-2023-03-03-11h00m00s"]#, "trained/gpt-neo-350M-2023-03-07-19h11m23s"]

    for model_name in model_names:
        model = GPTNeoForCausalLM.from_pretrained(model_name).cuda(device=0)
        model.config.pad_token_id = tokenizer.pad_token_id

        all_results = []
        sampled_duplicates = parse_json_file("memorization/dataset/stats/train_stats/duplicates.json", num_copies_list)

        results = run_memorization_test(model_name, tokenizer, model, sampled_duplicates, context_length, top_p)

        all_results.extend(results)

        # Save results to a JSON file
        with open(f"{model_name}_memorization_coherence_results.json", "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    main()
