import json
import os

all_results = os.listdir("results")
buckets = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
methods = ["greedy_decoding", "nucleus_sampling"]
models = ["125M", "350M", "gpt2-small", "gpt2-medium"]

for method in methods:
    print("Method:", method)
    for model in models:
        print("Model:", model)
        num_copies_dict = {}
        num_copies_total_dict = {}
        for bucket in buckets:
            for result in all_results:
                if method in result and model in result and f"_{bucket}." in result:
                    print("Reading file:", result)
                    with open(os.path.join("results", result), "r") as f:
                        json_file = json.load(f)
                    for f in json_file:
                        num_copies = f["num_copies"]
                        if num_copies > 30:
                            continue
                        if num_copies not in num_copies_total_dict.keys():
                            num_copies_total_dict[num_copies] = 1
                        else:
                            num_copies_total_dict[num_copies] += 1
                        if f["memorized"]:
                            if num_copies not in num_copies_dict.keys():
                                num_copies_dict[num_copies] = 1
                            else:
                                num_copies_dict[num_copies] += 1

        for num_copies in sorted(num_copies_dict.keys()):
            if num_copies < 31:
                print(f"Num_copies: {num_copies}")
                print(f"Total memorized: {num_copies_dict[num_copies]}")
                print(
                    f"Percentage memorized: {num_copies_dict[num_copies] / num_copies_total_dict[num_copies]}"
                )
                print("\n")
        print(
            "------------------------------------------\n------------------------------------------"
        )
