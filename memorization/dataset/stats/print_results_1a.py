import json
import os

all_results = os.listdir("results")
buckets = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
methods = ["greedy_decoding", "nucleus_sampling"]
models = ["125M", "350M"]  # , "gpt2-small", "gpt2-medium"]


def iterate(models, buckets, all_results, nucleus_bucket=""):
    for model in models:
        print("Model:", model)
        model_results = []
        for bucket in buckets:
            print("Bucket:", bucket)
            for result in all_results:
                if (
                    method in result
                    and model in result
                    and f"_{bucket}." in result
                    and nucleus_bucket in result
                ):
                    print("Reading file:", result)
                    with open(os.path.join("results", result), "r") as f:
                        json_file = json.load(f)
                    num_memorized = 0
                    num_total = 0
                    for f in json_file:
                        num_copies = f["num_copies"]
                        if num_copies > 30:
                            continue
                        num_total += 1
                        if f["memorized"]:
                            num_memorized += 1
                    model_results.append(num_memorized / num_total)
        print("Average percentage memorized:", sum(model_results) / len(model_results))
        print()


for method in methods:
    print("Method:", method)
    if method == "greedy_decoding":
        iterate(models, buckets, all_results)
    elif method == "nucleus_sampling":
        nucleus_buckets = [0.2, 0.4, 0.6, 0.8, 1.0]
        for nucleus_b in nucleus_buckets:
            iterate(models, buckets, all_results, str(nucleus_b))
