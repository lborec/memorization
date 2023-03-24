import json
import os

all_results = os.listdir("results")
buckets = [[50, 100], [150, 200], [250, 300], [350, 400], [450, 500]]
methods = ["greedy_decoding", "nucleus_sampling"]
models = ["125M", "350M"]

for method in methods:
    print("Method:", method)
    for model in models:
        print("Model:", model)
        for bucket in buckets:
            print("Bucket:", bucket)
            bucket_results = []
            for result in all_results:
                if method in result and model in result and (f"_{bucket[0]}." in result or f"_{bucket[1]}." in result ):
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
                else:
                    continue
            print("Percentage memorized:", num_memorized / num_total)
            print()





