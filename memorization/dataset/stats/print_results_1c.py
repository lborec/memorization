import json
import os

all_results = os.listdir("results")
buckets = [[50, 100], [150, 200], [250, 300], [350, 400], [450, 500]]
methods = ["greedy_decoding", "nucleus_sampling"]
models = ["125M", "350M", "gpt2-small", "gpt2-medium"]
top_p_values = ["0.2", "0.4", "0.6", "0.8"]

for method in methods:
    print("Method:", method)
    if method == "nucleus_sampling":
        for model in models:
            print("Model:", model)
            for top_p in top_p_values:
                print("Top_p:", top_p)
                for bucket in buckets:
                    print("Bucket:", bucket)
                    bucket_results = []
                    for result in all_results:
                        if (
                            method in result
                            and model in result
                            and (
                                f"_{bucket[0]}." in result or f"_{bucket[1]}." in result
                            )
                            and top_p in result
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
                            bucket_results.append(num_memorized / num_total)
                    print(
                        "Average percentage memorized:",
                        sum(bucket_results) / len(bucket_results),
                    )
                    print()
    else:
        for model in models:
            print("Model:", model)
            for bucket in buckets:
                print("Bucket:", bucket)
                bucket_results = []
                for result in all_results:
                    if (
                        method in result
                        and model in result
                        and (f"_{bucket[0]}." in result or f"_{bucket[1]}." in result)
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
                    else:
                        continue
                print("Percentage memorized:", num_memorized / num_total)
                print()
