"""
A pretty terrible script for ad-hoc printing of the results.
"""
import json

with open("results/model_123.json", "r") as f:
    json_file = json.load(f)

total_memorized = 0
total_length = len(json_file)
total_num_files = 280
num_copies_dict = {}

total__context_len = {}
memorized__context_len = {}
total__context_len["up_to_350"] = 0
total__context_len["350_to_450"] = 0
total__context_len["over_450"] = 0
memorized__context_len["up_to_350"] = 0
memorized__context_len["350_to_450"] = 0
memorized__context_len["over_450"] = 0

for f in json_file:
    len = f["max_length"] - 50
    num_copies = f["num_copies"]

    if len < 350:
        flag = "up_to_350"
    elif 350 <= len < 450:
        flag = "350_to_450"
    elif len >= 450:
        flag = "over_450"

    total__context_len[flag] += 1

    if f["memorized"] == "True":
        total_memorized += 1
        if num_copies not in num_copies_dict:
            num_copies_dict[num_copies] = 1
        else:
            num_copies_dict[num_copies] += 1
        memorized__context_len[flag] += 1

for num_copies in num_copies_dict:
    print(f"Num_copies: {num_copies}")
    print(f"Total memorized: {num_copies_dict[num_copies]}")
    print(f"Percentage memorized: {num_copies_dict[num_copies] / 280}")
    print("\n")

print("---\n---")

for range in ["up_to_350", "350_to_450", "over_450"]:
    print(f"Range: {range}")
    print(f"Total items: {total__context_len[range]}")
    print(f"Total items memorized: {memorized__context_len[range]}")
    print(
        f"Percentage memorized: {memorized__context_len[range] / total__context_len[range]}"
    )
    print("\n")
