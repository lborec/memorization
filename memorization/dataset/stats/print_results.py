"""
A pretty terrible script for ad-hoc printing of the results.
"""
import json

# import argparse
import os

# parser = argparse.ArgumentParser(description='Process one argument.')
# parser.add_argument('arg', metavar='ARG', type=str,
#                     help='the argument to process')
#
# args = parser.parse_args()
#
# method = args.arg

all_results = os.listdir("results")

# all_results = [f for f in all_results if method in f]

for json_file in all_results:
    print()
    print(
        "------------------------------------------\n------------------------------------------"
    )
    print("Processing:", json_file)

    with open(f"results/{json_file}", "r") as f:
        json_file = json.load(f)

    total_memorized = 0
    total_length = len(json_file)
    num_copies_dict = {}
    num_copies_total_dict = {}

    for f in json_file:
        num_copies = f["num_copies"]
        if num_copies not in num_copies_total_dict.keys():
            num_copies_total_dict[num_copies] = 1
        else:
            num_copies_total_dict[num_copies] += 1
        if f["memorized"]:
            total_memorized += 1
            if num_copies not in num_copies_dict.keys():
                num_copies_dict[num_copies] = 1
            else:
                num_copies_dict[num_copies] += 1

    for num_copies in num_copies_dict:
        print(f"Num_copies: {num_copies}")
        print(f"Total memorized: {num_copies_dict[num_copies]}")
        print(
            f"Percentage memorized: {num_copies_dict[num_copies] / num_copies_total_dict[num_copies]}"
        )
        print("\n")
    print("total_memorized:", total_memorized)
    print(
        "------------------------------------------\n------------------------------------------"
    )
    print()
