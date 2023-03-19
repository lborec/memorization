"""
A pretty terrible script for ad-hoc printing of the results.
"""
import json
import argparse

parser = argparse.ArgumentParser(description='Process one argument.')
parser.add_argument('arg', metavar='ARG', type=str,
                    help='the argument to process')

args = parser.parse_args()

with open(f"results/{args.arg}.json", "r") as f:
    json_file = json.load(f)

total_memorized = 0
total_length = len(json_file)
print(total_length)
total_num_files = 280
num_copies_dict = {}

for f in json_file:
    num_copies = f["num_copies"]
    # import pdb;pdb.set_trace()
    if f["memorized"]:
        total_memorized += 1
        if num_copies not in num_copies_dict.keys():
            num_copies_dict[num_copies] = 1
        else:
            num_copies_dict[num_copies] += 1


for num_copies in num_copies_dict:
    print(f"Num_copies: {num_copies}")
    print(f"Total memorized: {num_copies_dict[num_copies]}")
    print(f"Percentage memorized: {num_copies_dict[num_copies] / 280}")
    print("\n")
print("total_memorized:", total_memorized)
print("---\n---")
