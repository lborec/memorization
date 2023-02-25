import json

files = ['train_stats/duplicates.json',
         'train_stats/nonduplicates.json']


def print_stats(file):
    print("---------------")
    with open(file) as f:
        data = json.load(f)

    num_copies_counts = {}

    for obj in data:
        num_copies = obj['num_copies']
        if num_copies not in num_copies_counts:
            num_copies_counts[num_copies] = 1
        else:
            num_copies_counts[num_copies] += 1

    for num_copies, count in num_copies_counts.items():
        print(f'num_copies={num_copies}: {count} objects')

    print("---------------")


for file in files:
    print_stats(file)
