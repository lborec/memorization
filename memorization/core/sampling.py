import os
import re
import shutil
import tarfile
import json
import math
import random
import numpy as np
from collections import Counter
from memorization.core.helpers import progressBar
from memorization.core.dataset import load_tokenizer

SAMPLING_TOKEN_LENGTHS = [150, 200, 250, 300, 350, 400, 450, 500]
SAMPLING_SUBFOLDER_PREFIX = "length_"


def create_target_paths(project_path):
    """
    This function does bookkeeping related to the target folder where the sampled openwebtext will be stored.
    """
    # Create {project_path}/memorization/dataset/sampled_dataset
    sampled_dataset_path = os.path.join(
        project_path, "memorization/dataset/sampled_dataset"
    )

    # Create folder if it doesn't exist
    if not os.path.exists(sampled_dataset_path):
        print(f"Sampled openwebtext will be stored at {sampled_dataset_path}")

    train_path = os.path.join(sampled_dataset_path, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    valid_path = os.path.join(sampled_dataset_path, "valid")
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    # Create {project_path}/memorization/stats
    stats_path = os.path.join(project_path, "memorization/dataset/stats")

    # Create folder if it doesn't exist
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
        print(
            f"JSON files containing duplicate statistics will be stored at {stats_path}"
        )

    # # Create subfolders where different length subsets will be stored, e.g. len = 50, 100, 150...
    # for length in SAMPLING_TOKEN_LENGTHS:
    #     temp_json_path = os.path.join(duplicates_path, f"length_{length}")
    #     open(temp_json_path, "w").close()

    return [train_path, valid_path], stats_path


def unpack_dataset(unpacked_dataset_path):
    """
    Unpacks the .xz files in openwebtext. The files are actually folders containing multiple .txt files.
    The files are unpacked inside the directory.
    """
    for filename in progressBar(
            os.listdir(unpacked_dataset_path), prefix="Progress", suffix="Complete"
    ):
        # Iterate through .xz files
        if filename.endswith(".xz"):
            filepath = os.path.join(unpacked_dataset_path, filename)
            with tarfile.open(fileobj=filepath, mode="r:xz") as tar:
                # Create a new folder
                new_folder_path = os.path.join(unpacked_dataset_path, filename[:-3])
                if not os.path.exists(new_folder_path):
                    os.mkdir(new_folder_path)
                # Extract into  folder
                tar.extractall(new_folder_path)
                # # Remove .xz file
                # os.remove(filepath)


def sample_dataset(
        dataset_path, sampled_dataset_path, sample_ratio=0.25, split="train"
):
    """
    This functions samples the original openwebtext and stores it to the target folder.
    """
    print(f"Beginning sampling for the '{split}' split...")
    # Get a list of all folders from the original dataset and prune out non-folder items
    all_folders_in_dataset = os.listdir(dataset_path)
    all_folders_in_dataset = [
        folder
        for folder in all_folders_in_dataset
        if os.path.isdir(os.path.join(dataset_path, folder))
    ]
    len_all_folders_in_dataset = len(all_folders_in_dataset)
    # If sampling for the validation split, remove folders that are in the train split
    if split == "train":  # later: != "train"
        print(f"Pruning folders... currently {len_all_folders_in_dataset} folders.")
        with open("memorization/dataset/stats/train_folders.txt", "r") as f:
            train_folders = [line.strip() for line in f.readlines()]
        all_folders_in_dataset = [
            folder for folder in all_folders_in_dataset if folder not in train_folders
        ]
        print(f"Folders pruned! Currently {len(all_folders_in_dataset)} folders.")

    # Get the indices of the folders that we will sample for the new dataset
    num_files_to_keep = math.floor(len_all_folders_in_dataset * sample_ratio)
    indices_to_keep = random.sample(
        range(len_all_folders_in_dataset), num_files_to_keep
    )
    folders_to_keep = [all_folders_in_dataset[ind] for ind in indices_to_keep]

    # Move the chosen files from dataset_path to sampled_dataset_path
    for folder in progressBar(folders_to_keep, prefix="Progress", suffix="Complete"):
        with open(f"memorization/dataset/stats/{split}_folders.txt", "a") as f:
            f.write(f"{folder}\n")
        source_filepath = os.path.join(dataset_path, folder)
        destination_filepath = os.path.join(sampled_dataset_path, folder)
        shutil.copytree(source_filepath, destination_filepath)


def generate_duplicates(sampled_dataset_path):
    """
    Generates duplicates randomly according to exponential distribution.
    e.g. for a dataset of 3 samples (["text_1", "text_2", "text_3"]) and scale=0.5,
         the distribution may end up being [1,2,1].
         The duplicated openwebtext would then be:
         (["text_1", "text_2", "text_2". "text_3"])
    """
    all_folders = os.listdir(sampled_dataset_path)
    all_folders = [
        folder
        for folder in all_folders
        if os.path.isdir(os.path.join(sampled_dataset_path, folder))
    ]
    for folder in progressBar(all_folders, prefix="Progress", suffix="Complete"):
        folder_path = os.path.join(sampled_dataset_path, folder)
        all_files = os.listdir(folder_path)
        length = len(all_files)
        sample_indices = np.random.exponential(size=length)
        num_duplicates = np.ceil(sample_indices).astype(int)
        for index, num in enumerate(num_duplicates):
            if num > 1:
                for n in range(1, num):
                    # Read in the file to be duplicated
                    full_file_name = all_files[index]
                    file_path = os.path.join(folder_path, full_file_name)
                    txt = open(file_path, "r").read()

                    # Create a new file and append it with the number of the copy
                    # E.g. if abc.txt has three duplicates, it'll be abc.txt, abc_2.txt, abc_3.txt
                    file_name_without_extension = full_file_name.split(".txt")[0]
                    new_file_name = f"{file_name_without_extension}_{n + 1}.txt"
                    new_file_path = os.path.join(folder_path, new_file_name)
                    with open(new_file_path, "w") as f:
                        f.write(txt)


# def generate_duplicates_controlled(sampled_dataset_path, copy_up_to=40, num_objects_copied=250):
#     # Put all files into a single list
#     all_folders = os.listdir(sampled_dataset_path)
#     all_folders = [
#         folder
#         for folder in all_folders
#         if os.path.isdir(os.path.join(sampled_dataset_path, folder))
#     ]
#     all_files = []
#     for folder in all_folders:
#         folder_path = os.path.join(sampled_dataset_path, folder)
#         files = os.listdir(folder_path)
#         for f in files:
#             filepath = os.path.join(folder_path, f)
#             all_files.append(filepath)
#
#     for i in range(2, copy_up_to): # 2 to 35
#         for j in range(num_objects_copied): #
#             import pdb; pdb.set_trace()
#             random_filepath = all_files.pop(random.randrange(len(all_files)))
#             txt = open(random_filepath, "r").read()
#             for n in range(1, i):
#                 # ... copy from above


def generate_stats(split_path, stats_folder_path):
    """
    Goes through the entire sampled_dataset, and collects statistics for each data sample (txt file).
    data = {
                "file_path": filepath,
                "num_copies": num_copies,
                "length": length,
                "tokens": tokenized_txt["input_ids"],
            }
    This file is later used to sample data from for the experiments.
    """
    # Make folders
    if not os.path.exists(stats_folder_path):
        os.makedirs(stats_folder_path)

    # Create buckets
    duplicates, nonduplicates = [], []

    # Define the regex pattern
    regex = r"(?:_[0-9]+)*\.txt"

    # Get all folders
    all_folders = os.listdir(split_path)
    all_folders = [
        folder
        for folder in all_folders
        if os.path.isdir(os.path.join(split_path, folder))
    ]
    # Iterate over each folder's txt files and write stats
    for folder in progressBar(all_folders, prefix="Progress", suffix="Complete"):
        folder_path = os.path.join(split_path, folder)
        files = os.listdir(folder_path)
        files = [file for file in files if file.endswith(".txt")]
        temp_files = []
        for file in files:
            extensionless_filename = re.split(regex, file)
            temp_files.append(extensionless_filename[0])
        # Get the duplicate counts
        counts = Counter(temp_files)
        ### Write stats to file
        for filename, num_copies in counts.items():
            filename_txt = filename + ".txt"
            filepath = os.path.join(folder_path, filename_txt)

            # Define the data to be included in the JSON file
            data = {
                "file_path": filepath,
                "num_copies": num_copies
            }
            if num_copies > 1:
                duplicates.append(data)
            else:
                nonduplicates.append(data)
    # Write JSONs
    duplicates_json = f"{stats_folder_path}/duplicates.json"
    nonduplicates_json = f"{stats_folder_path}/nonduplicates.json"

    with open(duplicates_json, "w") as json_file:
        json.dump(duplicates, json_file)
    with open(nonduplicates_json, "w") as json_file:
        json.dump(nonduplicates, json_file)

    return duplicates_json, nonduplicates_json


def generate_stats_masterlist(files, save_path, num_files_to_keep=250):
    """

    """
    # Load the tokenizer
    tokenizer = load_tokenizer()

    def tokenize(element):
        text = "<|endoftext|> " + element["text"] + " <|endoftext|>"
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=512,
        )
        outputs["input_ids"][-1] = tokenizer.eos_token_id
        return {"input_ids": outputs["input_ids"]}

    pruned_buckets = {}

    # Iterate over duplicates and nonduplicates
    for file in progressBar(files, prefix="Progress", suffix="Complete"):
        with open(file) as f:
            data = json.load(f)

        buckets = {}

        for obj in progressBar(data, prefix="Progress", suffix="Complete"):
            num_copies = obj["num_copies"]
            if num_copies != 22:
                continue

            # Get txt file stats
            txt = open(obj["file_path"], "r").read()
            tokenized_txt = tokenize({"text": txt})
            length = len(tokenized_txt["input_ids"])

            obj["length"] = length
            obj["tokens"] = tokenized_txt["input_ids"]

            if num_copies not in buckets:
                buckets[num_copies] = []
            else:
                buckets[num_copies].append(obj)

        for key, values in buckets.items():
            if len(values) <= num_files_to_keep:
                pruned_buckets[key] = values
            else:
                sampled_values = random.sample(values, 250)
                pruned_buckets[key] = sampled_values

    with open(f"{save_path}/experiment_masterlist.json", "w") as json_file:
        json.dump(pruned_buckets, json_file)
