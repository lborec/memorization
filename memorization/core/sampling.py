import os
import shutil
import tarfile
import math
import random
import numpy as np
from memorization.core.helpers import progressBar

SAMPLING_TOKEN_LENGTHS = [150, 200, 250, 300, 350, 400, 450, 500]
SAMPLING_SUBFOLDER_PREFIX = "length_"


def create_target_paths(project_path):
    """
    This function does bookkeeping related to the target folder where the sampled openwebtext will be stored.
    """
    # Create {project_path}/memorization/dataset/sampled_dataset
    sampled_dataset_path = os.path.join(project_path, "memorization/dataset/sampled_dataset")

    # Create folder if it doesn't exist
    if not os.path.exists(sampled_dataset_path):
        os.makedirs(sampled_dataset_path)
        print(f"Sampled openwebtext will be stored at {sampled_dataset_path}")

    # Create {project_path}/memorization/datasetduplicates
    duplicates_path = os.path.join(project_path, "memorization/dataset/duplicates")

    # Create folder if it doesn't exist
    if not os.path.exists(duplicates_path):
        os.makedirs(duplicates_path)
        print(f"JSON files containing duplicates and their counts will be stored at {duplicates_path}")

    # Create subfolders where different length subsets will be stored, e.g. len = 50, 100, 150...
    for length in SAMPLING_TOKEN_LENGTHS:
        temp_json_path = os.path.join(duplicates_path, f"length_{length}")
        open(temp_json_path, "w").close()

    return sampled_dataset_path, duplicates_path


def unpack_dataset(unpacked_dataset_path):
    """
    Unpacks the .xz files in openwebtext. The files are actually folders containing multiple .txt files.
    The files are unpacked inside the directory.
    """
    for filename in progressBar(os.listdir(unpacked_dataset_path), prefix='Progress', suffix='Complete'):
        # Iterate through .xz files
        if filename.endswith(".xz"):
            filepath = os.path.join(unpacked_dataset_path, filename)
            with tarfile.open(fileobj=filepath, mode='r:xz') as tar:
                # Create a new folder
                new_folder_path = os.path.join(unpacked_dataset_path, filename[:-3])
                if not os.path.exists(new_folder_path):
                    os.mkdir(new_folder_path)
                # Extract into  folder
                tar.extractall(new_folder_path)
                # # Remove .xz file
                # os.remove(filepath)


def sample_dataset(dataset_path, sampled_dataset_path, sample_ratio=0.25):
    """
    This is the function that does the actual sampling of the original openwebtext and stores it to the target folder.
    """
    print("Beginning sampling...")
    # Get a list of all folders from the original dataset and prune out non-folder items
    all_folders_in_dataset = os.listdir(dataset_path)
    all_folders_in_dataset = [file for file in all_folders_in_dataset if
                              os.path.isdir(os.path.join(dataset_path, file))]

    # Get the indices of the folders that we will sample for the new dataset
    dataset_length = len(all_folders_in_dataset)
    num_files_to_keep = math.floor(dataset_length * sample_ratio)
    indices_to_keep = random.sample(range(dataset_length), num_files_to_keep)
    folders_to_keep = [all_folders_in_dataset[ind] for ind in indices_to_keep]

    # Move the chosen files from dataset_path to sampled_dataset_path
    for folder in progressBar(folders_to_keep, prefix='Progress', suffix='Complete'):
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
    all_folders = [folder for folder in all_folders if os.path.isdir(os.path.join(sampled_dataset_path, folder))]
    for folder in progressBar(all_folders, prefix='Progress', suffix='Complete'):
        folder_path = os.path.join(sampled_dataset_path, folder)
        all_files = os.listdir(folder_path)
        length = len(all_files)
        sample_indices = np.random.exponential(scale=0.5, size=length)
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


def generate_duplicate_stats():
    """
    Generates statistics about the duplicates in dataset/duplicates
    """
    # TODO: Can be implemented anytime
    pass
