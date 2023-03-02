from memorization.core.sampling import *


def sample_entrypoint(cmd):
    """
    Order of execution:

    2. Create paths where the sampled and processed openwebtext will be stored
    3. Generate duplicates
    4. Generate duplicate statistics
    """
    # Assert that project path exists
    project_path = cmd.project_path
    assert os.path.exists(
        project_path
    ), "Provided project path doesn't exist. Check again."

    # Assert that the openwebtext has been unpacked
    dataset_path = cmd.dataset_path
    assert os.path.exists(
        dataset_path
    ), f"{dataset_path} doesn't exist. Make sure to download and unpack the openwebtext there."

    # Unpack dataset
    # dataset_path = cmd.dataset_path
    # assert os.path.exists(dataset_path), "Provided dataset path doesn't exist. Check again."
    # print("...Unpacking the dataset...")
    # unpack_dataset(dataset_path)

    # # Create dirs and stuff
    sampled_dataset_path, stats_path = create_target_paths(project_path)
    train_path, valid_path = sampled_dataset_path[0], sampled_dataset_path[1]

    # # Randomly sample a portion of the dataset (40 GB is too much)
    print("...Sampling from the original dataset...")
    sample_dataset(dataset_path, train_path, sample_ratio=0.05, split="train")
    # sample_dataset(dataset_path, valid_path, sample_ratio=0.025, split="valid")

    # Generate duplicates
    print("..Generating duplicates...")
    # generate_duplicates(train_path)
    # generate_duplicates(valid_path)
    generate_duplicates_controlled(train_path)

    # Generate duplicate statistics
    print("...Generating stats...")
    stats_folder_path = os.path.join(project_path, "memorization/dataset/stats")
    train_json = os.path.join(stats_folder_path, "train_stats")
    train_duplicates_json, train_nonduplicates_json = generate_stats(
        train_path, train_json
    )

    # # Generate stats list that's used for the experiments
    print("...Generating stats masterlist...")
    generate_stats_masterlist(
        [train_duplicates_json, train_nonduplicates_json], stats_path
    )
