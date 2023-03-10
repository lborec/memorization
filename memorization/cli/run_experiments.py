from memorization.core.running_experiments import *


def run_experiments_entrypoint(cmd):
    # project_path = cmd.project_path
    model_path = cmd.model_path
    json_file_path = "memorization/dataset/stats/experiment_masterlist.json"
    save_path = "memorization/dataset/stats/results"

    run_experiments(model_path, json_file_path, save_path, "greedy_decoding")
    # run_experiments(model_path, json_file_path, save_path, "nucleus_sampling")
