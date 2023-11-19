from memorization.core.running_experiments import *


def run_experiments_entrypoint(cmd):
    # project_path = cmd.project_path
    model_path = cmd.model_path
    json_file_path = "memorization/dataset/stats/experiment_masterlist.json"
    save_path = "memorization/dataset/stats/results"

    run_experiments(model_path, json_file_path, save_path, "greedy_decoding")
    top_p_values = [0.2, 0.4, 0.6, 0.8]
    for top_p in top_p_values:
        run_experiments(model_path, json_file_path, save_path, "nucleus_sampling", top_p)
