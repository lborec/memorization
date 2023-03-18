from memorization.core.evaluate import *


def evaluate_entrypoint(cmd):
    model_identifier = cmd.model_identifier

    calculate_perplexity(model_identifier)
