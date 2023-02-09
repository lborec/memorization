from memorization.core.training import *


def train_entrypoint(cmd):
    model_type = cmd.model_type
    train_transformer(model_type)
