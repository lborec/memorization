from memorization.core.training import *

# ALLOWED_MODELS = ["lstm", "transformer"]

def train_entrypoint(cmd):
    train_transformer()

# TODO: Read in parameters from a config file like in biskia
# TODO: Have predefined files for 50, 100, 250, 500 mil parameters? roughly
# model_type = cmd.model_type
# assert model_type.lower() in ALLOWED_MODELS, f"Allowed models are: {ALLOWED_MODELS}"

# Load the model
# model = load_model(mock_model_type, mock_params)

# Create data loaders for training and validation
# data_loader_train = load_data_loader() # split="train
# for i, e in enumerate(data_loader_train):
#     print(i,e)
#     break
