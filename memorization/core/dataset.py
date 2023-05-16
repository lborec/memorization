import os
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def load_tokenizer():
    # Load the GPT tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    return tokenizer


# webtxt_dataset = load_dataset(
#     "text", data_dir="memorization/dataset/sampled_dataset", sample_by="document"
# )

# class WebTxtDataset(Dataset):
#     def __init__(self, tokenizer, root_dir="memorization/dataset/sampled_dataset"):
#         self.root_dir = root_dir
#         self.tokenizer = tokenizer
#         self.file_list = []
#         for subdir, _, files in os.walk(root_dir):
#             for file in files:
#                 if file.endswith('.txt'):
#                     self.file_list.append(os.path.join(subdir, file))
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         with open(self.file_list[idx], 'r') as f:
#             file = f.read()
#             tokens = self.tokenizer(file, truncation=True, max_length=512)
#             return tokens

# def load_data_loader(default_path="memorization/dataset/sampled_dataset", batch_size=8):
#     webtxtdataset = WebTxtDataset(default_path)
#     data_loader = DataLoader(webtxtdataset, batch_size=batch_size, shuffle=True)
#     return data_loader
