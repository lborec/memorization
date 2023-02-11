from transformers import GPT2Config, AutoConfig


VOCAB_SIZE = 50257

gpt2_100_config = GPT2Config(
    vocab_size=VOCAB_SIZE, n_embd=512, n_head=16, n_layer=24  # 768
)  # 9

gpt2_250_config = GPT2Config(vocab_size=VOCAB_SIZE, n_embd=1024, n_head=16, n_layer=16)

gpt2_500_config = GPT2Config(vocab_size=VOCAB_SIZE, n_embd=1024, n_head=16, n_layer=32)
