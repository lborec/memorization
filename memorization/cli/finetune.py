from memorization.core.finetuning import *

ALLOWED_VERSIONS = ["small", "xl"]


def finetune_entrypoint(cmd):
    model_version = cmd.model_version
    assert (
        model_version.lower() in ALLOWED_VERSIONS
    ), f"Allowed models are: {ALLOWED_VERSIONS}"

    gpt2_version = f"gpt2-{model_version}"

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)
    model = GPT2Model.from_pretrained(gpt2_version)
