from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS


# from . import sample, version, train, finetune

# THIS IS UGLY BUT IT AVOIDS "from . import sample, version, train, finetune..." WHICH IS VERY SLOW
def sample_subcommand(args):
    from . import sample

    sample.sample_entrypoint(args)


def train_subcommand(args):
    from . import train

    train.train_entrypoint(args)


def finetune_subcommand(args):
    from . import finetune

    finetune.finetune_entrypoint(args)


# Entry points
def parse_args():
    # PARSER OBJECT
    parser = ArgumentParser(
        description="Arg parser", formatter_class=RawTextHelpFormatter, usage=SUPPRESS
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # SAMPLE ARGS
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument(
        "--project_path",
        type=str,
        help="Path of the memorization project. E.g., /Users/luka/memorization.",
        required=True,
    )
    sample_parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path of the unpacked openwebtext dataset.",
        required=True,
    )
    sample_parser.set_defaults(func=sample_subcommand)

    # TRAIN ARGS
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model to train. Choices: LSTM, Transformer",
        required=True,
    )
    train_parser.set_defaults(func=train_subcommand)

    # FINETUNE ARGS
    finetune_parser = subparsers.add_parser("finetune")
    finetune_parser.add_argument(
        "--model_version",
        type=str,
        help="GPT-2 model version to finetune. Choices: small, xl",
        required=True,
    )
    finetune_parser.set_defaults(func=finetune_subcommand)

    # ~EL FIN~
    return parser.parse_args()
