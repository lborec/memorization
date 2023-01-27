"""
Project version command used to simply display the version to the terminal.
"""
from argparse import Namespace


__version__ = "0.0.0"


def version_entrypoint(cmd: Namespace) -> None:
    print(__version__)
