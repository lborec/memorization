from memorization.cli import parse_args


def main():
    cmd = parse_args()

    if type(cmd.func) is list:
        for func in cmd.func:
            func(cmd)
    else:
        cmd.func(cmd)


if __name__ == "__main__":
    main()
