"""Replace this with one small, standalone demo."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = 2 + 2
    assert result == 4
    print("PASS: example works" if args.verify else f"Result: {result}")


if __name__ == "__main__":
    main()
