"""Command-line interface for WFDB record checks."""

import argparse
from pathlib import Path

from wfdb.io.check import check_dataset, check_record


def main(argv=None):
    """Run the WFDB command-line interface."""
    parser = argparse.ArgumentParser(prog="wfdb")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check local WFDB records")
    check.add_argument("path", type=Path, help="record or dataset directory")
    args = parser.parse_args(argv)

    if args.path.is_dir():
        issues = check_dataset(args.path)
    else:
        issues = check_record(args.path)

    for issue in issues:
        print(f"{issue.path}: {issue.code}: {issue.message}")
    if not issues:
        print("No WFDB problems found")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
