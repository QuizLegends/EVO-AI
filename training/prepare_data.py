
"""
EVO AI - Dataset Preparation
Version: 0.1.0

Collects authorized programming and documentation files
and combines them into a text corpus for training.
"""

import argparse
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".json"
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode"
}

IGNORED_FILENAMES = {
    ".env",
    ".env.local",
    "secrets.json",
    "credentials.json"
}


def collect_files(source, max_file_size):
    source = Path(source)

    if not source.is_dir():
        raise NotADirectoryError(source)

    collected = []

    for path in sorted(source.rglob("*")):

        if not path.is_file():
            continue

        if any(
            part in IGNORED_DIRECTORIES
            for part in path.parts
        ):
            continue

        if path.name in IGNORED_FILENAMES:
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            size = path.stat().st_size

            if size == 0 or size > max_file_size:
                continue

            collected.append(path)

        except OSError:
            continue

    return collected


def prepare_dataset(
    source,
    output,
    max_file_size=1_000_000
):
    if max_file_size < 1:
        raise ValueError(
            "Maximum file size must be positive."
        )

    source = Path(source).resolve()
    output = Path(output).resolve()

    if output == source or source in output.parents:
        raise ValueError(
            "Output file cannot be inside the source directory."
        )

    files = collect_files(
        source,
        max_file_size
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_files = 0
    total_characters = 0

    with output.open(
        "w",
        encoding="utf-8"
    ) as destination:

        for path in files:

            try:
                content = path.read_text(
                    encoding="utf-8"
                ).strip()

            except (OSError, UnicodeError):
                continue

            if not content:
                continue

            relative_path = path.relative_to(
                source
            )

            destination.write(
                f"\n\n### FILE: {relative_path}\n\n"
            )

            destination.write(content)

            total_files += 1
            total_characters += len(content)

    if total_files == 0:
        output.unlink(missing_ok=True)

        raise ValueError(
            "No usable text files were found."
        )

    return {
        "output": str(output),
        "files_processed": total_files,
        "total_characters": total_characters
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare EVO AI training data."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing authorized source material."
    )

    parser.add_argument(
        "--output",
        default="datasets/train.txt"
    )

    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1_000_000
    )

    args = parser.parse_args()

    result = prepare_dataset(
        source=args.source,
        output=args.output,
        max_file_size=args.max_file_size
    )

    print("\n===== DATASET PREPARATION =====")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("===============================")


if __name__ == "__main__":
    main()
