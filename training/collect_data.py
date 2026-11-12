
"""
EVO AI - Programming Dataset Collector
Version: 0.1.0

Downloads and filters authorized programming datasets
from Hugging Face using streaming mode.

This script does NOT train the neural network.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


DEFAULT_DATASET = "nvidia/OpenCodeInstruct"

ALLOWED_LICENSES = {
    "cc-by-4.0",
    "cc0-1.0",
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cdla-permissive-2.0",
}


def normalize_text(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return "\n".join(
            normalize_text(item)
            for item in value
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False
        )

    return str(value).strip()


def get_test_status(record):
    statuses = record.get(
        "tests_execution_status"
    )

    if not statuses:
        return None

    if isinstance(statuses, str):
        try:
            statuses = json.loads(statuses)
        except json.JSONDecodeError:
            statuses = [statuses]

    if not isinstance(statuses, list):
        return None

    return all(
        str(status).strip().lower() == "pass"
        for status in statuses
    )


def get_test_score(record):
    value = record.get("average_test_score")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_example(record, dataset_id):
    instruction = normalize_text(
        record.get("input")
        or record.get("instruction")
        or record.get("prompt")
        or record.get("question")
    )

    solution = normalize_text(
        record.get("output")
        or record.get("response")
        or record.get("answer")
        or record.get("completion")
    )

    tests = normalize_text(
        record.get("unit_tests")
        or record.get("tests")
    )

    if not instruction or not solution:
        return None

    sample_id = normalize_text(
        record.get("id")
    )

    sections = [
        "TASK: PROGRAMMING PROBLEM",
        f"SOURCE: {dataset_id}",
        f"SAMPLE_ID: {sample_id or 'unavailable'}",
        "",
        "INSTRUCTION:",
        instruction,
        "",
        "SOLUTION:",
        solution,
    ]

    if tests:
        sections.extend([
            "",
            "TESTS:",
            tests,
        ])

    return "\n".join(sections)


def collect_dataset(
    dataset_id,
    split,
    output,
    manifest,
    license_name,
    max_examples,
    max_scan,
    min_test_score,
    max_chars
):
    if license_name not in ALLOWED_LICENSES:
        raise ValueError(
            "License not in the approved list. "
            "Verify the dataset license first."
        )

    if max_examples < 1 or max_scan < 1:
        raise ValueError(
            "Example limits must be positive."
        )

    if max_chars < 100:
        raise ValueError(
            "Maximum sample size is too small."
        )

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install dependencies with: "
            "pip install datasets"
        ) from error

    print("EVO AI - Dataset collection")
    print("Dataset:", dataset_id)
    print("Split:", split)
    print("Mode: streaming")
    print("Maximum accepted examples:", max_examples)

    dataset = load_dataset(
        dataset_id,
        split=split,
        streaming=True
    )

    output = Path(output)
    manifest = Path(manifest)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    accepted = 0
    scanned = 0
    rejected = 0

    with output.open(
        "w",
        encoding="utf-8"
    ) as destination:

        for record in dataset:

            if scanned >= max_scan:
                break

            scanned += 1

            score = get_test_score(record)
            test_status = get_test_status(record)

            if (
                score is not None
                and score < min_test_score
            ):
                rejected += 1
                continue

            if test_status is False:
                rejected += 1
                continue

            content = format_example(
                record,
                dataset_id
            )

            if not content:
                rejected += 1
                continue

            if len(content) > max_chars:
                rejected += 1
                continue

            destination.write(
                content + "\n\n"
            )

            accepted += 1

            if accepted % 100 == 0:
                print(
                    f"Accepted: {accepted} "
                    f"| Scanned: {scanned}"
                )

            if accepted >= max_examples:
                break

    if accepted == 0:
        output.unlink(missing_ok=True)

        raise RuntimeError(
            "No usable examples were collected. "
            "Check the dataset schema and filters."
        )

    report = {
        "project": "EVO AI",
        "dataset": dataset_id,
        "source_url": (
            "https://huggingface.co/datasets/"
            + dataset_id
        ),
        "license": license_name,
        "split": split,
        "collected_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "examples_scanned": scanned,
        "examples_accepted": accepted,
        "examples_rejected": rejected,
        "minimum_test_score": min_test_score,
        "output": str(output),
        "note": (
            "Verify dataset terms and attribution "
            "before redistribution."
        )
    }

    manifest.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nCOLLECTION COMPLETED")
    print("Accepted:", accepted)
    print("Scanned:", scanned)
    print("Rejected:", rejected)
    print("Dataset file:", output)
    print("Manifest:", manifest)

    return report


def main():

    parser = argparse.ArgumentParser(
        description="Collect programming data for EVO AI."
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--split",
        default="train"
    )

    parser.add_argument(
        "--output",
        default="/content/evo-data/opencode_sample.txt"
    )

    parser.add_argument(
        "--manifest",
        default="/content/evo-data/collection_manifest.json"
    )

    parser.add_argument(
        "--license",
        required=True
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--max-scan",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--min-test-score",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=30000
    )

    args = parser.parse_args()

    collect_dataset(
        dataset_id=args.dataset,
        split=args.split,
        output=args.output,
        manifest=args.manifest,
        license_name=args.license.lower(),
        max_examples=args.max_examples,
        max_scan=args.max_scan,
        min_test_score=args.min_test_score,
        max_chars=args.max_chars
    )


if __name__ == "__main__":
    main()
