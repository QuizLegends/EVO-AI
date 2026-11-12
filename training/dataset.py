"""
EVO AI - Training Dataset
Version: 0.2.0

Preserves complete programming examples as independent
training sequences.

Features:
- Keeps example boundaries.
- Does not concatenate unrelated examples.
- Does not arbitrarily cut examples into blocks.
- Rejects sequences exceeding the configured context.
- Supports variable-length sequences with batch_size=1.
"""

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EVODataset(Dataset):
    """
    Dataset for complete byte-level programming examples.

    Each item contains:
    - input_ids: all tokens except the last
    - targets: all tokens except the first

    Each example is processed independently.
    """

    EXAMPLE_MARKER = "TASK: PROGRAMMING PROBLEM"

    def __init__(
        self,
        file_path,
        tokenizer,
        block_size=8192
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.block_size = block_size

        if block_size < 1:
            raise ValueError(
                "block_size must be positive"
            )

        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}"
            )

        text = self.file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            raise ValueError(
                "Dataset file is empty"
            )

        # Separate examples without mixing their contents.
        if self.EXAMPLE_MARKER in text:
            raw_examples = re.split(
                r"(?=TASK: PROGRAMMING PROBLEM)",
                text
            )
        else:
            # Compatibility with ordinary text datasets.
            raw_examples = [text]

        self.examples = []

        for index, raw_example in enumerate(raw_examples):

            raw_example = raw_example.strip()

            if not raw_example:
                continue

            tokens = self.tokenizer.encode(
                raw_example
            )

            if len(tokens) < 2:
                raise ValueError(
                    f"Example {index} has insufficient tokens."
                )

            input_length = len(tokens) - 1

            if input_length > self.block_size:
                raise ValueError(
                    f"Example {index} exceeds context limit: "
                    f"{input_length} > {self.block_size}. "
                    "Increase the context or prepare the example."
                )

            self.examples.append(
                torch.tensor(
                    tokens,
                    dtype=torch.long
                )
            )

        if not self.examples:
            raise ValueError(
                "No valid training examples found."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        if index < 0 or index >= len(self):
            raise IndexError(index)

        tokens = self.examples[index]

        input_ids = tokens[:-1]
        targets = tokens[1:]

        return input_ids, targets
