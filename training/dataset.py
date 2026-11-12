
"""
EVO AI - Training Dataset
Version: 0.1.0

Loads text data and prepares byte-level sequences
for next-token language model training.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class EVODataset(Dataset):
    """
    Dataset for training EVO AI.

    Each item contains:
    - input_ids: sequence of bytes
    - targets: the same sequence shifted by one position
    """

    def __init__(
        self,
        file_path,
        tokenizer,
        block_size=128
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.block_size = block_size

        if block_size < 1:
            raise ValueError("block_size must be positive")

        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}"
            )

        text = self.file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            raise ValueError("Dataset file is empty")

        tokens = self.tokenizer.encode(text)

        if len(tokens) <= block_size:
            raise ValueError(
                "Dataset must contain more bytes than block_size"
            )

        self.data = torch.tensor(
            tokens,
            dtype=torch.long
        )

    def __len__(self):
        return (
            len(self.data) - 1
        ) // self.block_size

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.block_size

        chunk = self.data[
            start:start + self.block_size + 1
        ]

        input_ids = chunk[:-1]
        targets = chunk[1:]

        return input_ids, targets
