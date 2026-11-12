
"""
EVO AI - Byte-Level Tokenizer
Version: 0.1.0

UTF-8 byte tokenizer.
Compatible with EVOConfig(vocab_size=256).
"""

from pathlib import Path


class EVOTokenizer:
    """Converts UTF-8 text into numerical tokens."""

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        return list(text.encode("utf-8"))

    def decode(self, tokens):
        values = []

        for token in tokens:
            token = int(token)

            if not 0 <= token < self.vocab_size:
                raise ValueError(
                    f"Invalid token ID: {token}"
                )

            values.append(token)

        return bytes(values).decode(
            "utf-8",
            errors="replace"
        )

    def encode_batch(self, texts):
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list or tuple")

        return [self.encode(text) for text in texts]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_text(
            "EVO_BYTE_TOKENIZER_V1",
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        version = path.read_text(encoding="utf-8")

        if version != "EVO_BYTE_TOKENIZER_V1":
            raise ValueError("Unknown tokenizer version")

        return cls()
