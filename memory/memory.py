
"""
EVO AI - Persistent Memory
Version: 0.1.0

Stores interactions and examples for future training.
This module does not modify neural network weights.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class EVOMemory:

    def __init__(self, storage_path="memory/data.jsonl"):
        self.storage_path = Path(storage_path)

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.storage_path.exists():
            self.storage_path.touch()

    def add(
        self,
        role,
        content,
        tags=None
    ):
        if role not in ("user", "assistant"):
            raise ValueError(
                "Role must be 'user' or 'assistant'."
            )

        if not isinstance(content, str):
            raise TypeError("Content must be a string.")

        if not content.strip():
            raise ValueError("Content cannot be empty.")

        entry = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "role": role,
            "content": content,
            "tags": tags or []
        }

        with self.storage_path.open(
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    entry,
                    ensure_ascii=False
                ) + "\n"
            )

        return entry

    def get_all(self):
        entries = []

        with self.storage_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                try:
                    entries.append(
                        json.loads(line)
                    )
                except json.JSONDecodeError:
                    continue

        return entries

    def search(self, query, limit=20):
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if limit < 1:
            raise ValueError("Limit must be positive.")

        query = query.casefold()

        results = []

        for entry in reversed(self.get_all()):
            content = entry.get("content", "")

            if query in content.casefold():
                results.append(entry)

            if len(results) >= limit:
                break

        return results

    def export_training_text(
        self,
        output_path="memory/training_data.txt"
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        entries = self.get_all()

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            for entry in entries:
                role = entry.get("role")
                content = entry.get("content", "")

                if role == "user":
                    label = "User"
                elif role == "assistant":
                    label = "EVO AI"
                else:
                    continue

                file.write(
                    f"{label}: {content}\n\n"
                )

        return output_path

    def count(self):
        return len(self.get_all())
