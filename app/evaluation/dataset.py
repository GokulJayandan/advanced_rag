"""
Evaluation dataset — loads/saves curated test questions with expected answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    """A single evaluation item."""
    query: str
    expected_answer: str = ""
    expected_sources: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvaluationDataset:
    """Manages evaluation datasets."""

    def __init__(self, filepath: str | Path | None = None):
        self.filepath = Path(filepath) if filepath else None
        self.items: list[EvalItem] = []
        if self.filepath and self.filepath.exists():
            self.load(self.filepath)

    def load(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        with open(self.filepath, "r") as f:
            data = json.load(f)
        self.items = [EvalItem(**item) for item in data.get("items", data)]

    def save(self, filepath: str | Path | None = None) -> None:
        path = Path(filepath) if filepath else self.filepath
        if path is None:
            raise ValueError("No filepath specified")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"items": [item.model_dump() for item in self.items]}, f, indent=2)

    def add_item(self, item: EvalItem) -> None:
        self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
