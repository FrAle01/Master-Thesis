from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class TripletExample:
    query: str
    positive: str
    negative: Optional[str] = None


class TrainingDatasetLoader:
    """Loads training data for fine-tuning.

    Supported formats are intentionally simple:
    - Hugging Face dataset with explicit query/positive[/negative] columns
    - local JSONL with the same fields
    """

    def __init__(
        self,
        dataset_name: Optional[str],
        split: str,
        fmt: str,
        query_column: str,
        positive_column: str,
        negative_column: Optional[str],
        local_path: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.fmt = fmt
        self.query_column = query_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.local_path = local_path

    def load(self):
        if self.fmt == "hf_triplet":
            from datasets import load_dataset

            ds = load_dataset(self.dataset_name, split=self.split)
            return ds
        if self.fmt == "jsonl_triplet":
            from datasets import load_dataset

            ds = load_dataset("json", data_files=self.local_path, split="train")
            return ds
        raise ValueError(f"Unsupported training format: {self.fmt}")
