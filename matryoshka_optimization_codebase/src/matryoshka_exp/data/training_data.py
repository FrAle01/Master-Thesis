from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm


@dataclass
class TripletExample:
    query: str
    positive: str
    negative: Optional[str] = None


class TrainingDatasetLoader:
    """Loads training data for fine-tuning.

    Supported formats are intentionally simple:
    - Hugging Face dataset with explicit query/positive[/negative] columns
    - Hugging Face Tevatron MSMARCO passage schema
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
        tevatron_positive_passages_column: str = "positive_passages",
        tevatron_negative_passages_column: str = "negative_passages",
        tevatron_passage_text_field: str = "text",
        training_local_path: Optional[str] = None,
        local_path: Optional[str] = None,
        logger: Optional[Any] = None,
        verbose: bool = True,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.fmt = fmt
        self.query_column = query_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.tevatron_positive_passages_column = tevatron_positive_passages_column
        self.tevatron_negative_passages_column = tevatron_negative_passages_column
        self.tevatron_passage_text_field = tevatron_passage_text_field
        self.training_local_path = training_local_path or local_path
        self.verbose = verbose
        self.logger = logger
    def load(self):
        if self.fmt == "hf_triplet":
            if not self.dataset_name:
                raise ValueError("`dataset_name` is required for `hf_triplet` format.")
            from datasets import load_dataset

            self.logger.info("Loading dataset '%s' split '%s' for training, using %s", self.dataset_name, self.split, self.fmt)
            ds = load_dataset(self.dataset_name, split=self.split)
            self.logger.info("Loaded dataset with %d rows", ds.num_rows)
            return self._normalize_triplet_dataset(ds)
        
        if self.fmt == "hf_tevatron_passage":
            if not self.dataset_name:
                raise ValueError("`dataset_name` is required for `hf_tevatron_passage` format.")
            from datasets import load_dataset

            self.logger.info("Loading dataset '%s' split '%s' for training, using %s", self.dataset_name, self.split, self.fmt)
            ds = load_dataset(self.dataset_name, split=self.split)
            self.logger.info("Loaded dataset with %d rows", ds.num_rows)
            return self._normalize_tevatron_passage_dataset(ds)
        if self.fmt == "jsonl_triplet":
            if not self.training_local_path:
                raise ValueError("`training_local_path` is required for `jsonl_triplet` format.")
            from datasets import load_dataset

            self.logger.info("Loading dataset from local path '%s' for training, using %s", self.training_local_path, self.fmt)
            ds = load_dataset("json", data_files=self.training_local_path, split="train")
            self.logger.info("Loaded dataset with %d rows", ds.num_rows)
            return self._normalize_triplet_dataset(ds)
        raise ValueError(f"Unsupported training format: {self.fmt}")

    def _normalize_triplet_dataset(self, ds):
        required_columns = [self.query_column, self.positive_column]
        output_columns = ["anchor", "positive"]
        if self.negative_column:
            required_columns.append(self.negative_column)
            output_columns.append("negative")

        self._validate_required_columns(ds, required_columns, self.fmt)
        normalized = ds.select_columns(required_columns).rename_columns(dict(zip(required_columns, output_columns)))
        self.logger.info("Normalized dataset with %d rows", normalized.num_rows)
        self._validate_non_empty_rows(normalized, output_columns, self.fmt)
        return normalized

    def _normalize_tevatron_passage_dataset(self, ds):
        from datasets import Dataset

        required_columns = [
            self.query_column,
            self.tevatron_positive_passages_column,
        ]
        has_negatives = bool(self.tevatron_negative_passages_column)
        if has_negatives:
            required_columns.append(self.tevatron_negative_passages_column)
        self._validate_required_columns(ds, required_columns, self.fmt)

        def generator():
            for row_idx, row in enumerate(
                tqdm(ds, desc="Normalizing Tevatron passages", disable=not self.verbose)
            ):
                query = self._validate_text(row.get(self.query_column), row_idx, self.query_column, self.fmt)
                positives = self._extract_passage_texts(
                    row=row,
                    row_idx=row_idx,
                    column=self.tevatron_positive_passages_column,
                )

                if has_negatives:
                    negatives = self._extract_passage_texts(
                        row=row,
                        row_idx=row_idx,
                        column=self.tevatron_negative_passages_column,
                    )
                    for positive_text in positives:
                        for negative_text in negatives:
                            yield {"anchor": query, "positive": positive_text, "negative": negative_text}
                else:
                    for positive_text in positives:
                        yield {"anchor": query, "positive": positive_text}

        normalized = Dataset.from_generator(generator)
        expected_output_columns = ["anchor", "positive", "negative"] if has_negatives else ["anchor", "positive"]
        self._ensure_non_zero_rows(normalized, self.fmt)
        self._validate_non_empty_rows(normalized, expected_output_columns, self.fmt)
        return normalized

    def _extract_passage_texts(self, row: dict[str, Any], row_idx: int, column: str) -> list[str]:
        passages = row.get(column)
        if not isinstance(passages, list):
            raise ValueError(
                f"`{self.fmt}` expects `{column}` to be a list at row {row_idx}, got {type(passages).__name__}."
            )
        if not passages:
            raise ValueError(f"`{self.fmt}` expects non-empty `{column}` at row {row_idx}.")

        texts: list[str] = []
        for passage_idx, passage in enumerate(passages):
            if not isinstance(passage, dict):
                raise ValueError(
                    f"`{self.fmt}` expects `{column}` elements to be objects at row {row_idx}, "
                    f"index {passage_idx}, got {type(passage).__name__}."
                )
            text = self._validate_text(
                passage.get(self.tevatron_passage_text_field),
                row_idx=row_idx,
                column=f"{column}[{passage_idx}].{self.tevatron_passage_text_field}",
                fmt=self.fmt,
            )
            texts.append(text)
        return texts

    def _validate_required_columns(self, ds, columns: list[str], fmt: str) -> None:
        missing = [column for column in columns if column not in ds.column_names]
        if missing:
            raise ValueError(
                f"Missing required columns for `{fmt}`: {missing}. "
                f"Available columns: {list(ds.column_names)}"
            )

    def _ensure_non_zero_rows(self, ds, fmt: str) -> None:
        if ds.num_rows == 0:
            raise ValueError(f"`{fmt}` produced an empty training dataset.")

    def _validate_non_empty_rows(self, ds, text_columns: list[str], fmt: str) -> None:
        self._ensure_non_zero_rows(ds, fmt)
        for row_idx, row in enumerate(ds):
            for column in text_columns:
                self._validate_text(row.get(column), row_idx, column, fmt)

    @staticmethod
    def _validate_text(value: Any, row_idx: int, column: str, fmt: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"`{fmt}` expects non-empty string in column `{column}` at row {row_idx}, "
                f"got {type(value).__name__}."
            )
        return value
