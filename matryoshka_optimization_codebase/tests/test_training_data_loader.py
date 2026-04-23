from __future__ import annotations

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover - optional at test collection time
    Dataset = None

from matryoshka_exp.data.training_data import TrainingDatasetLoader

try:
    from matryoshka_exp.config import load_config
except ImportError:  # pragma: no cover - optional at test collection time
    load_config = None


@unittest.skipIf(Dataset is None, "datasets is required for loader tests")
class TrainingDatasetLoaderTests(unittest.TestCase):
    def test_hf_triplet_projects_and_renames_columns(self):
        ds = Dataset.from_dict(
            {
                "query": ["q1"],
                "positive": ["p1"],
                "negative": ["n1"],
                "extra": ["drop-me"],
            }
        )
        with patch("datasets.load_dataset", return_value=ds):
            loader = TrainingDatasetLoader(
                dataset_name="dummy",
                split="train",
                fmt="hf_triplet",
                query_column="query",
                positive_column="positive",
                negative_column="negative",
            )
            out = loader.load()

        self.assertEqual(out.column_names, ["anchor", "positive", "negative"])
        self.assertEqual(out.num_rows, 1)
        self.assertEqual(out[0]["anchor"], "q1")
        self.assertEqual(out[0]["positive"], "p1")
        self.assertEqual(out[0]["negative"], "n1")

    def test_hf_tevatron_passage_explodes_triplets(self):
        ds = Dataset.from_dict(
            {
                "query": ["q1"],
                "positive_passages": [[{"text": "p1"}, {"text": "p2"}]],
                "negative_passages": [[{"text": "n1"}, {"text": "n2"}]],
            }
        )
        with patch("datasets.load_dataset", return_value=ds):
            loader = TrainingDatasetLoader(
                dataset_name="dummy",
                split="train",
                fmt="hf_tevatron_passage",
                query_column="query",
                positive_column="positive",
                negative_column="negative",
                tevatron_positive_passages_column="positive_passages",
                tevatron_negative_passages_column="negative_passages",
                tevatron_passage_text_field="text",
            )
            out = loader.load()

        self.assertEqual(out.column_names, ["anchor", "positive", "negative"])
        self.assertEqual(out.num_rows, 4)
        self.assertEqual(
            set((row["positive"], row["negative"]) for row in out),
            {("p1", "n1"), ("p1", "n2"), ("p2", "n1"), ("p2", "n2")},
        )

    def test_hf_triplet_raises_on_missing_column(self):
        ds = Dataset.from_dict({"query": ["q1"], "positive": ["p1"]})
        with patch("datasets.load_dataset", return_value=ds):
            loader = TrainingDatasetLoader(
                dataset_name="dummy",
                split="train",
                fmt="hf_triplet",
                query_column="query",
                positive_column="positive",
                negative_column="negative",
            )
            with self.assertRaises(ValueError):
                loader.load()

    def test_hf_tevatron_passage_raises_on_malformed_passages(self):
        ds = Dataset.from_dict(
            {
                "query": ["q1"],
                "positive_passages": [["not-a-dict"]],
                "negative_passages": [[{"text": "n1"}]],
            }
        )
        with patch("datasets.load_dataset", return_value=ds):
            loader = TrainingDatasetLoader(
                dataset_name="dummy",
                split="train",
                fmt="hf_tevatron_passage",
                query_column="query",
                positive_column="positive",
                negative_column="negative",
                tevatron_positive_passages_column="positive_passages",
                tevatron_negative_passages_column="negative_passages",
                tevatron_passage_text_field="text",
            )
            with self.assertRaises(ValueError):
                loader.load()

    def test_hf_triplet_raises_on_zero_rows(self):
        ds = Dataset.from_dict({"query": [], "positive": [], "negative": []})
        with patch("datasets.load_dataset", return_value=ds):
            loader = TrainingDatasetLoader(
                dataset_name="dummy",
                split="train",
                fmt="hf_triplet",
                query_column="query",
                positive_column="positive",
                negative_column="negative",
            )
            with self.assertRaises(ValueError):
                loader.load()


class TrainingDatasetLoaderValidationTests(unittest.TestCase):
    def test_jsonl_triplet_requires_training_local_path(self):
        loader = TrainingDatasetLoader(
            dataset_name=None,
            split="train",
            fmt="jsonl_triplet",
            query_column="query",
            positive_column="positive",
            negative_column="negative",
            training_local_path=None,
        )
        with self.assertRaises(ValueError):
            loader.load()


@unittest.skipIf(load_config is None, "pyyaml is required for config consistency tests")
class ConfigConsistencyTests(unittest.TestCase):
    def test_tevatron_training_configs_are_aligned(self):
        paths = [
            ROOT / "configs" / "finetune_nomic_msmarco_batch.yaml",
            ROOT / "configs" / "finetune_nomic_lora_msmarco_batch.yaml",
        ]
        for path in paths:
            cfg = load_config(path)
            self.assertEqual(cfg.data.training_dataset_name, "Tevatron/msmarco-passage")
            self.assertEqual(cfg.data.training_format, "hf_tevatron_passage")
            self.assertEqual(cfg.data.query_column, "query")
            self.assertEqual(cfg.data.tevatron_positive_passages_column, "positive_passages")
            self.assertEqual(cfg.data.tevatron_negative_passages_column, "negative_passages")
            self.assertEqual(cfg.data.tevatron_passage_text_field, "text")


if __name__ == "__main__":
    unittest.main()
