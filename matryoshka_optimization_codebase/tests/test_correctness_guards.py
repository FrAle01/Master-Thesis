from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional in some environments
    pd = None

try:
    import torch
except ImportError:  # pragma: no cover - optional in some environments
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ImportError:  # pragma: no cover - optional in some environments
    yaml = None

try:
    from matryoshka_exp.config import (
        DataConfig,
        ExecutionConfig,
        ExperimentConfig,
        ModelConfig,
        OptimizationConfig,
        ProfileConfig,
        RetrievalConfig,
        TrainingConfig,
        UtilityConfig,
        load_config,
    )
except ImportError:  # pragma: no cover - optional in some environments
    DataConfig = None
    ExecutionConfig = None
    ExperimentConfig = None
    ModelConfig = None
    OptimizationConfig = None
    ProfileConfig = None
    RetrievalConfig = None
    TrainingConfig = None
    UtilityConfig = None
    load_config = None

try:
    from matryoshka_exp.data.pyterrier_utils import PyTerrierLoader
except ImportError:  # pragma: no cover - optional in some environments
    PyTerrierLoader = None

try:
    from matryoshka_exp.experiment import ExperimentRunner
except ImportError:  # pragma: no cover - optional in some environments
    ExperimentRunner = None

try:
    from matryoshka_exp.models.base import build_profile_catalog, RepresentationProfile
except ImportError:  # pragma: no cover - optional in some environments
    build_profile_catalog = None
    RepresentationProfile = None

try:
    from matryoshka_exp.optimization.lagrangian import LagrangianProfileOptimizer
except ImportError:  # pragma: no cover - optional in some environments
    LagrangianProfileOptimizer = None

try:
    from matryoshka_exp.retrieval.dense import DenseGroupedRetriever
except ImportError:  # pragma: no cover - optional in some environments
    DenseGroupedRetriever = None


class _DummyAdapter:
    def similarity(self, queries, docs):
        return queries @ docs.T


@unittest.skipIf(
    DenseGroupedRetriever is None or RepresentationProfile is None or pd is None or torch is None,
    "core modules unavailable",
)
class DenseRetrieverGuardsTests(unittest.TestCase):
    def test_rerank_candidates_returns_empty_schema_when_candidates_empty(self):
        retriever = DenseGroupedRetriever(
            adapter=_DummyAdapter(),
            profiles={"full": RepresentationProfile("full", 4, None, True, 8)},
            similarity="dot",
            top_k=10,
        )
        out = retriever.rerank_candidates(
            candidates=pd.DataFrame(columns=["qid", "docno", "score", "rank"]),
            query_embeddings_full={},
            corpus_lookup={},
        )
        self.assertEqual(list(out.columns), ["qid", "docno", "score", "rank"])
        self.assertTrue(out.empty)


@unittest.skipIf(LagrangianProfileOptimizer is None or RepresentationProfile is None, "optimizer modules unavailable")
class OptimizerGuardTests(unittest.TestCase):
    def test_choose_profile_online_raises_for_empty_profile_catalog(self):
        optimizer = LagrangianProfileOptimizer(profiles=[], budget_bytes=1, max_iter=1, tolerance=1e-3)
        with self.assertRaises(ValueError):
            optimizer.choose_profile_online({}, 0.0)

    def test_choose_profile_online_raises_for_missing_profile_utility(self):
        profiles = [RepresentationProfile("full", 8, None, True, 16)]
        optimizer = LagrangianProfileOptimizer(profiles=profiles, budget_bytes=16, max_iter=1, tolerance=1e-3)
        with self.assertRaises(ValueError):
            optimizer.choose_profile_online({}, 0.0)


@unittest.skipIf(PyTerrierLoader is None or DataConfig is None or pd is None, "data loader modules unavailable")
class PyTerrierLoaderGuardsTests(unittest.TestCase):
    def test_iter_corpus_local_respects_max_docs(self):
        cfg = DataConfig(local_corpus_path="unused.parquet", max_docs=2, text_fields=["text"], docno_column="docno")
        loader = PyTerrierLoader(cfg)
        fake_df = pd.DataFrame(
            [
                {"docno": "d1", "text": "t1"},
                {"docno": "d2", "text": "t2"},
                {"docno": "d3", "text": "t3"},
            ]
        )
        with patch("pandas.read_parquet", return_value=fake_df):
            records = list(loader.iter_corpus())
        self.assertEqual([r.docno for r in records], ["d1", "d2"])

    def test_load_qrels_local_respects_max_queries(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            topics = pd.DataFrame([{"qid": "q1", "query": "one"}])
            qrels = pd.DataFrame([{"qid": "q1", "docno": "d1"}, {"qid": "q2", "docno": "d2"}])
            topics_path = tmp / "topics.tsv"
            qrels_path = tmp / "qrels.tsv"
            topics.to_csv(topics_path, sep="\t", index=False)
            qrels.to_csv(qrels_path, sep="\t", index=False)

            cfg = DataConfig(
                local_topics_path=str(topics_path),
                local_qrels_path=str(qrels_path),
                max_queries=1,
            )
            loader = PyTerrierLoader(cfg)
            out = loader.load_qrels()

        self.assertEqual(set(out["qid"].astype(str).tolist()), {"q1"})


@unittest.skipIf(
    ExperimentRunner is None or ExperimentConfig is None or build_profile_catalog is None or pd is None or torch is None,
    "experiment modules unavailable",
)
class ExperimentGuardTests(unittest.TestCase):
    def test_run_raises_not_implemented_for_streaming_mode(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(
                data=DataConfig(),
                model=ModelConfig(full_profile_name="full"),
                profiles=[ProfileConfig(name="full", dimension=4, normalize=True)],
                utility=UtilityConfig(metric="relative_score_dissimilarity"),
                optimization=OptimizationConfig(mode="streaming"),
                retrieval=RetrievalConfig(),
                execution=ExecutionConfig(output_dir=td, experiment_name="exp_streaming_guard", verbose=False),
                training=TrainingConfig(enabled=False),
            )
            runner = ExperimentRunner(cfg)
            with self.assertRaisesRegex(NotImplementedError, "Not implemented"):
                runner.run()

    def test_estimate_utilities_handles_empty_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(
                data=DataConfig(),
                model=ModelConfig(full_profile_name="full"),
                profiles=[
                    ProfileConfig(name="d2", dimension=2, normalize=True),
                    ProfileConfig(name="full", dimension=4, normalize=True),
                ],
                utility=UtilityConfig(metric="relative_score_dissimilarity"),
                optimization=OptimizationConfig(),
                retrieval=RetrievalConfig(),
                execution=ExecutionConfig(output_dir=td, experiment_name="exp_guard", verbose=False),
                training=TrainingConfig(enabled=False),
            )
            runner = ExperimentRunner(cfg)
            profiles = build_profile_catalog(cfg.profiles, cfg.execution.dtype)
            profile_by_name = {p.name: p for p in profiles}
            full_profile = profile_by_name["full"]

            topics = pd.DataFrame([{"qid": "q1", "query": "q"}])
            qrels = pd.DataFrame([{"qid": "q1", "docno": "outside"}])
            subset_docnos = ["d1"]
            subset_doc_embeddings = torch.randn(1, 4, dtype=torch.float16)
            full_query_embeddings = torch.randn(1, 4, dtype=torch.float16)

            utility_pairs_df, utility_table_df = runner._estimate_document_utilities_for_subset(
                adapter=_DummyAdapter(),
                profiles=profiles,
                full_profile=full_profile,
                topics=topics,
                qrels=qrels,
                subset_docnos=subset_docnos,
                subset_doc_embeddings=subset_doc_embeddings,
                full_query_embeddings=full_query_embeddings,
            )

        self.assertEqual(list(utility_pairs_df.columns), ["qid", "docno", "profile", "full_score", "reduced_score", "utility"])
        self.assertTrue(utility_pairs_df.empty)
        self.assertEqual(set(utility_table_df["docno"].tolist()), {"d1"})
        self.assertEqual(set(utility_table_df["profile"].tolist()), {"d2", "full"})


@unittest.skipIf(build_profile_catalog is None or ProfileConfig is None, "profile modules unavailable")
class ProfileCostTests(unittest.TestCase):
    def test_build_profile_catalog_marks_explicit_costs(self):
        profiles = build_profile_catalog(
            [
                ProfileConfig(name="derived", dimension=8, normalize=True),
                ProfileConfig(name="explicit", dimension=8, normalize=True, cost_bytes=99),
            ],
            dtype="float16",
        )
        derived = next(p for p in profiles if p.name == "derived")
        explicit = next(p for p in profiles if p.name == "explicit")
        self.assertEqual(derived.cost_bytes, 16)
        self.assertFalse(derived.cost_is_explicit)
        self.assertEqual(explicit.cost_bytes, 99)
        self.assertTrue(explicit.cost_is_explicit)


@unittest.skipIf(load_config is None or yaml is None, "yaml/config modules unavailable")
class ConfigValidationTests(unittest.TestCase):
    def _write_cfg(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        with tmp:
            yaml.safe_dump(payload, tmp, sort_keys=False)
        return Path(tmp.name)

    def test_deprecated_candidate_fields_are_mapped(self):
        path = self._write_cfg(
            {
                "data": {"candidate_source": "pyterrier_bm25", "candidates_per_query": 77},
                "model": {"full_profile_name": "full"},
                "profiles": [{"name": "full", "dimension": 8, "normalize": True}],
                "training": {"enabled": False},
            }
        )
        try:
            with self.assertWarns(Warning):
                cfg = load_config(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(cfg.retrieval.mode, "pyterrier_candidates")
        self.assertEqual(cfg.retrieval.candidate_k, 77)

    def test_invalid_retrieval_mode_raises(self):
        path = self._write_cfg(
            {
                "data": {},
                "model": {"full_profile_name": "full"},
                "profiles": [{"name": "full", "dimension": 8, "normalize": True}],
                "retrieval": {"mode": "not_supported"},
                "training": {"enabled": False},
            }
        )
        try:
            with self.assertRaises(ValueError):
                load_config(path)
        finally:
            path.unlink(missing_ok=True)

    def test_streaming_optimization_mode_is_accepted_by_loader(self):
        path = self._write_cfg(
            {
                "data": {},
                "model": {"full_profile_name": "full"},
                "profiles": [{"name": "full", "dimension": 8, "normalize": True}],
                "optimization": {"mode": "streaming"},
                "training": {"enabled": False},
            }
        )
        try:
            cfg = load_config(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(cfg.optimization.mode, "streaming")


if __name__ == "__main__":
    unittest.main()
