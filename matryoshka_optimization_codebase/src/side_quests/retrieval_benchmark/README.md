# Retrieval Benchmark Side Quest

Run:

`python -m side_quests.retrieval_benchmark.cli run --config configs/side_quests/retrieval_benchmark_modernbert.yaml`

Notes:
- Uses PyTerrier for dataset loading and metric evaluation.
- Matryoshka flow: embeddings are computed once at the maximum requested dimension, then smaller dimensions are obtained by vector truncation (`[:dim]`).
- Retrieval modes:
  - `pyterrier_dr_faiss` (default): `pyterrier_dr`-gated dense retrieval with FAISS indexing/search.
  - `dense_exact`: exact dense scoring.
  - `bm25_rerank`: PyTerrier BM25 candidates reranked by dense model.
- FAISS controls:
  - `faiss_use_gpu: true` enables GPU retrieval when supported by your FAISS build.
  - `faiss_backend`: `flat`, `hnsw`, or `ivf`.
  - `require_gpu: true` enforces GPU-only retrieval.
  - `GpuIndexFlatIP` is built directly on GPU for the direct FAISS path (no CPU index handoff).
  - If backend setup fails and `fallback_to_exact_on_backend_error: false`, the run fails fast.
- Embedding cache/checkpoints:
  - `embedding_cache.reuse_if_available: true` reuses saved full embeddings.
  - `embedding_cache.checkpoint_every_batches` writes resumable partial checkpoints during long encoding.
