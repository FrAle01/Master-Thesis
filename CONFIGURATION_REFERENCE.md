# Configuration Reference (Code-Derived)

This document lists all configuration elements available in the codebase and groups them by the same logical sections used by the implementation.

Scope covered:
- Main experiment config: `matryoshka_exp` (`src/matryoshka_exp/config.py`)
- Side quest retrieval benchmark config: `side_quests.retrieval_benchmark` (`src/side_quests/retrieval_benchmark/config.py`)

## 1) Main Experiment Configuration (`matryoshka_exp`)

Required top-level sections:
- `data`
- `model`
- `profiles`

Optional top-level sections (have defaults):
- `utility`
- `optimization`
- `retrieval`
- `execution`
- `training`

### 1.1 `data` (DataConfig)
- `pyterrier_dataset` (str|null)
- `dataset_provider` (str|null)
- `topics_variant` (str|null)
- `qrels_variant` (str|null)
- `eval_pyterrier_dataset` (str|null)
- `eval_dataset_provider` (str|null)
- `eval_topics_variant` (str|null)
- `eval_qrels_variant` (str|null)
- `text_fields` (list[str])
- `max_docs` (int|null)
- `max_queries` (int|null)
- `topic_column` (str)
- `docno_column` (str)
- `local_corpus_path` (str|null)
- `local_topics_path` (str|null)
- `local_qrels_path` (str|null)
- `local_eval_topics_path` (str|null)
- `local_eval_qrels_path` (str|null)
- `single_log_policy` (str)
  - Allowed: `shared`, `split_by_qrels`
- `qrels_label_column` (str)
- `training_dataset_name` (str|null)
- `training_local_path` (str|null)
- `training_split` (str)
- `training_format` (str)
  - Allowed: `hf_triplet`, `hf_tevatron_passage`, `jsonl_triplet`
- `query_column` (str)
- `positive_column` (str)
- `negative_column` (str|null)
- `tevatron_positive_passages_column` (str)
- `tevatron_negative_passages_column` (str)
- `tevatron_passage_text_field` (str)
- `candidate_source` (str, deprecated)
  - Deprecated supported values: `pyterrier_bm25`, `dense_exact`
  - Mapping behavior:
    - `pyterrier_bm25` -> `retrieval.mode = pyterrier_candidates`
    - `dense_exact` -> `retrieval.mode = dense_exact`
- `candidates_per_query` (int, deprecated)
  - Deprecated mapping: `retrieval.candidate_k`
- `local_terrier_index_path` (str|null)
- `build_local_terrier_index_if_missing` (bool)
- `terrier_index_threads` (int)
- `terrier_index_overwrite` (bool)
- `terrier_meta_lengths` (dict[str,int])
- `full_embeddings_source` (str)
  - Allowed: `auto`, `hf_dataset`, `compute`
- `hf_embeddings_repo_id` (str|null)
- `hf_embeddings_split` (str)
- `hf_embeddings_docno_column` (str)
- `hf_embeddings_vector_column` (str)

Validation/logic constraints:
- If `pyterrier_dataset` is null, all are required:
  - `local_corpus_path`, `local_topics_path`, `local_qrels_path`
- If `full_embeddings_source` is `auto` or `hf_dataset`, `hf_embeddings_repo_id` is required.
- If `training.enabled=true`:
  - `training_format=jsonl_triplet` requires `training_local_path`.
  - `training_format` in `hf_triplet|hf_tevatron_passage` requires `training_dataset_name`.

### 1.2 `model` (ModelConfig)
- `mode` (str)
- `backend` (str)
  - Allowed: `sentence_transformers`, `transformers`
- `model_name_or_path` (str)
- `tokenizer_name_or_path` (str|null)
- `adapter_type` (str|null)
- `adapter_path` (str|null)
- `adapter_name` (str)
- `trust_remote_code` (bool)
- `query_prompt` (str)
- `document_prompt` (str)
- `similarity` (str)
  - Allowed: `dot`, `cosine`
- `full_profile_name` (str)
  - Must match one `profiles[].name`
- `pooling` (str)
- `sentence_transformer_truncate_dim` (int|null)

### 1.3 `profiles` (list[ProfileConfig])
Each item:
- `name` (str)
- `dimension` (int)
  - Must be `> 0`
- `layer` (int|null)
- `normalize` (bool)
- `cost_bytes` (int|null)
- `cost_is_explicit` (bool)
  - If true, `cost_bytes` must be provided and `> 0`

Validation/logic constraints:
- At least one profile required.
- `model.full_profile_name` must exist in this list.

### 1.4 `utility` (UtilityConfig)
- `metric` (str)
  - Allowed:
    - `relative_score_dissimilarity`
    - `absolute_score_utility`
    - `squared_score_utility`
    - `relative_margin_utility`
    - `hybrid_score_margin_utility`
- `epsilon` (float)
- `alpha` (float)
- `margin_negatives` (int)
  - Must be `> 0`
- `aggregate` (str)
  - Allowed: `mean`
- `sample_pairs_per_query` (int)
  - Allowed range: `-1` (unlimited) or any `> 0`
- `relevance_threshold` (float)
- `seed` (int)
- `default_utility_profile_name` (str|null)
  - If set, must match one `profiles[].name`
  - Controls default/fallback utility assignment for documents without utility evidence:
    - selected profile gets utility `1.0`
    - all other profiles get utility `0.0`
  - If null, fallback defaults to `model.full_profile_name`

Nested `utility.relevance`:
- `mode` (str)
  - Allowed: `weak`, `model`, `hybrid`
- `weak_source` (str)
  - Allowed: `bm25`, `dense`, `hybrid_rerank`
- `calibration` (str)
  - Allowed: `minmax_score`, `rank_log_discount`, `constant_one`
- `cross_encoder_model_name` (str)
- `top_k_candidates` (int)
  - Must be `> 0`
- `rerank_k` (int)
  - Must be `> 0`
- `uncertainty` (str)
  - Allowed: `margin`
- `qrel_adjustment` (str)
  - Allowed: `hard_override`, `relevant_only_to_one`
- `high_impact_fraction` (float)
  - Must be in `(0,1]`
- `validation_qids_limit` (int)
  - Must be `> 0`

Nested `utility.relevance.scale`:
- `mode` (str)
  - Allowed: `minmax`, `explicit_map`
- `min_label` (float)
- `max_label` (float)
  - Must be `> min_label`
- `explicit_map` (dict[str,float])
  - Required when `mode=explicit_map`

Nested `utility.tail`:
- `mode` (str)
  - Allowed: `static_default`, `rbp_residual_interval`
  - Default: `static_default`
- `target_residual` (float)
  - Must be in `[0,1]`
  - Default: `0.01`
- `conservatism_beta` (float|null)
  - Must be null or `>= 0`
  - Default: `0.5`
- `expected_fraction` (float)
  - Must be in `[0,1]`
  - Default: `0.5`
- `max_tail_relevance` (float)
  - Must be `>= 0`
  - Default: `0.01`
- `profile_quality` (str)
  - Allowed: `cosine_padding`, `sampled_dense_score_preservation`
  - Default: `cosine_padding`
- `sampled_quality_queries` (int)
  - Must be `> 0`
  - Default: `32`
- `utility_low` (float)
  - Default: `0.0`

`utility.tail` controls the fallback utility for documents that never appear in any relevance-estimation ranking. These are documents in:

```text
D_unranked = D_all \ D_ranked
```

where `D_ranked` is calculated from `pair_relevance_df.docno.unique()` after BM25/dense/hybrid/model relevance construction.

With `mode=static_default`, the previous behavior is preserved:

```text
utility(d, profile) = 1.0 if profile == utility.default_utility_profile_name else 0.0
```

With `mode=rbp_residual_interval`, the fallback uses an RBP-inspired residual interval. It does not use raw `p^K`, because for large candidate depths such as `K=2000` or `K=20000`, ordinary RBP residuals collapse near zero. Instead, `target_residual` is the explicit residual mass reserved for the unseen tail:

```text
R = utility.tail.target_residual
expected_mass = R * utility.tail.expected_fraction
uncertainty_mass = R - expected_mass
```

The residual is distributed over unranked documents:

```text
expected_relevance_per_doc =
    min(utility.tail.max_tail_relevance, expected_mass / |D_unranked|)

uncertainty_per_doc =
    uncertainty_mass / |D_unranked|
```

The final utility for an unranked document/profile pair is profile-dependent:

```text
expected_utility(d,m) =
    expected_relevance_per_doc * profile_quality(d,m)

uncertainty_utility(d,m) =
    uncertainty_per_doc * profile_quality(d,m)
```

If `conservatism_beta` is set:

```text
utility(d,m) =
    max(0, expected_utility(d,m) - conservatism_beta * uncertainty_utility(d,m))
```

If `conservatism_beta: null`:

```text
utility(d,m) = utility.tail.utility_low
```

`profile_quality=cosine_padding` computes:

```text
profile_quality(d,m) =
    cosine(e_full(d), pad(truncate(e_full(d), m)))
```

This is cheap, deterministic, and recommended for large runs.

`profile_quality=sampled_dense_score_preservation` samples up to `sampled_quality_queries` query embeddings using `utility.seed`, computes full and reduced dense scores for unranked documents, and reuses the configured `utility.metric` through the internal score-preservation function:

```text
profile_quality(d,m) =
    mean_q preservation(score_full(q,d), score_reduced(q,d,m))
```

This is more retrieval-faithful but can be substantially slower.

Recommended `utility.tail` configurations:

Static baseline / backwards-compatible:

```yaml
utility:
  tail:
    mode: "static_default"
```

Balanced RBP residual fallback for `top_k_candidates=2000` or `20000`:

```yaml
utility:
  tail:
    mode: "rbp_residual_interval"
    target_residual: 0.01
    expected_fraction: 0.5
    conservatism_beta: 0.5
    max_tail_relevance: 0.01
    profile_quality: "cosine_padding"
    sampled_quality_queries: 32
    utility_low: 0.0
```

Very conservative tail fallback:

```yaml
utility:
  tail:
    mode: "rbp_residual_interval"
    target_residual: 0.001
    expected_fraction: 0.5
    conservatism_beta: 0.5
    max_tail_relevance: 0.005
    profile_quality: "cosine_padding"
```

Lower-bound-only fallback:

```yaml
utility:
  tail:
    mode: "rbp_residual_interval"
    target_residual: 0.01
    expected_fraction: 0.5
    conservatism_beta: null
    utility_low: 0.0
```

More retrieval-faithful but slower fallback:

```yaml
utility:
  tail:
    mode: "rbp_residual_interval"
    target_residual: 0.01
    expected_fraction: 0.5
    conservatism_beta: 0.5
    max_tail_relevance: 0.01
    profile_quality: "sampled_dense_score_preservation"
    sampled_quality_queries: 32
```

Important tuning relation:

```text
expected_fraction > conservatism_beta * (1 - expected_fraction)
```

Equivalently:

```text
expected_fraction > conservatism_beta / (1 + conservatism_beta)
```

If this is not satisfied, the beta-calibrated utility is clipped to zero for all tail documents. For example, with `conservatism_beta=0.5`, `expected_fraction` should be greater than about `0.333` unless the goal is intentionally zero tail utility.

For the current deep-candidate setup (`top_k_candidates=20000`, `retrieval.candidate_k=20000`), start with:

```yaml
target_residual: 0.01
expected_fraction: 0.5
conservatism_beta: 0.5
max_tail_relevance: 0.01
profile_quality: "cosine_padding"
```

Then run ablations with `target_residual: 0.001` and `conservatism_beta: null`.

### 1.5 `optimization` (OptimizationConfig)
- `budget_bytes` (int|null)
- `budget_gb` (float|null)
- `algorithm` (str)
- `max_iter` (int)
- `tolerance` (float)
- `lambda_low` (float)
- `lambda_high` (float)
- `streaming_reestimate_every_docs` (int)
- `mode` (str)
  - Allowed: `batch`, `streaming`
  - Runtime status: `streaming` is currently disabled (validation raises error)

Validation/logic constraints:
- At least one of `budget_bytes` or `budget_gb` must resolve to a budget.

### 1.6 `retrieval` (RetrievalConfig)
- `mode` (str)
  - Allowed: `dense_exact`, `pyterrier_candidates`
- `top_k` (int)
  - Must be `> 0`
- `candidate_k` (int)
  - Must be `> 0`
- `include_bm25_baseline` (bool)
- `batch_size_queries` (int)
- `batch_size_docs` (int)
- `use_pyterrier_bm25` (bool)
- `terrier_wmodel` (str)
- `use_torch_gpu_exact` (bool)

### 1.7 `execution` (ExecutionConfig)
- `output_dir` (str)
- `experiment_name` (str)
- `device` (str)
- `retrieval_device` (str)
  - Allowed: `cuda`, `cpu`
- `dtype` (str)
- `doc_batch_size` (int)
- `query_batch_size` (int)
- `num_workers` (int)
- `seed` (int)
- `save_embeddings` (bool)
- `save_score_pairs` (bool)
- `save_runs` (bool)
- `verbose` (bool)
- `retrieval_vram_utilization_limit` (float)
- `retrieval_vram_temp_overhead_factor` (float)
  - Must be `>= 1.0`
- `embedding_checkpoint_enabled` (bool)
- `embedding_checkpoint_every_docs` (int)
  - Must be `> 0`
- `embedding_checkpoint_dir` (str|null)
- `embedding_checkpoint_resume` (str)
  - Allowed: `auto`, `restart`, `fail`

### 1.8 `training` (TrainingConfig)
- `enabled` (bool)
- `finetune_strategy` (str)
  - Allowed: `full`, `lora`
- `output_model_dir` (str)
- `base_loss` (str)
- `use_matryoshka` (bool)
- `use_2d_matryoshka` (bool)
- `matryoshka_dimensions` (list[int])
- `n_layers_per_step` (int)
- `epochs` (int)
- `per_device_train_batch_size` (int)
- `gradient_accumulation_steps` (int)
- `learning_rate` (float)
- `warmup_ratio` (float)
- `weight_decay` (float)
- `max_steps` (int)
- `fp16` (bool)
- `bf16` (bool)
- `save_steps` (int)
- `eval_steps` (int)
- `logging_steps` (int)
- `resume_from_checkpoint` (str|null)
- `auto_resume_from_last_checkpoint` (bool)

Nested `training.lora`:
- `enabled` (bool)
- `r` (int)
- `alpha` (int)
- `dropout` (float)
- `bias` (str)
- `task_type` (str)
- `target_modules` (list[str])
- `exclude_modules` (list[str])
- `modules_to_save` (list[str])
- `adapter_name` (str)
- `output_adapter_dir` (str)

Nested `training.hub`:
- `enabled` (bool)
- `repo_prefix` (str)
- `private` (bool)
- `token_env` (str)
- `auto_repo_from_experiment` (bool)
  - Currently only `true` is supported
- `repo_suffix_full` (str)
- `repo_suffix_lora` (str)

Validation/logic constraints:
- If `training.hub.enabled=true`, `repo_prefix` is required.
- If `training.hub.enabled=true`, `auto_repo_from_experiment` must be `true`.

## 2) Side Quest Retrieval Benchmark Configuration (`side_quests.retrieval_benchmark`)

Required top-level sections:
- `dataset`
- `models`
- `dimensions`

Optional top-level sections:
- `retrieval`
- `evaluation`
- `output`
- `embedding_cache`
- `seed`
- `device`

### 2.1 `dataset` (DatasetConfig)
- `pyterrier_dataset` (str|null)
- `topics_variant` (str|null)
- `qrels_variant` (str|null)
- `text_fields` (list[str])
- `topic_column` (str)
- `docno_column` (str)
- `max_docs` (int|null)
- `max_queries` (int|null)
- `local_corpus_path` (str|null)
- `local_topics_path` (str|null)
- `local_qrels_path` (str|null)

Validation/logic constraints:
- If `pyterrier_dataset` is null, local paths required:
  - `local_corpus_path`, `local_topics_path`, `local_qrels_path`

### 2.2 `models` (list[ModelEntry])
Each item:
- `id` (str)
  - Must be unique
- `type` (str)
  - Allowed: `hf_model`, `local_model`, `lora_adapter`
- `path_or_repo` (str)
- `base_model` (str|null)
  - Required when `type=lora_adapter`
- `adapter_name` (str)
- `trust_remote_code` (bool)
- `query_prompt` (str)
- `document_prompt` (str)
- `normalize` (bool)

### 2.3 `dimensions` (list[int])
- Must contain at least one value
- All values must be `> 0`

### 2.4 `retrieval` (RetrievalConfig)
- `top_k` (int)
  - Must be `> 0`
- `batch_size_queries` (int)
- `batch_size_docs` (int)
- `similarity` (str)
  - Allowed: `dot`, `cosine`
- `mode` (str)
  - Allowed: `pyterrier_dr_faiss`, `dense_exact`, `bm25_rerank`
- `candidate_k` (int)
  - Must be `> 0`
- `terrier_wmodel` (str)
- `faiss_enabled` (bool)
- `faiss_use_gpu` (bool)
- `fallback_to_exact_on_backend_error` (bool)
- `faiss_backend` (str)
  - Allowed: `flat`, `hnsw`, `ivf`
- `require_gpu` (bool)

### 2.5 `evaluation` (EvaluationConfig)
- `metrics` (list[str])

### 2.6 `output` (OutputConfig)
- `output_dir` (str)
- `run_name` (str)

### 2.7 `embedding_cache` (EmbeddingCacheConfig)
- `enabled` (bool)
- `cache_dir` (str|null)
- `checkpoint_every_batches` (int)
  - Must be `> 0`
- `reuse_if_available` (bool)

### 2.8 Global fields
- `seed` (int)
- `device` (str)

## 3) CLI Interface (How config is selected)

Main experiment CLI:
- `python -m matryoshka_exp.cli train --config <yaml_path>`
- `python -m matryoshka_exp.cli run --config <yaml_path>`

Side quest benchmark CLI:
- `python -m side_quests.retrieval_benchmark.cli run --config <yaml_path>`

## 4) Notes on "possible choices"

This file distinguishes:
- Explicit choices: values enforced by validation (`_validate_choice`, set-membership checks).
- Open string fields: no hard validation in config layer (choices are implementation-dependent and may still fail later if unsupported by libraries/backends).

## 5) Source of truth

Primary sources used:
- `matryoshka_optimization_codebase/src/matryoshka_exp/config.py`
- `matryoshka_optimization_codebase/src/matryoshka_exp/cli.py`
- `matryoshka_optimization_codebase/src/side_quests/retrieval_benchmark/config.py`
- `matryoshka_optimization_codebase/src/side_quests/retrieval_benchmark/cli.py`
- `matryoshka_optimization_codebase/configs/*.yaml`
- `matryoshka_optimization_codebase/configs/side_quests/*.yaml`
- `matryoshka_optimization_codebase/src/side_quests/retrieval_benchmark/README.md`
