# Matryoshka GPU Retrieval Experiment

This codebase implements the experiment discussed in the thesis project:

- encode documents with **Matryoshka-style embeddings**;
- support either an **existing checkpoint** (e.g. `ielabgroup/Starbucks-msmarco`) or **fine-tuning** a model on a user-selected dataset;
- estimate the utility of reduced representations with configurable score-preservation metrics;
- solve a **document-wise memory-constrained optimization problem** using a Lagrangian relaxation;
- compare retrieval effectiveness against the **full-embedding baseline**;
- save all intermediate and final artefacts for later plotting and analysis.

The code is designed to be modular and readable. The environment in this sandbox does **not** include PyTerrier / Sentence Transformers / Transformers / FAISS, so the code was written to be executed on your own GPU VM, not run here.

## Main features

- **Model options**
  - direct use of existing Matryoshka-like models through `SentenceTransformer`
  - direct use of `ielabgroup/Starbucks-msmarco` through a hidden-state / layer-aware adapter
  - optional fine-tuning of a base embedding model with `MatryoshkaLoss` or `Matryoshka2dLoss`
  - optional LoRA adapter fine-tuning for `sentence_transformers` backends (base model + adapter at inference)
- **Utility metrics**
  - query-log utility estimation (default)
  - query-independent residual-norm utility estimation
  - relative score dissimilarity (default)
  - absolute score error
  - squared score error
  - relative margin preservation
  - hybrid score + margin utility
- **Execution modes**
  - batch corpus processing
- **Retrieval modes**
  - exact dense retrieval over the embedded corpus
  - PyTerrier candidate generation + dense re-scoring
- **Persistence**
  - profile catalogue
  - sampled score pairs
  - per-document utility table
  - optimized assignments
  - full and optimized run files
  - PyTerrier evaluation output
  - JSON summaries for memory and optimization statistics
  - optional full-document embedding cache on a dedicated Hugging Face dataset (load + fallback compute + push)

## Suggested environment

Python 3.10+
CUDA GPU with enough VRAM for the chosen encoder

Install dependencies:

```bash
pip install -r requirements.txt
```

### CUDA compatibility for this VM (driver 535 / CUDA 12.2)

This project is pinned for a **CUDA 12.1 PyTorch wheel set (`cu121`)** to match VMs with NVIDIA driver `535.x`.

Recommended reproducible setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

Quick verification before running experiments:

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); x=torch.empty(1, device='cuda'); print('device', x.device)"
```

Expected: `torch.version.cuda` reports `12.1` and `cuda_available` is `True`.

## Structure

```text
configs/
src/matryoshka_exp/
  cli.py
  config.py
  logging_utils.py
  experiment.py
  data/
  models/
  training/
  metrics/
  optimization/
  retrieval/
  results/
```

## Typical workflows

### 1) Use Starbucks-msmarco directly

```bash
python -m matryoshka_exp.cli run --config configs/starbucks_msmarco_batch.yaml
```

### 2) Fine-tune a base embedding model first

```bash
python -m matryoshka_exp.cli train --config configs/finetune_nomic_msmarco_batch.yaml
python -m matryoshka_exp.cli run --config configs/finetune_nomic_msmarco_batch.yaml
```

### 3) Fine-tune with LoRA adapters

```bash
python -m matryoshka_exp.cli train --config configs/finetune_nomic_lora_msmarco_batch.yaml
python -m matryoshka_exp.cli run --config configs/finetune_nomic_lora_msmarco_batch.yaml
```

LoRA is controlled through:

- `training.finetune_strategy: lora`
- `training.lora.*` for adapter hyperparameters and save path
- `model.adapter_type/adapter_path/adapter_name` for inference-time loading (automatically injected when running `train` + `run` in one execution)

## Notes about PyTerrier integration

PyTerrier is used **when possible** for:

- dataset access via `pt.get_dataset(...)`
- corpus iteration via `dataset.get_corpus_iter()`
- candidate generation with BM25 if a Terrier index is available
- evaluation with `pt.Experiment(...)` or `pt.Evaluate(...)`

When exact dense search is selected, dense retrieval is executed directly with grouped profile scoring, because variable representation sizes are not naturally represented by a standard PyTerrier transformer.

## Important implementation choices

### 0) Full embedding cache on Hugging Face

`data.full_embeddings_source` controls how full document embeddings are resolved:

- `auto` (default): try load from Hugging Face, otherwise compute locally and push to Hugging Face
- `hf_dataset`: try load from Hugging Face first; on missing/incomplete data, fallback to compute and push
- `compute`: compute locally and then push to Hugging Face

Relevant config keys:

- `data.hf_embeddings_repo_id`
- `data.hf_embeddings_split`
- `data.hf_embeddings_docno_column`
- `data.hf_embeddings_vector_column`

`data.hf_embeddings_repo_id` is treated as a base repo name. The runtime derives a
model-specific dataset id by appending a deterministic signature of the active model
(and adapter, when present), so each model writes/reads from its own embedding dataset.

Authentication is read from `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.

### 1) Representation profiles

The code does not assume that profiles differ only by dimension.
Each profile can define:

- `dimension`
- optional `layer`
- `normalize`
- optional `cost_bytes`

This allows the same optimization code to work with:

- standard Matryoshka models (dimension only)
- Starbucks / 2D Matryoshka style models (layer + dimension)

### 2) Relaxed optimization

The optimizer applies a Lagrangian relaxation to:

```text
maximize   sum_i sum_k x_{ik} u_{ik}
subject to sum_i sum_k x_{ik} c_k <= B
           sum_k x_{ik} = 1
           x_{ik} in [0,1]
```

For a fixed lambda, the relaxed problem decomposes and each document independently selects the profile maximizing:

```text
u_{ik} - lambda * c_k
```

The multiplier search produces candidate assignments for the relaxed problem. A deterministic repair and upgrade phase then constructs and improves a feasible assignment for the original MCKP.

### 3) Utility estimation

`utility.estimator` selects one of two independent utility strategies.

#### Query-log estimator

With `utility.estimator: query_log` (the default), the utility table is estimated from sampled `(query, document)` pairs:

- full score `s(d,q)` is computed using the full profile
- reduced score `s_r(d,q)` is computed using the candidate profile
- the configured metric maps the difference into a utility in `[0,1]`

The default metric is the robust version of **relative score dissimilarity**:

```text
1 - min(1, |s - s_r| / max(|s|, eps))
```

#### Residual-norm estimator

With `utility.estimator: residual_norm`, optimization does not load or encode an optimization query log. It uses the evaluation corpus embeddings directly:

```text
utility(d, m) =
    clamp(1 - norm(d[m:]) / max(norm(d), epsilon), 0, 1)
```

- `utility.residual_norm.norm` selects the L1 (`1`) or L2 (`2`) norm.
- `utility.residual_norm.epsilon` protects the denominator.
- zero document vectors and the full profile receive utility `1.0`.
- profile dimensions are interpreted as prefix truncations of the full document embedding.

Residual mode uses `data.eval_pyterrier_dataset` as its dataset source, or the local evaluation inputs `local_eval_corpus_path`, `local_eval_topics_path`, and `local_eval_qrels_path`. Primary optimization dataset fields are optional in this mode.

The generic estimator report is written to `utility_estimation_report.json`. Query-log mode also retains `relevance_estimation_report.json`; residual mode does not create sampled score-pair output.

## Outputs

Every run creates a folder like:

```text
outputs/<experiment_name>/
  config.snapshot.yaml
  profile_catalog.csv
  sampled_score_pairs.parquet
  per_document_utility.parquet
  assignments.parquet
  full_run.parquet
  optimized_run.parquet
  pt_experiment.csv
  summary.json
  memory_summary.json
```

## Recommended starting configurations

- `configs/starbucks_msmarco_batch.yaml`
- `configs/finetune_nomic_msmarco_batch.yaml`
- `configs/finetune_nomic_lora_msmarco_batch.yaml`
- `configs/starbucks_msmarco_residual_norm_eval.yaml`
