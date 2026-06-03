# Experiment Outputs Dashboard

This folder contains a Jupyter notebook to discover, load, visualize, and compare experiment outputs.

## Files

- `experiment_outputs_dashboard.ipynb`: main notebook
- `exports/`: generated tables and plots (`<timestamp>` subfolders)

## Expected Output Location

The notebook scans:

- `../matryoshka_optimization_codebase/outputs`

and discovers run folders containing at least one known artifact:

- `assignments.parquet`
- `per_document_utility.parquet`
- `pt_experiment.csv`
- `summary.json`
- `memory_summary.json`
- `full_run.parquet`
- `optimized_run.parquet`
- `profile_catalog.csv`
- `metrics_long.csv`
- `metrics_wide.csv`
- `ranking_summary.csv`
- `run_metadata.json`

Folders under any `models` directory are excluded.

## How To Use

1. Open `experiment_outputs_dashboard.ipynb`.
2. Run cells top-to-bottom.
3. Inspect:
   - run manifest and schema profile
   - per-run plots
   - cross-run comparison plots/tables
4. Run export cells to save outputs in:
   - `analysis_viz/exports/<timestamp>/`

## Notes

- If no outputs are present yet, the notebook shows a warning and exits gracefully from plotting/export sections.
- Missing artifacts per run are handled without failing the whole notebook.
