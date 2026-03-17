# Experiments Workspace

This directory is for fast A/B comparison on `src/retrieval/retrieval_PBR.py`.

## Structure

- `configs/`: experiment matrix files.
- `run_retrieval_matrix.py`: batch runner for multiple retrieval settings.
- `summarize_retrieval_results.py`: aggregate metrics from multiple output JSON files.

## Quick Start

1. Run all experiments in a matrix:

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_temporal_matrix.json
```

2. Summarize outputs:

```bash
python experiments/summarize_retrieval_results.py --glob "data/longmemeval_data/*_PBR*.json"
```

## Notes

- Each experiment should define a unique `save_suffix` to avoid overwriting.
- `temporal_profile` maps to `--temporal_profile` in `retrieval_PBR.py`.
- Keep the matrix file as the single source of truth for reproducible comparisons.
