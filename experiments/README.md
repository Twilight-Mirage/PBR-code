# Experiments Workspace

This directory is for fast A/B comparison on `src/retrieval/retrieval_PBR.py`.

## Structure

- `configs/`: experiment matrix files.
- `build_coldstart_prototype_bank.py`: build prototype-bank json for cold-start routing.
- `run_retrieval_matrix.py`: batch runner for multiple retrieval settings.
- `summarize_retrieval_results.py`: aggregate metrics from multiple output JSON files.

## Quick Start

1. Build prototype bank (required for cold-start experiments):

```bash
python experiments/build_coldstart_prototype_bank.py \
  --input_json data/longmemeval_data/longmemeval_s.json \
  --output_json data/longmemeval_data/prototype_bank_longmemeval_s.json \
  --retrieval_model_name multi-qa-MiniLM-L6-cos-v1 \
  --num_clusters 8 \
  --label_keys department,role,team
```

2. Run temporal matrix:

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_temporal_matrix.json
```

3. Run cold-start matrix:

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_coldstart_matrix.json
```

4. Summarize outputs:

```bash
python experiments/summarize_retrieval_results.py --glob "data/longmemeval_data/*_PBR*.json"
```

## Notes

- Each experiment should define a unique `save_suffix` to avoid overwriting.
- `temporal_profile` maps to `--temporal_profile`.
- `cold_start_router` maps to `--cold_start_router`.
- Most args can be put in `global` for shared defaults, and overridden in each experiment's `args`.
- Keep matrix files as the single source of truth for reproducible comparisons.
