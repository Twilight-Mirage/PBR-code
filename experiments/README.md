# Experiments Workspace

This directory is for fast A/B comparison on `src/retrieval/retrieval_PBR.py`.

## Structure

- `configs/`: experiment matrix files.
- `build_coldstart_prototype_bank.py`: build prototype-bank json for cold-start routing.
- `build_explicit_contrastive_pairs.py`: build triplets for contrastive user-representation training.
- `train_explicit_user_encoder.py`: train explicit-user projector from triplets.
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

2. Build contrastive triplets for explicit-user encoder:

```bash
python experiments/build_explicit_contrastive_pairs.py \
  --input_json data/longmemeval_data/longmemeval_s.json \
  --output_jsonl data/longmemeval_data/explicit_contrastive_pairs_s.jsonl \
  --retrieval_model_name multi-qa-MiniLM-L6-cos-v1 \
  --max_pairs_per_user 12
```

3. Train explicit-user encoder (third innovation point):

```bash
python experiments/train_explicit_user_encoder.py \
  --triplets_jsonl data/longmemeval_data/explicit_contrastive_pairs_s.jsonl \
  --output_ckpt data/longmemeval_data/explicit_user_encoder_s.pt \
  --retrieval_model_name multi-qa-MiniLM-L6-cos-v1 \
  --epochs 8 \
  --batch_size 64 \
  --device cpu
```

4. Run temporal matrix:

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_temporal_matrix.json
```

5. Run cold-start matrix:

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_coldstart_matrix.json
```

6. Run explicit-feature matrix (without trained projector):

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_explicit_matrix.json
```

7. Run explicit-feature matrix (with trained projector):

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_explicit_trained_matrix.json
```

8. Summarize outputs:

```bash
python experiments/summarize_retrieval_results.py --glob "data/longmemeval_data/*_PBR*.json"
```

## Notes

- Each experiment should define a unique `save_suffix` to avoid overwriting.
- `temporal_profile` maps to `--temporal_profile`.
- `cold_start_router` maps to `--cold_start_router`.
- `explicit_profile` maps to `--explicit_profile`.
- Trained explicit encoder is loaded with `--explicit_encoder_ckpt`.
- Most args can be put in `global` for shared defaults, and overridden in each experiment's `args`.
- Keep matrix files as the single source of truth for reproducible comparisons.
