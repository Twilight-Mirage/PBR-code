# Personalized RAG Experiment Structure

This repo is now organized so each innovation can be switched on/off with CLI flags and matrix configs.

## 1) Core Retrieval Entry

- `src/retrieval/retrieval_PBR.py`
  - Base PBR pipeline
  - Temporal extension (innovation-1)
  - Cold-start router extension (innovation-2)
  - Explicit user-feature + contrastive extension (innovation-3)

All comparison runs should use this single entrypoint to keep evaluation consistent.

## 2) Innovation Modules

- `src/retrieval/cold_start_router.py`
  - Supervised prototype routing (labels: department/role/team, etc.)
  - Unsupervised prototype routing (cluster centroids)
  - Adaptive mode switch by history size:
    - `cohort_only` (`m == 0`)
    - `blend` (`0 < m < m0`)
    - `individual` (`m >= m0`)

- `src/retrieval/explicit_profile_utils.py`
  - Explicit user feature extraction (style/keywords/phrases)
  - Explicit feature prompt block generation
  - Contrastive example selection (external negatives first, history fallback)

- `src/retrieval/explicit_user_encoder.py`
  - Trainable explicit-user projector adapter
  - Checkpoint save/load for inference-time feature projection

Keep each innovation in an isolated module and inject via config into `RAGRetriever`.

## 3) Reproducible Experiment Layer

- `experiments/run_retrieval_matrix.py`
  - Reads one matrix JSON and expands CLI commands.
- `experiments/configs/longmemeval_temporal_matrix.json`
  - Temporal ablations.
- `experiments/configs/longmemeval_coldstart_matrix.json`
  - Cold-start ablations (base/temporal/cold/hybrid).
- `experiments/configs/longmemeval_explicit_matrix.json`
  - Explicit-feature/contrastive ablations.
- `experiments/configs/longmemeval_explicit_trained_matrix.json`
  - Explicit-feature ablation with trained projector.
- `experiments/summarize_retrieval_results.py`
  - Aggregates output JSON into markdown table.

Use matrix files as the single source of truth for comparisons.

## 4) Data and Training Utilities

- `experiments/build_coldstart_prototype_bank.py`
  - Builds `prototype_bank.json` from dataset history.
  - Outputs:
    - `supervised` prototypes
    - `unsupervised.centroids`
    - `user_to_cluster`
    - `global_mean`

- `experiments/build_explicit_contrastive_pairs.py`
  - Builds contrastive triplets for training explicit user representations.
  - Output format: `anchor_profile_text`, `positive_text`, `negative_text`.

- `experiments/train_explicit_user_encoder.py`
  - Trains explicit-user projector with contrastive objectives.
  - Produces checkpoint consumed by `retrieval_PBR.py` via `--explicit_encoder_ckpt`.

Recommended output paths:

- `data/longmemeval_data/prototype_bank_longmemeval_s.json`
- `data/longmemeval_data/explicit_contrastive_pairs_s.jsonl`
- `data/longmemeval_data/explicit_user_encoder_s.pt`

## 5) Result Convention

`retrieval_PBR.py` writes output names using mode tags:

- `PBR`
- `PBR_temporal`
- `PBR_coldstart`
- `PBR_explicit`
- `PBR_temporal_coldstart`
- `PBR_temporal_explicit`
- `PBR_temporal_coldstart_explicit`

Plus your matrix `save_suffix`.

This makes summary scripts and downstream plotting easier.
