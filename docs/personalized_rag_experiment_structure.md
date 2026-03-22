# Personalized RAG Experiment Structure

This repo is now organized so each innovation can be switched on/off with CLI flags and matrix configs.

## 1) Core Retrieval Entry

- `src/retrieval/retrieval_PBR.py`
  - Base PBR pipeline
  - Temporal extension (innovation-1)
  - Cold-start router extension (innovation-2)

All comparison runs should use this single entrypoint to keep evaluation consistent.

## 2) Innovation Modules

- `src/retrieval/cold_start_router.py`
  - Supervised prototype routing (labels: department/role/team, etc.)
  - Unsupervised prototype routing (cluster centroids)
  - Adaptive mode switch by history size:
    - `cohort_only` (`m == 0`)
    - `blend` (`0 < m < m0`)
    - `individual` (`m >= m0`)

Keep each new innovation in an isolated module and inject via config into `RAGRetriever`.

## 3) Reproducible Experiment Layer

- `experiments/run_retrieval_matrix.py`
  - Reads one matrix JSON and expands CLI commands.
- `experiments/configs/longmemeval_temporal_matrix.json`
  - Temporal ablations.
- `experiments/configs/longmemeval_coldstart_matrix.json`
  - Cold-start ablations (base/temporal/cold/hybrid).
- `experiments/summarize_retrieval_results.py`
  - Aggregates output JSON into markdown table.

Use matrix files as the single source of truth for comparisons.

## 4) Prototype Bank Build Step

- `experiments/build_coldstart_prototype_bank.py`
  - Builds `prototype_bank.json` from dataset history.
  - Outputs:
    - `supervised` prototypes
    - `unsupervised.centroids`
    - `user_to_cluster`
    - `global_mean`

Recommended output path:

- `data/longmemeval_data/prototype_bank_longmemeval_s.json`

## 5) Result Convention

`retrieval_PBR.py` writes output names using mode tags:

- `PBR`
- `PBR_temporal`
- `PBR_coldstart`
- `PBR_temporal_coldstart`

Plus your matrix `save_suffix`.

This makes summary scripts and downstream plotting easier.
