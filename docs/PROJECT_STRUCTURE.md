# Project Structure for Fast Comparative Experiments

## Current convention

- Core retrieval implementation: `src/retrieval/`
- Generation/evaluation: `src/generation/`, `src/evaluation/`
- Experiment matrix and batch scripts: `experiments/`

## Recommended experiment split

1. `baseline`: original PBR (`model_type=PBR`)
2. `innovation1-temporal`: temporal profile (`model_type=PBR++`, `--temporal_profile`)
3. `ablation`: disable one mechanism each time

## Suggested directory map

```text
src/
  retrieval/
    retrieval_PBR.py                # main retriever (baseline + temporal)
    eval_utils.py
  generation/
  evaluation/
experiments/
  configs/
    longmemeval_temporal_matrix.json
  run_retrieval_matrix.py
  summarize_retrieval_results.py
docs/
  PROJECT_STRUCTURE.md
```

## Naming rules for reproducibility

- Use unique `save_suffix` for each run.
- Keep one matrix json per dataset split.
- Do not change metrics scripts during ablation rounds.

## Next extension hooks

- Innovation 2 (cold-start prototype routing): add cohort/prototype loader and routing flags in `retrieval_PBR.py`.
- Innovation 3 (explicit user representation + contrastive learning): add offline encoder training under a separate module, then expose inference-time user embedding loading in retrieval.
