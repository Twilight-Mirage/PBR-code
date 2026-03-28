# Experiments Workspace

This directory is for fast A/B comparison on `src/retrieval/retrieval_PBR.py` and unified baseline orchestration.

## Structure

- `configs/`: experiment matrix files.
- `build_coldstart_prototype_bank.py`: build prototype-bank json for cold-start routing.
- `build_explicit_contrastive_pairs.py`: build triplets for contrastive user-representation training.
- `train_explicit_user_encoder.py`: train explicit-user projector from triplets.
- `run_retrieval_matrix.py`: batch runner for multiple retrieval settings.
- `run_baseline_matrix.py`: unified runner for local baselines and official third-party baselines.
- `summarize_retrieval_results.py`: aggregate metrics from multiple output JSON files.

Project-level defaults (API key/base URL/default LongMemEval file) are centralized in project_settings.py.
CLI args still have highest priority, then env vars, then project_settings.py.

## Quick Start

1. Build prototype bank (required for cold-start experiments):

```bash
python experiments/build_coldstart_prototype_bank.py \
  --input_json data/longmemeval_data/longmemeval_s_cleaned.json \
  --output_json data/longmemeval_data/prototype_bank_longmemeval_s_cleaned.json \
  --retrieval_model_name multi-qa-MiniLM-L6-cos-v1 \
  --num_clusters 8 \
  --label_keys department,role,team
```

2. Build contrastive triplets for explicit-user encoder:

```bash
python experiments/build_explicit_contrastive_pairs.py \
  --input_json data/longmemeval_data/longmemeval_s_cleaned.json \
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

## Unified Baselines

`run_baseline_matrix.py` unifies six baseline entry points:

- `naive_rag`: local retrieval + generation pipeline (`run_retrieval.py` + `run_generation.py`).
- `history_rag`: local history-only generation (`run_generation.py` with `orig-session` or `orig-turn`).
- `pbr_pipeline`: local PBR retrieval+generation pipeline (no temporal/cold-start/explicit modules).
- `dua_rag`: local DUA-RAG pipeline (temporal + cold-start + explicit enabled by default).
- `llm_gt_baseline`: independent LLM-GT baseline (same generator/prompt/token budget, evidence source switched to `retrieved` or `oracle`).
- `pgraphrag_official`: official `third_party_baselines/PGraphRAG/master_generation.py`.
- `afce_official`: official `third_party_baselines/AP-Bots` (`python -m AP_Bots.run_exp`).
- `lightrag_official`: official `third_party_baselines/LightRAG` script/command wrapper.

Sample matrices:

- `experiments/configs/baseline_unified_matrix.sample.json`
- `experiments/configs/longmemeval_llmgt_baseline.sample.json` (retrieved vs oracle evidence only)
- `experiments/configs/chapter4_baselines_partial_lamp.sample.json` (Chapter-4 baseline suite; LaMP-4/LaMP-7 only in partial baseline tracks)
- `experiments/configs/chapter4_main_table_partial_lamp.sample.json` (Chapter-4 main-table oriented unified matrix)
- `experiments/configs/chapter4_dua_main_partial_lamp.sample.json` (DUA main runs; LongMemEval full, PersonaBench partial)
Dry-run (recommended first):

```bash
<python_bin> experiments/run_baseline_matrix.py \
  --matrix experiments/configs/baseline_unified_matrix.sample.json \
  --dry_run
```

Run selected experiments only:

```bash
<python_bin> experiments/run_baseline_matrix.py \
  --matrix experiments/configs/baseline_unified_matrix.sample.json \
  --only naive_rag_local,history_rag_local
```

Notes for official baselines:

- `PGraphRAG` and `AP-Bots` use their own input formats and dataset conventions; do not assume they directly consume LongMemEval/PersonaBench json.
- `LightRAG` has no single benchmark CLI in upstream; matrix field `entry_script`/`command` is used as an adapter.
- `run_baseline_matrix.py` now checks required file/repo paths before execution.

## DUA End-to-End Matrix

Use `run_dua_e2e_matrix.py` to run retrieval -> generation -> evaluation in one pass for ablations.

Sample config:

- `experiments/configs/longmemeval_dua_e2e_matrix.sample.json`

Dry-run:

```bash
<python_bin> experiments/run_dua_e2e_matrix.py \
  --matrix experiments/configs/longmemeval_dua_e2e_matrix.sample.json \
  --dry_run
```

Run selected experiments:

```bash
<python_bin> experiments/run_dua_e2e_matrix.py \
  --matrix experiments/configs/longmemeval_dua_e2e_matrix.sample.json \
  --only pbr_base,pbr_temporal
```
## Readiness Check

Before launching long jobs, run preflight checks for missing files/keys:

```bash
<python_bin> experiments/check_experiment_readiness.py --kind retrieval --matrix experiments/configs/longmemeval_temporal_matrix.json
<python_bin> experiments/check_experiment_readiness.py --kind baseline --matrix experiments/configs/baseline_unified_matrix.sample.json
<python_bin> experiments/check_experiment_readiness.py --kind dua --matrix experiments/configs/longmemeval_dua_e2e_matrix.sample.json
```

The checker reports:

- `BLOCKER`: must be fixed before run.
- `WARN`: recommended to fix, but execution may still continue depending on endpoint setup.
## Notes

- Each experiment should define a unique `save_suffix` to avoid overwriting.
- `temporal_profile` maps to `--temporal_profile`.
- `cold_start_router` maps to `--cold_start_router`.
- `explicit_profile` maps to `--explicit_profile`.
- Trained explicit encoder is loaded with `--explicit_encoder_ckpt`.
- Most args can be put in `global` for shared defaults, and overridden in each experiment's `args`.
- Keep matrix files as the single source of truth for reproducible comparisons.



## Chapter 4 Alignment (Partial LaMP)

To align with thesis Chapter 4 while keeping current code-path constraints:

- **LongMemEval**: full pipeline (`retrieval + generation + eval`) is supported.
- **PersonaBench**: partial participation in unified local pipeline (usually generation-focused, eval optional).
- **LaMP-4 / LaMP-7**: participate only in **partial baseline tracks** (for example AF+CE official), not in oracle-evidence or LongMemEval-style retrieval ablations.

Recommended commands (including unified main-table run):

```bash
<python_bin> experiments/run_baseline_matrix.py \
  --matrix experiments/configs/chapter4_baselines_partial_lamp.sample.json \
  --dry_run

<python_bin> experiments/run_dua_e2e_matrix.py \
  --matrix experiments/configs/chapter4_dua_main_partial_lamp.sample.json \
  --dry_run

<python_bin> experiments/run_baseline_matrix.py \
  --matrix experiments/configs/chapter4_main_table_partial_lamp.sample.json \
  --dry_run
```

`run_dua_e2e_matrix.py` now supports experiment-level `args.in_file` and `args.ref_json`,
so each experiment can target different datasets within one matrix.

## Official Baseline Adapter (Unified Eval)

To align official baseline outputs with local evaluator input format (`question_id`, `hypothesis`), use:

```bash
<python_bin> -m experiments.adapt_official_baseline_output \
  --baseline afce \
  --pred_file <raw_pred_json_or_jsonl_or_csv> \
  --ref_file <reference_json_or_jsonl> \
  --out_file <unified_hypotheses.jsonl>
```

Then evaluate with the same evaluator:

```bash
<python_bin> -m src.evaluation.evaluate_qa gpt-4o-mini <unified_hypotheses.jsonl> <reference_json>
```

`run_baseline_matrix.py` now supports optional post-processing for official methods:

- Set `args.unified_eval=true` to append `adapt_to_unified` + `eval_unified` steps.
- For `lightrag_official`, `args.adapter_pred_file` is required.
- For `pgraphrag_official`, auto inference works when exactly one `mode` and one `k` are provided; otherwise set `args.adapter_pred_file`.
- For `afce_official`, when `adapter_pred_file` is omitted, adapter picks the latest matching file in `AP_Bots/files/preds` by dataset tag.

Additional sample matrix:

- `experiments/configs/official_adapter_eval.sample.json`
