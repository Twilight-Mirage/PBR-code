import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_ID_CANDIDATES = {
    "afce": ["id", "question_id", "user_id"],
    "pgraphrag": ["question_id", "id", "user_id"],
    "lightrag": ["question_id", "id", "qid", "sample_id"],
    "generic": ["question_id", "id", "qid", "sample_id", "user_id"],
}

DEFAULT_HYP_CANDIDATES = {
    "afce": ["output", "hypothesis", "answer", "prediction", "response", "text"],
    "pgraphrag": ["output", "hypothesis", "answer", "prediction", "response", "text"],
    "lightrag": ["hypothesis", "answer", "prediction", "response", "output", "text"],
    "generic": ["hypothesis", "answer", "prediction", "response", "output", "text"],
}


def _load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_any(path: Path) -> Any:
    ext = path.suffix.lower()
    if ext in {".json", ".jsonl"}:
        return _load_json_or_jsonl(path)
    if ext == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported prediction file extension: {path}")


def _pick_first(d: Dict[str, Any], candidates: Iterable[str]) -> Optional[Any]:
    for k in candidates:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _ensure_records(raw: Any, root_candidates: Iterable[str]) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for root in root_candidates:
            node = raw.get(root)
            if isinstance(node, list):
                return [x for x in node if isinstance(x, dict)]
        if all(isinstance(v, dict) for v in raw.values()):
            return list(raw.values())
    raise ValueError("Cannot locate list-of-dict records from prediction file.")


def _build_ref_maps(ref_data: Any, ref_id_field: str, ref_match_field: str) -> Tuple[set, Dict[str, str]]:
    if isinstance(ref_data, dict):
        refs = [ref_data]
    elif isinstance(ref_data, list):
        refs = [x for x in ref_data if isinstance(x, dict)]
    else:
        refs = []

    qids = set()
    by_match = {}
    for r in refs:
        qid = r.get(ref_id_field)
        if qid is None:
            continue
        qid = str(qid)
        qids.add(qid)
        mv = r.get(ref_match_field)
        if mv is not None:
            mvs = str(mv)
            if mvs not in by_match:
                by_match[mvs] = qid
    return qids, by_match


def _parse_id_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    raw = _load_json_or_jsonl(path)
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[str(k)] = str(v)
        return out
    if isinstance(raw, list):
        for e in raw:
            if not isinstance(e, dict):
                continue
            rk = e.get("raw_id", e.get("id", e.get("source_id")))
            qk = e.get("question_id", e.get("target_id"))
            if rk is None or qk is None:
                continue
            out[str(rk)] = str(qk)
        return out
    raise ValueError("Unsupported id_map format; expected JSON object or list-of-dict.")


def _infer_pgraphrag_output(repo_root: Path, ranking_input: Path, model: str, mode: str, k: int) -> Path:
    filename = ranking_input.stem
    if filename.startswith("RANKING-"):
        filename = filename[len("RANKING-"):]
    parts = filename.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse ranking input filename: {ranking_input.name}")
    dataset, split, task, ranker = parts[0], parts[1], parts[2], parts[3]
    model_tag = model.upper()
    out = repo_root / "results" / dataset / split / task / model_tag / ranker / f"OUTPUT-{dataset}_{split}_{task}_{model_tag}_{ranker}-{mode}_k{k}.json"
    return out


def _infer_latest_afce_pred(repo_root: Path, dataset_tag: str) -> Path:
    pred_dir = repo_root / "AP_Bots" / "files" / "preds"
    if not pred_dir.exists():
        raise FileNotFoundError(f"AF+CE pred directory not found: {pred_dir}")
    cands = sorted(
        [p for p in pred_dir.glob(f"{dataset_tag}*.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No AF+CE prediction json found for dataset tag '{dataset_tag}' under {pred_dir}")
    return cands[0]


def main():
    parser = argparse.ArgumentParser(description="Adapt official baseline output into unified {question_id,hypothesis} jsonl format.")
    parser.add_argument("--baseline", type=str, required=True, choices=["afce", "pgraphrag", "lightrag", "generic"])
    parser.add_argument("--pred_file", type=str, default="", help="Explicit raw prediction file path (json/jsonl/csv).")
    parser.add_argument("--out_file", type=str, required=True, help="Unified output jsonl path.")
    parser.add_argument("--ref_file", type=str, default="", help="Optional reference file for ID alignment.")
    parser.add_argument("--id_map_json", type=str, default="", help="Optional raw_id->question_id map file.")
    parser.add_argument("--id_field", type=str, default="", help="Comma-separated id field candidates.")
    parser.add_argument("--hyp_field", type=str, default="", help="Comma-separated hypothesis field candidates.")
    parser.add_argument("--root_field", type=str, default="", help="Comma-separated root list field candidates.")
    parser.add_argument("--ref_id_field", type=str, default="question_id")
    parser.add_argument("--ref_match_field", type=str, default="question_id")
    parser.add_argument("--use_index_if_missing_id", action="store_true")
    parser.add_argument("--strict", action="store_true")

    # Inference helpers for official baselines
    parser.add_argument("--repo_root", type=str, default="")
    parser.add_argument("--ranking_input", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--dataset_tag", type=str, default="")

    args = parser.parse_args()

    baseline = args.baseline.strip().lower()

    pred_file = Path(args.pred_file).resolve() if args.pred_file else None
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None

    if pred_file is None:
        if baseline == "pgraphrag":
            if not repo_root or not args.ranking_input or not args.model or not args.mode or args.k <= 0:
                raise ValueError("For pgraphrag auto-infer, require --repo_root --ranking_input --model --mode --k.")
            pred_file = _infer_pgraphrag_output(
                repo_root=repo_root,
                ranking_input=Path(args.ranking_input).resolve(),
                model=args.model,
                mode=args.mode,
                k=args.k,
            )
        elif baseline == "afce":
            if not repo_root or not args.dataset_tag:
                raise ValueError("For afce auto-infer, require --repo_root and --dataset_tag.")
            pred_file = _infer_latest_afce_pred(repo_root=repo_root, dataset_tag=args.dataset_tag)
        else:
            raise ValueError(f"Baseline '{baseline}' requires --pred_file when auto-infer is not supported.")

    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    root_candidates = [x.strip() for x in args.root_field.split(",") if x.strip()] if args.root_field else ["golds", "results", "data", "predictions"]
    id_candidates = [x.strip() for x in args.id_field.split(",") if x.strip()] if args.id_field else DEFAULT_ID_CANDIDATES[baseline]
    hyp_candidates = [x.strip() for x in args.hyp_field.split(",") if x.strip()] if args.hyp_field else DEFAULT_HYP_CANDIDATES[baseline]

    raw_obj = _load_any(pred_file)
    records = _ensure_records(raw_obj, root_candidates)

    ref_qids = set()
    ref_map = {}
    if args.ref_file:
        ref_path = Path(args.ref_file).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_path}")
        ref_obj = _load_any(ref_path)
        ref_qids, ref_map = _build_ref_maps(ref_obj, args.ref_id_field, args.ref_match_field)

    id_map = _parse_id_map(Path(args.id_map_json).resolve()) if args.id_map_json else {}

    out_rows = []
    missing_id = 0
    missing_hyp = 0
    unresolved_qid = 0

    for idx, rec in enumerate(records):
        raw_id_val = _pick_first(rec, id_candidates)
        if raw_id_val is None and args.use_index_if_missing_id:
            raw_id_val = idx
        if raw_id_val is None:
            missing_id += 1
            if args.strict:
                raise ValueError(f"Record {idx} missing ID fields {id_candidates}.")
            continue

        raw_id = str(raw_id_val)
        hyp_val = _pick_first(rec, hyp_candidates)
        if hyp_val is None:
            missing_hyp += 1
            if args.strict:
                raise ValueError(f"Record {idx} missing hypothesis fields {hyp_candidates}.")
            continue

        qid = None
        if raw_id in id_map:
            qid = id_map[raw_id]
        elif ref_qids:
            if raw_id in ref_qids:
                qid = raw_id
            elif raw_id in ref_map:
                qid = ref_map[raw_id]
        if qid is None:
            if ref_qids:
                unresolved_qid += 1
                if args.strict:
                    raise ValueError(f"Record {idx} raw_id={raw_id} cannot be aligned to reference IDs.")
                continue
            qid = raw_id

        out_rows.append({
            "question_id": str(qid),
            "hypothesis": str(hyp_val),
            "_meta": {
                "baseline": baseline,
                "raw_id": raw_id,
                "pred_file": str(pred_file),
            },
        })

    out_path = Path(args.out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[adapter] baseline={baseline}")
    print(f"[adapter] pred_file={pred_file}")
    print(f"[adapter] output={out_path}")
    print(f"[adapter] total_records={len(records)}")
    print(f"[adapter] converted={len(out_rows)}")
    print(f"[adapter] missing_id={missing_id} missing_hyp={missing_hyp} unresolved_qid={unresolved_qid}")


if __name__ == "__main__":
    main()
