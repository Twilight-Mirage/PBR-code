import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.evidence_source import (
    EvalConfig,
    build_oracle_evidence,
    infer_dataset_name,
    is_natural_oracle_dataset,
)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return obj["data"]
        raise ValueError(f"Unsupported JSON structure in {path}")
    except json.JSONDecodeError:
        rows: List[Dict[str, Any]] = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
        return rows


def _save_records(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _answer_text(sample: Dict[str, Any]) -> str:
    for key in ("answer", "gold_answer", "target", "reference"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _joined_evidence_text(retrieval_results: Dict[str, Any]) -> str:
    texts = []
    for item in retrieval_results.get("ranked_items", []):
        if isinstance(item, dict):
            txt = item.get("text")
            if isinstance(txt, str) and txt.strip():
                texts.append(txt.strip())
    return "\n".join(texts)


def _question_id(sample: Dict[str, Any], idx: int) -> str:
    for key in ("question_id", "id", "sample_id"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return f"idx_{idx}"



def _ensure_generation_schema(sample: Dict[str, Any], idx: int) -> None:
    required = [
        "question_id",
        "question",
        "question_date",
        "answer",
        "haystack_dates",
        "haystack_sessions",
        "haystack_session_ids",
    ]
    missing = [k for k in required if k not in sample]
    if missing:
        raise ValueError(
            f"Sample {_question_id(sample, idx)} missing generation-required keys: {missing}. "
            "Current LLM-GT baseline path expects LongMemEval-style schema because generator/prompt are unchanged."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build unified LLM-GT baseline input by switching evidence source only."
    )
    parser.add_argument("--in_file", type=str, required=True, help="Input dataset json/jsonl.")
    parser.add_argument("--out_file", type=str, required=True, help="Output json path for generation.")
    parser.add_argument(
        "--evidence_mode",
        type=str,
        default="retrieved",
        choices=["retrieved", "oracle"],
        help="Use retrieved evidence from input, or oracle evidence from natural labels.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Optional dataset name override (e.g., LongMemEval, PersonaBench).",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="session",
        choices=["session", "turn"],
        help="Oracle evidence granularity for LongMemEval.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k evidence to keep in oracle mode. Keep aligned with generation topk_context.",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=3000,
        help="Metadata only; generation token budget remains unchanged.",
    )
    parser.add_argument(
        "--oracle_sanity_check",
        action="store_true",
        help="Warn when answer string appears in oracle evidence text.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()

    samples = _load_records(in_path)
    if not samples:
        raise ValueError(f"Empty input: {in_path}")

    dataset_name = infer_dataset_name(args.dataset_name, str(in_path), samples[0])
    if args.evidence_mode == "oracle" and not is_natural_oracle_dataset(dataset_name):
        raise NotImplementedError(
            f"Oracle evidence is only supported for natural-oracle datasets. "
            f"Resolved dataset='{dataset_name or 'unknown'}' from {in_path}."
        )

    cfg = EvalConfig(
        evidence_mode=args.evidence_mode,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        granularity=args.granularity,
        dataset_name=dataset_name,
    )

    out_rows: List[Dict[str, Any]] = []
    leak_hits = 0
    leak_examples: List[str] = []
    for idx, sample in enumerate(samples):
        _ensure_generation_schema(sample, idx)
        row = copy.deepcopy(sample)
        if args.evidence_mode == "retrieved":
            retrieval = row.get("retrieval_results")
            if not isinstance(retrieval, dict) or not isinstance(retrieval.get("ranked_items"), list):
                raise ValueError(
                    f"Sample {_question_id(row, idx)} has no retrieval_results.ranked_items in retrieved mode."
                )
            retrieval["evidence_source"] = "retrieved"
        else:
            oracle = build_oracle_evidence(row, cfg)
            oracle["evidence_source"] = "oracle"
            row["retrieval_results"] = oracle

            if args.oracle_sanity_check:
                ans = _answer_text(row).lower()
                joined = _joined_evidence_text(oracle).lower()
                if ans and joined and ans in joined:
                    leak_hits += 1
                    if len(leak_examples) < 10:
                        leak_examples.append(_question_id(row, idx))
        out_rows.append(row)

    _save_records(out_path, out_rows)
    print(
        json.dumps(
            {
                "in_file": str(in_path),
                "out_file": str(out_path),
                "evidence_mode": args.evidence_mode,
                "dataset_name": dataset_name,
                "samples": len(out_rows),
                "oracle_sanity_hits": leak_hits,
                "oracle_sanity_examples": leak_examples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

