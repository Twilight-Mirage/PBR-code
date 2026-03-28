import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text[0] == "[":
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [x for x in obj if isinstance(x, dict)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return [x for x in rows if isinstance(x, dict)]


def parse_buckets(spec: str) -> List[Tuple[str, int, Optional[int]]]:
    buckets: List[Tuple[str, int, Optional[int]]] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.endswith("+"):
            lo = int(token[:-1])
            buckets.append((token, lo, None))
            continue
        if "-" not in token:
            raise ValueError(f"Invalid bucket token: {token}")
        a, b = token.split("-", 1)
        lo = int(a.strip())
        b = b.strip().lower()
        if b in {"inf", "infty", "infinite", "max"}:
            hi = None
        else:
            hi = int(b)
        buckets.append((token, lo, hi))
    if not buckets:
        raise ValueError("No valid buckets parsed.")
    return buckets


def assign_bucket(value: int, buckets: List[Tuple[str, int, Optional[int]]]) -> Optional[str]:
    for name, lo, hi in buckets:
        if value < lo:
            continue
        if hi is None or value <= hi:
            return name
    return None


def count_history(entry: Dict[str, Any], unit: str) -> Optional[int]:
    sessions = entry.get("haystack_sessions")
    if not isinstance(sessions, list):
        ids = entry.get("haystack_session_ids")
        if unit == "sessions" and isinstance(ids, list):
            return len(ids)
        return None

    if unit == "sessions":
        return len(sessions)

    if unit == "user_sessions":
        c = 0
        for sess in sessions:
            if not isinstance(sess, list):
                continue
            ok = any(isinstance(t, dict) and t.get("role") == "user" and str(t.get("content", "")).strip() for t in sess)
            if ok:
                c += 1
        return c

    if unit == "turns":
        c = 0
        for sess in sessions:
            if isinstance(sess, list):
                c += len(sess)
        return c

    if unit == "user_turns":
        c = 0
        for sess in sessions:
            if not isinstance(sess, list):
                continue
            for t in sess:
                if isinstance(t, dict) and t.get("role") == "user" and str(t.get("content", "")).strip():
                    c += 1
        return c

    raise ValueError(f"Unsupported history unit: {unit}")


def dig(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def detect_mode(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "qa"
    probe = rows[0]
    if isinstance(probe.get("autoeval_label"), dict) and "label" in probe["autoeval_label"]:
        return "qa"
    metrics = dig(probe, "retrieval_results.metrics.session")
    if isinstance(metrics, dict):
        return "retrieval"
    return "custom"


def metric_from_row(row: Dict[str, Any], mode: str, metric_field: str, retrieval_metric: str) -> Optional[float]:
    if metric_field:
        val = dig(row, metric_field)
    elif mode == "qa":
        val = dig(row, "autoeval_label.label")
    elif mode == "retrieval":
        val = dig(row, f"retrieval_results.metrics.session.{retrieval_metric}")
    else:
        return None

    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return None


def render_markdown_table(rows: List[Dict[str, Any]], metric_name: str) -> str:
    lines = [
        f"| bucket | size | {metric_name}_mean |",
        "| --- | ---: | ---: |",
    ]
    for r in rows:
        mean_str = "nan" if r["mean"] is None else f"{r['mean']:.4f}"
        lines.append(f"| {r['bucket']} | {r['size']} | {mean_str} |")
    return "\n".join(lines)


def summarize_by_bucket(
    rows: List[Dict[str, Any]],
    ref_by_qid: Dict[str, Dict[str, Any]],
    buckets: List[Tuple[str, int, Optional[int]]],
    history_unit: str,
    mode: str,
    metric_field: str,
    retrieval_metric: str,
    by_question_type: bool,
):
    agg: Dict[str, List[float]] = defaultdict(list)
    detail: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, int] = defaultdict(int)
    skipped = {"missing_qid": 0, "missing_history": 0, "missing_metric": 0, "out_of_bucket": 0}

    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            skipped["missing_qid"] += 1
            continue

        source_entry = row
        if count_history(source_entry, history_unit) is None:
            source_entry = ref_by_qid.get(qid, {})

        h_count = count_history(source_entry, history_unit)
        if h_count is None:
            skipped["missing_history"] += 1
            continue

        bname = assign_bucket(h_count, buckets)
        if bname is None:
            skipped["out_of_bucket"] += 1
            continue

        val = metric_from_row(row, mode=mode, metric_field=metric_field, retrieval_metric=retrieval_metric)
        if val is None:
            skipped["missing_metric"] += 1
            continue

        counts[bname] += 1
        agg[bname].append(val)
        if by_question_type:
            qtype = row.get("question_type") or ref_by_qid.get(qid, {}).get("question_type") or "unknown"
            detail[bname][str(qtype)].append(val)

    bucket_rows = []
    for name, _, _ in buckets:
        vals = agg.get(name, [])
        bucket_rows.append(
            {
                "bucket": name,
                "size": counts.get(name, 0),
                "mean": (sum(vals) / len(vals)) if vals else None,
            }
        )

    detail_rows = []
    if by_question_type:
        for bname, by_type in detail.items():
            for qtype, vals in by_type.items():
                detail_rows.append(
                    {
                        "bucket": bname,
                        "question_type": qtype,
                        "size": len(vals),
                        "mean": (sum(vals) / len(vals)) if vals else None,
                    }
                )
        detail_rows.sort(key=lambda x: (x["bucket"], x["question_type"]))

    return bucket_rows, detail_rows, skipped


def main():
    parser = argparse.ArgumentParser(description="Bucketized evaluation by user-history size.")
    parser.add_argument("--in_file", type=str, required=True, help="Input result file (json or jsonl).")
    parser.add_argument("--ref_file", type=str, default="", help="Optional reference file with question_id->history.")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "qa", "retrieval", "custom"], help="Metric mode.")
    parser.add_argument("--metric_field", type=str, default="", help="Dot path for custom metric field.")
    parser.add_argument("--retrieval_metric", type=str, default="ndcg_any@10", help="Metric key in retrieval_results.metrics.session.*")
    parser.add_argument("--history_unit", type=str, default="user_sessions", choices=["sessions", "user_sessions", "turns", "user_turns"], help="How to count history size.")
    parser.add_argument("--buckets", type=str, default="0-2,3-5,6-inf", help="Bucket spec, e.g. 0-2,3-5,6-inf")
    parser.add_argument("--by_question_type", action="store_true", help="Also output per-question-type bucket stats.")
    parser.add_argument("--out_json", type=str, default="", help="Optional output json path.")
    parser.add_argument("--out_md", type=str, default="", help="Optional output markdown table path.")
    args = parser.parse_args()

    in_path = Path(args.in_file).resolve()
    rows = load_json_or_jsonl(in_path)
    if not rows:
        raise SystemExit(f"No rows loaded: {in_path}")

    mode = detect_mode(rows) if args.mode == "auto" else args.mode
    if mode == "custom" and not args.metric_field:
        raise SystemExit("mode=custom requires --metric_field")

    ref_by_qid: Dict[str, Dict[str, Any]] = {}
    if args.ref_file.strip():
        ref_rows = load_json_or_jsonl(Path(args.ref_file).resolve())
        ref_by_qid = {str(x.get("question_id")): x for x in ref_rows if isinstance(x.get("question_id"), str)}

    buckets = parse_buckets(args.buckets)
    bucket_rows, detail_rows, skipped = summarize_by_bucket(
        rows=rows,
        ref_by_qid=ref_by_qid,
        buckets=buckets,
        history_unit=args.history_unit,
        mode=mode,
        metric_field=args.metric_field.strip(),
        retrieval_metric=args.retrieval_metric,
        by_question_type=args.by_question_type,
    )

    metric_name = args.metric_field if args.metric_field else ("accuracy" if mode == "qa" else args.retrieval_metric)

    print(f"mode={mode}, history_unit={args.history_unit}, metric={metric_name}")
    print(json.dumps({"skipped": skipped}, ensure_ascii=False))
    print(render_markdown_table(bucket_rows, metric_name=metric_name))

    if args.by_question_type and detail_rows:
        print("\nPer-question-type:")
        print("| bucket | question_type | size | mean |")
        print("| --- | --- | ---: | ---: |")
        for r in detail_rows:
            mean_str = "nan" if r["mean"] is None else f"{r['mean']:.4f}"
            print(f"| {r['bucket']} | {r['question_type']} | {r['size']} | {mean_str} |")

    payload = {
        "mode": mode,
        "history_unit": args.history_unit,
        "metric": metric_name,
        "buckets": bucket_rows,
        "by_question_type": detail_rows,
        "skipped": skipped,
    }

    if args.out_json.strip():
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_md.strip():
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md = render_markdown_table(bucket_rows, metric_name=metric_name)
        if args.by_question_type and detail_rows:
            lines = [md, "", "Per-question-type", "", "| bucket | question_type | size | mean |", "| --- | --- | ---: | ---: |"]
            for r in detail_rows:
                mean_str = "nan" if r["mean"] is None else f"{r['mean']:.4f}"
                lines.append(f"| {r['bucket']} | {r['question_type']} | {r['size']} | {mean_str} |")
            md = "\n".join(lines)
        out_md.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
