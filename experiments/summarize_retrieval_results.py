import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np


def summarize_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data:
        return None
    metric_keys = list(data[0]["retrieval_results"]["metrics"]["session"].keys())
    out = {}
    for key in metric_keys:
        vals = []
        for entry in data:
            if "_abs" in entry["question_id"]:
                continue
            vals.append(entry["retrieval_results"]["metrics"]["session"][key])
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, required=True, help="Glob pattern for result json files.")
    parser.add_argument("--out", type=str, default=None, help="Optional output markdown path.")
    args = parser.parse_args()

    files = sorted(glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    rows = []
    metric_order = None
    for f in files:
        metrics = summarize_file(f)
        if metrics is None:
            continue
        if metric_order is None:
            metric_order = list(metrics.keys())
        row = {"file": str(Path(f).name)}
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise SystemExit("No valid result rows to summarize.")

    header = ["file"] + metric_order
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        items = [row["file"]] + [f"{row[k]:.4f}" for k in metric_order]
        lines.append("| " + " | ".join(items) + " |")

    markdown = "\n".join(lines)
    print(markdown)

    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
