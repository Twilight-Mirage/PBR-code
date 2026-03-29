"""
检索指标汇总脚本
从多个baseline的检索评估结果中汇总指标，生成对比表格
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from collections import defaultdict


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def collect_retrieval_metrics(
    run_root: Path,
    metric_files: Optional[List[str]] = None,
    patterns: List[str] = None,
) -> Dict[str, Dict]:
    """
    从运行目录收集所有检索指标文件
    
    Args:
        run_root: 实验运行根目录
        metric_files: 指定的指标文件列表
        patterns: glob模式列表，用于匹配指标文件
    """
    metrics_data = {}
    
    patterns = patterns or ["**/*_retrieval_metrics.json"]
    
    if metric_files:
        for mf in metric_files:
            mf_path = Path(mf)
            if not mf_path.is_absolute():
                mf_path = run_root / mf_path
            if mf_path.exists():
                exp_name = mf_path.stem.replace("_retrieval_metrics", "")
                try:
                    data = load_json(mf_path)
                    metrics_data[exp_name] = data
                except Exception as e:
                    print(f"Warning: Failed to load {mf_path}: {e}")
    else:
        for pattern in patterns:
            for mf_path in run_root.glob(pattern):
                exp_name = mf_path.stem.replace("_retrieval_metrics", "")
                try:
                    data = load_json(mf_path)
                    metrics_data[exp_name] = data
                except Exception as e:
                    print(f"Warning: Failed to load {mf_path}: {e}")
    
    return metrics_data


def extract_summary_table(
    metrics_data: Dict[str, Dict],
    metric_names: List[str] = None,
    k_values: List[int] = None,
) -> pd.DataFrame:
    """
    从指标数据中提取汇总表格
    
    Args:
        metrics_data: 实验名称 -> 指标数据的映射
        metric_names: 要提取的指标名称列表
        k_values: k值列表
    """
    metric_names = metric_names or ["recall_any", "recall_all", "ndcg_any"]
    k_values = k_values or [1, 3, 5, 10, 20, 50]
    
    rows = []
    
    for exp_name, data in metrics_data.items():
        avg_metrics = data.get("averaged_metrics", {})
        total_samples = data.get("total_samples", 0)
        valid_samples = data.get("valid_samples", 0)
        
        row = {
            "experiment": exp_name,
            "total_samples": total_samples,
            "valid_samples": valid_samples,
        }
        
        for metric_name in metric_names:
            for k in k_values:
                key = f"{metric_name}@{k}"
                if key in avg_metrics:
                    stats = avg_metrics[key]
                    row[key] = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                else:
                    row[key] = "-"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    col_order = ["experiment", "total_samples", "valid_samples"]
    for metric_name in metric_names:
        for k in k_values:
            col = f"{metric_name}@{k}"
            if col in df.columns:
                col_order.append(col)
    
    df = df[[c for c in col_order if c in df.columns]]
    
    return df


def generate_latex_table(
    df: pd.DataFrame,
    caption: str = "Retrieval Performance Comparison",
    label: str = "tab:retrieval_comparison",
    highlight_best: bool = True,
) -> str:
    """
    生成LaTeX表格
    """
    if df.empty:
        return "% No data available"
    
    metric_cols = [c for c in df.columns if "@" in c]
    
    best_values = {}
    if highlight_best:
        for col in metric_cols:
            numeric_vals = []
            for v in df[col]:
                if isinstance(v, str) and "±" in v:
                    try:
                        numeric_vals.append(float(v.split("±")[0].strip()))
                    except:
                        pass
            if numeric_vals:
                best_values[col] = max(numeric_vals)
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    
    col_spec = "l" + "r" * len(metric_cols)
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("  \\toprule")
    
    header = "Experiment & " + " & ".join(metric_cols) + " \\\\"
    lines.append(f"  {header}")
    lines.append("  \\midrule")
    
    for _, row in df.iterrows():
        exp_name = str(row["experiment"]).replace("_", "\\_")
        vals = []
        for col in metric_cols:
            v = row[col]
            if isinstance(v, str) and "±" in v:
                mean_str, std_str = v.split("±")
                mean_val = float(mean_str.strip())
                std_val = float(std_str.strip())
                if highlight_best and col in best_values and abs(mean_val - best_values[col]) < 1e-6:
                    vals.append(f"\\textbf{{{mean_val:.4f}}} ± {std_val:.4f}")
                else:
                    vals.append(f"{mean_val:.4f} ± {std_val:.4f}")
            else:
                vals.append(str(v))
        line = f"  {exp_name} & " + " & ".join(vals) + " \\\\"
        lines.append(line)
    
    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize retrieval metrics across baselines")
    parser.add_argument("--run_root", type=str, default="./experiments/runs", help="Experiment run root directory")
    parser.add_argument("--metric_files", type=str, nargs="*", default=None, help="Specific metric files to include")
    parser.add_argument("--patterns", type=str, nargs="*", default=None, help="Glob patterns for metric files")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="retrieval_summary", help="Output file prefix")
    parser.add_argument("--metrics", type=str, nargs="*", default=["recall_any", "ndcg_any"], help="Metrics to include")
    parser.add_argument("--k_values", type=str, default="1,3,5,10", help="Comma-separated k values")
    parser.add_argument("--format", type=str, choices=["csv", "latex", "both"], default="both", help="Output format")
    parser.add_argument("--caption", type=str, default="Retrieval Performance Comparison", help="LaTeX table caption")
    args = parser.parse_args()
    
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else run_root
    output_prefix = args.output_prefix
    
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    
    print(f"Collecting retrieval metrics from: {run_root}")
    
    metrics_data = collect_retrieval_metrics(
        run_root=run_root,
        metric_files=args.metric_files,
        patterns=args.patterns,
    )
    
    if not metrics_data:
        print("No retrieval metrics found.")
        return
    
    print(f"Found {len(metrics_data)} experiments with retrieval metrics")
    
    df = extract_summary_table(
        metrics_data=metrics_data,
        metric_names=args.metrics,
        k_values=k_values,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ["csv", "both"]:
        csv_path = output_dir / f"{output_prefix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")
    
    if args.format in ["latex", "both"]:
        latex_path = output_dir / f"{output_prefix}.tex"
        latex_content = generate_latex_table(df, caption=args.caption)
        latex_path.write_text(latex_content, encoding="utf-8")
        print(f"Saved LaTeX to: {latex_path}")
    
    print("\n" + "=" * 80)
    print("Retrieval Metrics Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()
