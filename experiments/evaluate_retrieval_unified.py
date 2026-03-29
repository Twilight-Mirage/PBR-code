"""
统一检索结果评估脚本
从不同baseline的输出中提取检索结果，计算统一的检索指标

支持格式：
- DUA-RAG/PBR格式: retrieval_results.ranked_items
- Naive RAG格式: ranked_items (run_retrieval.py输出)
- AF+CE格式: 从neighbor_ratings提取检索结果
- PGraphRAG格式: 从neighbor_ratings提取检索结果
- LightRAG格式: 待扩展
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from collections import defaultdict

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.eval_utils import evaluate_retrieval


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_dua_rag_rankings(data: List[Dict]) -> List[Dict]:
    """从DUA-RAG/PBR格式提取检索结果"""
    results = []
    for item in data:
        qid = item.get("question_id", item.get("id", ""))
        retrieval_res = item.get("retrieval_results", {})
        ranked_items = retrieval_res.get("ranked_items", [])
        
        ranked_ids = []
        for ri in ranked_items:
            corpus_id = ri.get("corpus_id", ri.get("id", ""))
            if corpus_id:
                ranked_ids.append(corpus_id)
        
        correct_docs = item.get("answer_session_ids", [])
        if not correct_docs:
            haystack_ids = item.get("haystack_session_ids", [])
            correct_docs = [hid for hid in haystack_ids if "answer" in hid]
        
        results.append({
            "question_id": qid,
            "ranked_ids": ranked_ids,
            "correct_docs": correct_docs,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        })
    return results


def extract_naive_rag_rankings(data: List[Dict]) -> List[Dict]:
    """从Naive RAG格式提取检索结果"""
    results = []
    for item in data:
        qid = item.get("question_id", item.get("id", ""))
        retrieval_res = item.get("retrieval_results", {})
        ranked_items = retrieval_res.get("ranked_items", [])
        
        ranked_ids = [ri.get("corpus_id", "") for ri in ranked_items if ri.get("corpus_id")]
        
        correct_docs = item.get("answer_session_ids", [])
        if not correct_docs:
            haystack_ids = item.get("haystack_session_ids", [])
            correct_docs = [hid for hid in haystack_ids if "answer" in hid]
        
        results.append({
            "question_id": qid,
            "ranked_ids": ranked_ids,
            "correct_docs": correct_docs,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        })
    return results


def extract_afce_rankings(data: List[Dict], ref_data: List[Dict] = None) -> List[Dict]:
    """从AF+CE格式提取检索结果"""
    results = []
    
    ref_map = {}
    if ref_data:
        for ref in ref_data:
            qid = ref.get("question_id", ref.get("id", ""))
            if qid:
                ref_map[qid] = {
                    "answer_session_ids": ref.get("answer_session_ids", []),
                    "haystack_session_ids": ref.get("haystack_session_ids", []),
                }
    
    for item in data:
        qid = item.get("id", item.get("question_id", ""))
        
        neighbor_ratings = item.get("neighbor_ratings", [])
        ranked_ids = []
        for nr in neighbor_ratings:
            doc_id = nr.get("doc_id", nr.get("id", nr.get("user_id", "")))
            if doc_id is not None:
                ranked_ids.append(str(doc_id))
        
        correct_docs = []
        if qid in ref_map:
            correct_docs = ref_map[qid].get("answer_session_ids", [])
        
        results.append({
            "question_id": str(qid),
            "ranked_ids": ranked_ids,
            "correct_docs": correct_docs,
            "question": item.get("user_review_text", item.get("question", "")),
            "answer": item.get("user_review_title", item.get("answer", "")),
        })
    return results


def extract_pgraphrag_rankings(data: List[Dict], ref_data: List[Dict] = None) -> List[Dict]:
    """从PGraphRAG格式提取检索结果"""
    return extract_afce_rankings(data, ref_data)


def compute_retrieval_metrics(
    extracted_results: List[Dict],
    k_values: List[int] = [1, 3, 5, 10, 20, 50]
) -> Dict[str, Any]:
    """计算检索指标"""
    all_metrics = defaultdict(list)
    detailed_results = []
    
    for res in extracted_results:
        ranked_ids = res["ranked_ids"]
        correct_docs = set(res["correct_docs"])
        
        if not correct_docs:
            continue
        
        corpus_ids = ranked_ids
        rankings = list(range(len(ranked_ids)))
        
        item_metrics = {
            "question_id": res["question_id"],
            "num_retrieved": len(ranked_ids),
            "num_correct": len(correct_docs),
            "metrics": {}
        }
        
        for k in k_values:
            if len(rankings) < k:
                continue
            
            recall_any, recall_all, ndcg_any = evaluate_retrieval(
                rankings, list(correct_docs), corpus_ids, k=k
            )
            
            metric_key = f"@{k}"
            item_metrics["metrics"][f"recall_any{metric_key}"] = recall_any
            item_metrics["metrics"][f"recall_all{metric_key}"] = recall_all
            item_metrics["metrics"][f"ndcg_any{metric_key}"] = ndcg_any
            
            all_metrics[f"recall_any{metric_key}"].append(recall_any)
            all_metrics[f"recall_all{metric_key}"].append(recall_all)
            all_metrics[f"ndcg_any{metric_key}"].append(ndcg_any)
        
        detailed_results.append(item_metrics)
    
    averaged_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            averaged_metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values)
            }
    
    return {
        "averaged_metrics": averaged_metrics,
        "detailed_results": detailed_results,
        "total_samples": len(extracted_results),
        "valid_samples": len([r for r in extracted_results if r["correct_docs"]])
    }


def process_baseline_output(
    pred_file: Path,
    baseline_type: str,
    ref_file: Optional[Path] = None
) -> Tuple[List[Dict], Dict[str, Any]]:
    """处理单个baseline输出文件"""
    
    data = load_json(pred_file)
    if isinstance(data, dict):
        if "golds" in data:
            data = data["golds"]
        elif "results" in data:
            data = data["results"]
        else:
            data = [data]
    
    ref_data = None
    if ref_file and ref_file.exists():
        ref_data = load_json(ref_file)
        if isinstance(ref_data, dict):
            ref_data = [ref_data]
    
    if baseline_type in ["dua_rag", "pbr", "pbr_pp"]:
        extracted = extract_dua_rag_rankings(data)
    elif baseline_type in ["naive_rag", "run_retrieval"]:
        extracted = extract_naive_rag_rankings(data)
    elif baseline_type in ["afce", "af_ce"]:
        extracted = extract_afce_rankings(data, ref_data)
    elif baseline_type in ["pgraphrag", "pgrag"]:
        extracted = extract_pgraphrag_rankings(data, ref_data)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")
    
    metrics = compute_retrieval_metrics(extracted)
    
    return extracted, metrics


def main():
    parser = argparse.ArgumentParser(description="Unified retrieval evaluation for all baselines")
    parser.add_argument("--pred_file", type=str, required=True, help="Prediction file path")
    parser.add_argument("--baseline_type", type=str, required=True, 
                        choices=["dua_rag", "pbr", "pbr_pp", "naive_rag", "run_retrieval", 
                                 "afce", "af_ce", "pgraphrag", "pgrag"],
                        help="Baseline type")
    parser.add_argument("--ref_file", type=str, default="", help="Reference file for ground truth")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="", help="Output file prefix")
    parser.add_argument("--k_values", type=str, default="1,3,5,10,20,50", help="Comma-separated k values")
    args = parser.parse_args()
    
    pred_path = Path(args.pred_file).resolve()
    ref_path = Path(args.ref_file).resolve() if args.ref_file else None
    
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    output_prefix = args.output_prefix or pred_path.stem
    
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    
    print(f"Processing {args.baseline_type}: {pred_path}")
    
    extracted, metrics = process_baseline_output(
        pred_file=pred_path,
        baseline_type=args.baseline_type,
        ref_file=ref_path
    )
    
    metrics_path = output_dir / f"{output_prefix}_retrieval_metrics.json"
    detailed_path = output_dir / f"{output_prefix}_retrieval_detailed.json"
    
    save_json(metrics, metrics_path)
    save_json(extracted, detailed_path)
    
    print(f"\n{'='*60}")
    print(f"Retrieval Metrics Summary ({args.baseline_type})")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid samples (with ground truth): {metrics['valid_samples']}")
    print()
    
    for metric_name, stats in metrics["averaged_metrics"].items():
        print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    
    print()
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved detailed results to: {detailed_path}")
    
    return metrics


if __name__ == "__main__":
    main()
