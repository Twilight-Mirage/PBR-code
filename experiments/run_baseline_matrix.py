import argparse
import json
import os
import shlex
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.project_runtime import (
    default_longmemeval_input,
    resolve_api_key as resolve_project_api_key,
    resolve_default_embedding_task,
    resolve_default_retrieval_model_name,
)
from experiments.experiment_logging import MatrixRunLogger, redact_command, run_logged_step


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_subprocess_env():
    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        project_key = resolve_project_api_key(env_name="OPENAI_API_KEY")
        if project_key:
            env["OPENAI_API_KEY"] = project_key
    return env


def normalize_command(raw):
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return shlex.split(raw, posix=False)
    raise TypeError(f"Unsupported command type: {type(raw)}")


def render_placeholders(obj, ctx):
    if isinstance(obj, str):
        return obj.format(**ctx)
    if isinstance(obj, list):
        return [render_placeholders(x, ctx) for x in obj]
    if isinstance(obj, dict):
        return {k: render_placeholders(v, ctx) for k, v in obj.items()}
    return obj


def run_step(
    step,
    dry_run=False,
    logger=None,
    exp_name="",
    exp_index=1,
    exp_total=1,
    step_index=1,
    step_total=1,
):
    cmd = normalize_command(step["cmd"])
    wrapped_step = dict(step)
    wrapped_step["cmd"] = cmd

    if logger is None:
        cwd = wrapped_step.get("cwd")
        print(f"[STEP] {wrapped_step['name']}")
        print(" ".join(cmd))
        if cwd:
            print(f"[CWD]  {cwd}")
        if not dry_run:
            subprocess.run(cmd, cwd=cwd, check=True, env=build_subprocess_env())
        return {
            "status": "dry_run" if dry_run else "success",
            "returncode": 0,
            "duration_sec": 0.0,
            "step_log": "",
        }

    return run_logged_step(
        step=wrapped_step,
        env=build_subprocess_env(),
        dry_run=dry_run,
        logger=logger,
        exp_name=exp_name,
        exp_index=exp_index,
        exp_total=exp_total,
        step_index=step_index,
        step_total=step_total,
    )
def add_cli_arg(cmd, key, value):
    flag = f"--{key}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])

def _require_field(cfg, key, exp_name, method):
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"Experiment '{exp_name}' ({method}) missing required field: {key}")
    return cfg[key]


def _resolve_existing_path(path_value, exp_name, method, field_name):
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment '{exp_name}' ({method}) has invalid {field_name}: {path}"
        )
    return path


def _local_gen_common(global_cfg, exp_cfg, run_dir, exp_name):
    openai_key_env = exp_cfg.get("openai_key_env", global_cfg.get("openai_key_env", "OPENAI_API_KEY"))
    openai_key = exp_cfg.get("openai_key", os.getenv(openai_key_env, ""))
    if not openai_key:
        openai_key = resolve_project_api_key(env_name=openai_key_env)
    if not openai_key:
        openai_key = "EMPTY"

    model_name = exp_cfg.get("gen_model_name", global_cfg.get("gen_model_name"))
    model_alias = exp_cfg.get("gen_model_alias", global_cfg.get("gen_model_alias", model_name))
    if not model_name:
        raise ValueError(f"Experiment '{exp_name}' missing gen_model_name (exp or global).")

    return {
        "model_name": model_name,
        "model_alias": model_alias,
        "openai_key": openai_key,
        "openai_base_url": exp_cfg.get("openai_base_url", global_cfg.get("openai_base_url", None)),
        "openai_organization": exp_cfg.get("openai_organization", global_cfg.get("openai_organization", None)),
        "openai_enable_thinking": exp_cfg.get("openai_enable_thinking", global_cfg.get("openai_enable_thinking", None)),
        "openai_extra_body_json": exp_cfg.get("openai_extra_body_json", global_cfg.get("openai_extra_body_json", None)),
        "history_format": exp_cfg.get("history_format", global_cfg.get("history_format", "json")),
        "useronly": str(exp_cfg.get("useronly", global_cfg.get("useronly", "false"))).lower(),
        "cot": str(exp_cfg.get("cot", global_cfg.get("cot", "false"))).lower(),
        "con": str(exp_cfg.get("con", global_cfg.get("con", "false"))).lower(),
        "topk_context": int(exp_cfg.get("topk_context", global_cfg.get("topk_context", 10))),
        "merge_key_expansion_into_value": exp_cfg.get(
            "merge_key_expansion_into_value",
            global_cfg.get("merge_key_expansion_into_value", "none"),
        ),
        "gen_length": exp_cfg.get("gen_length", global_cfg.get("gen_length", None)),
        "out_file": str(Path(run_dir) / f"{exp_name}_hypotheses.jsonl"),
    }


def _append_retrieval_eval_step(
    steps: list,
    metadata: dict,
    repo_root: Path,
    python_bin: str,
    run_dir: Path,
    exp_name: str,
    baseline_type: str,
    retrieval_output: Path,
    ref_file: Path,
    output_prefix: str = "",
) -> tuple:
    """
    为涉及检索的baseline添加检索评估步骤
    
    Args:
        steps: 现有步骤列表
        metadata: 元数据字典
        repo_root: 仓库根目录
        python_bin: Python可执行文件路径
        run_dir: 运行目录
        exp_name: 实验名称
        baseline_type: baseline类型 (naive_rag, dua_rag, afce, pgraphrag等)
        retrieval_output: 检索结果文件路径
        ref_file: 参考文件路径（包含ground truth）
        output_prefix: 输出文件前缀
    """
    if not retrieval_output or not retrieval_output.exists():
        return steps, metadata
    
    prefix = output_prefix or f"{exp_name}"
    metrics_out = run_dir / f"{prefix}_retrieval_metrics.json"
    detailed_out = run_dir / f"{prefix}_retrieval_detailed.json"
    
    eval_cmd = [
        python_bin,
        "-m",
        "experiments.evaluate_retrieval_unified",
        "--pred_file", str(retrieval_output),
        "--baseline_type", baseline_type,
        "--ref_file", str(ref_file),
        "--output_dir", str(run_dir),
        "--output_prefix", prefix,
    ]
    
    steps.append({
        "name": "retrieval_eval",
        "cmd": eval_cmd,
        "cwd": str(repo_root),
    })
    
    metadata["retrieval_metrics"] = str(metrics_out)
    metadata["retrieval_detailed"] = str(detailed_out)
    
    return steps, metadata


def build_naive_rag_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    run_dir = run_root / exp_name
    ensure_dir(run_dir)

    input_json = cfg.get("input_json", global_cfg.get("input_json", default_longmemeval_input("s")))
    if not input_json:
        raise ValueError(f"Experiment '{exp_name}' missing input_json.")
    in_file = _resolve_existing_path(input_json, exp_name, "naive_rag", "input_json")

    retriever = cfg.get("retriever", global_cfg.get("retriever", "flat-contriever"))
    granularity = cfg.get("granularity", global_cfg.get("granularity", "session"))
    outfile_prefix = cfg.get("outfile_prefix", exp_name)
    retrieval_out = str(run_dir / f"{outfile_prefix}_retrievallog_{granularity}_{retriever}")

    retrieval_cmd = [
        python_bin,
        "-m",
        "src.retrieval.run_retrieval",
        "--in_file",
        str(in_file),
        "--out_dir",
        str(run_dir),
        "--outfile_prefix",
        outfile_prefix,
        "--retriever",
        retriever,
        "--granularity",
        granularity,
    ]
    if cfg.get("cache_dir", global_cfg.get("cache_dir")):
        retrieval_cmd.extend(["--cache_dir", str(cfg.get("cache_dir", global_cfg.get("cache_dir")))])

    for k in ("index_expansion_method", "index_expansion_llm", "index_expansion_result_cache", "index_expansion_result_join_mode"):
        if cfg.get(k, global_cfg.get(k)) not in (None, ""):
            retrieval_cmd.extend([f"--{k}", str(cfg.get(k, global_cfg.get(k)))])

    gen = _local_gen_common(global_cfg, cfg, run_dir, exp_name)
    retriever_type = "flat-session" if granularity == "session" else "flat-turn"
    generation_cmd = [
        python_bin,
        "-m",
        "src.generation.run_generation",
        "--in_file",
        retrieval_out,
        "--out_dir",
        str(run_dir),
        "--out_file",
        gen["out_file"],
        "--model_name",
        gen["model_name"],
        "--model_alias",
        gen["model_alias"],
        "--openai_key",
        gen["openai_key"],
        "--retriever_type",
        retriever_type,
        "--topk_context",
        str(gen["topk_context"]),
        "--history_format",
        gen["history_format"],
        "--useronly",
        gen["useronly"],
        "--cot",
        gen["cot"],
        "--con",
        gen["con"],
        "--merge_key_expansion_into_value",
        gen["merge_key_expansion_into_value"],
    ]
    if gen["openai_base_url"]:
        generation_cmd.extend(["--openai_base_url", str(gen["openai_base_url"])])
    if gen["openai_organization"]:
        generation_cmd.extend(["--openai_organization", str(gen["openai_organization"])])
    if gen["openai_enable_thinking"] is not None:
        generation_cmd.extend(["--openai_enable_thinking", str(gen["openai_enable_thinking"]).lower()])
    if gen["openai_extra_body_json"] not in (None, ""):
        generation_cmd.extend(["--openai_extra_body_json", str(gen["openai_extra_body_json"])])
    if gen["gen_length"] is not None:
        generation_cmd.extend(["--gen_length", str(gen["gen_length"])])

    steps = [
        {"name": "retrieval", "cmd": retrieval_cmd, "cwd": str(repo_root)},
        {"name": "generation", "cmd": generation_cmd, "cwd": str(repo_root)},
    ]

    if bool(cfg.get("run_eval", global_cfg.get("run_eval", False))):
        eval_model = cfg.get("eval_model", global_cfg.get("eval_model", "gpt-4o-mini"))
        ref_file = Path(cfg.get("ref_json", global_cfg.get("ref_json", in_file))).resolve()
        eval_cmd = [
            python_bin,
            "-m",
            "src.evaluation.evaluate_qa",
            eval_model,
            gen["out_file"],
            str(ref_file),
        ]
        steps.append({"name": "eval", "cmd": eval_cmd, "cwd": str(repo_root)})

    metadata = {
        "run_dir": str(run_dir),
        "retrieval_output": retrieval_out,
        "generation_output": gen["out_file"],
    }
    
    ref_file = Path(cfg.get("ref_json", global_cfg.get("ref_json", in_file))).resolve()
    steps, metadata = _append_retrieval_eval_step(
        steps=steps,
        metadata=metadata,
        repo_root=repo_root,
        python_bin=python_bin,
        run_dir=run_dir,
        exp_name=exp_name,
        baseline_type="naive_rag",
        retrieval_output=Path(retrieval_out),
        ref_file=ref_file,
        output_prefix=outfile_prefix,
    )
    
    return steps, metadata


def build_history_rag_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    run_dir = run_root / exp_name
    ensure_dir(run_dir)

    input_json = cfg.get("input_json", global_cfg.get("input_json", default_longmemeval_input("s")))
    if not input_json:
        raise ValueError(f"Experiment '{exp_name}' missing input_json.")
    in_file = _resolve_existing_path(input_json, exp_name, "history_rag", "input_json")

    gen = _local_gen_common(global_cfg, cfg, run_dir, exp_name)
    retriever_type = cfg.get("retriever_type", global_cfg.get("retriever_type", "orig-session"))
    generation_cmd = [
        python_bin,
        "-m",
        "src.generation.run_generation",
        "--in_file",
        str(in_file),
        "--out_dir",
        str(run_dir),
        "--out_file",
        gen["out_file"],
        "--model_name",
        gen["model_name"],
        "--model_alias",
        gen["model_alias"],
        "--openai_key",
        gen["openai_key"],
        "--retriever_type",
        retriever_type,
        "--topk_context",
        str(gen["topk_context"]),
        "--history_format",
        gen["history_format"],
        "--useronly",
        gen["useronly"],
        "--cot",
        gen["cot"],
        "--con",
        gen["con"],
        "--merge_key_expansion_into_value",
        gen["merge_key_expansion_into_value"],
    ]
    if gen["openai_base_url"]:
        generation_cmd.extend(["--openai_base_url", str(gen["openai_base_url"])])
    if gen["openai_organization"]:
        generation_cmd.extend(["--openai_organization", str(gen["openai_organization"])])
    if gen["openai_enable_thinking"] is not None:
        generation_cmd.extend(["--openai_enable_thinking", str(gen["openai_enable_thinking"]).lower()])
    if gen["openai_extra_body_json"] not in (None, ""):
        generation_cmd.extend(["--openai_extra_body_json", str(gen["openai_extra_body_json"])])
    if gen["gen_length"] is not None:
        generation_cmd.extend(["--gen_length", str(gen["gen_length"])])

    steps = [{"name": "generation", "cmd": generation_cmd, "cwd": str(repo_root)}]
    if bool(cfg.get("run_eval", global_cfg.get("run_eval", False))):
        eval_model = cfg.get("eval_model", global_cfg.get("eval_model", "gpt-4o-mini"))
        ref_file = Path(cfg.get("ref_json", global_cfg.get("ref_json", in_file))).resolve()
        eval_cmd = [
            python_bin,
            "-m",
            "src.evaluation.evaluate_qa",
            eval_model,
            gen["out_file"],
            str(ref_file),
        ]
        steps.append({"name": "eval", "cmd": eval_cmd, "cwd": str(repo_root)})

    metadata = {"run_dir": str(run_dir), "generation_output": gen["out_file"]}
    return steps, metadata



def _append_unified_adapter_eval_steps(steps, metadata, repo_root, python_bin, run_dir, exp_name, baseline_name, cfg, global_cfg, pred_file_hint=None, extra_adapter_args=None):
    if not bool(cfg.get("unified_eval", global_cfg.get("unified_eval", False))):
        return steps, metadata

    ref_json = cfg.get("ref_json", global_cfg.get("ref_json", global_cfg.get("input_json", global_cfg.get("in_file", ""))))
    if not ref_json:
        raise ValueError(f"Experiment '{exp_name}' ({baseline_name}) unified_eval=true but ref_json is missing.")
    ref_path = _resolve_existing_path(ref_json, exp_name, baseline_name, "ref_json")

    unified_out = Path(cfg.get("unified_out", Path(run_dir) / f"{exp_name}_unified_hypotheses.jsonl"))
    if not unified_out.is_absolute():
        unified_out = (Path(repo_root) / unified_out).resolve()

    adapter_cmd = [
        python_bin,
        "-m",
        "experiments.adapt_official_baseline_output",
        "--baseline",
        baseline_name,
        "--out_file",
        str(unified_out),
        "--ref_file",
        str(ref_path),
    ]

    adapter_pred_file = cfg.get("adapter_pred_file")
    if adapter_pred_file:
        adapter_pred_path = _resolve_existing_path(adapter_pred_file, exp_name, baseline_name, "adapter_pred_file")
        adapter_cmd.extend(["--pred_file", str(adapter_pred_path)])
    elif pred_file_hint:
        adapter_cmd.extend(["--pred_file", str(pred_file_hint)])

    id_map_json = cfg.get("id_map_json")
    if id_map_json:
        id_map_path = _resolve_existing_path(id_map_json, exp_name, baseline_name, "id_map_json")
        adapter_cmd.extend(["--id_map_json", str(id_map_path)])

    for key in ("id_field", "hyp_field", "root_field", "ref_id_field", "ref_match_field"):
        value = cfg.get(key, global_cfg.get(key))
        if value not in (None, ""):
            adapter_cmd.extend([f"--{key}", str(value)])
    if bool(cfg.get("use_index_if_missing_id", global_cfg.get("use_index_if_missing_id", False))):
        adapter_cmd.append("--use_index_if_missing_id")
    if bool(cfg.get("adapter_strict", global_cfg.get("adapter_strict", False))):
        adapter_cmd.append("--strict")

    if isinstance(extra_adapter_args, list):
        adapter_cmd.extend([str(x) for x in extra_adapter_args])

    steps.append({"name": "adapt_to_unified", "cmd": adapter_cmd, "cwd": str(repo_root)})

    eval_model = cfg.get("eval_model", global_cfg.get("eval_model", "gpt-4o-mini"))
    eval_cmd = [
        python_bin,
        "-m",
        "src.evaluation.evaluate_qa",
        str(eval_model),
        str(unified_out),
        str(ref_path),
    ]
    steps.append({"name": "eval_unified", "cmd": eval_cmd, "cwd": str(repo_root)})

    metadata["unified_output"] = str(unified_out)
    metadata["unified_ref"] = str(ref_path)
    return steps, metadata
def build_pgraphrag_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    run_dir = run_root / exp_name
    ensure_dir(run_dir)

    repo = _resolve_existing_path(
        cfg.get("repo_path", global_cfg.get("pgraphrag_repo", "./third_party_baselines/PGraphRAG")),
        exp_name,
        "pgraphrag_official",
        "repo_path",
    )
    input_path = _resolve_existing_path(
        _require_field(cfg, "input", exp_name, "pgraphrag_official"),
        exp_name,
        "pgraphrag_official",
        "input",
    )
    model = cfg.get("model", global_cfg.get("pgraphrag_model", "gpt"))

    modes = cfg.get("mode") if isinstance(cfg.get("mode"), list) and cfg["mode"] else []
    ks = cfg.get("k") if isinstance(cfg.get("k"), list) and cfg["k"] else []

    cmd = [python_bin, "master_generation.py", "--input", str(input_path), "--model", str(model)]
    if modes:
        cmd.extend(["--mode"] + [str(x) for x in modes])
    if ks:
        cmd.extend(["--k"] + [str(x) for x in ks])

    steps = [{"name": "pgraphrag_generation", "cmd": cmd, "cwd": str(repo)}]
    metadata = {"repo": str(repo)}

    pred_file_hint = None
    if len(modes) == 1 and len(ks) == 1:
        filename = Path(input_path).stem
        if filename.startswith("RANKING-"):
            filename = filename[len("RANKING-"):]
        parts = filename.split("_")
        if len(parts) >= 4:
            dataset, split, task, ranker = parts[0], parts[1], parts[2], parts[3]
            model_tag = str(model).upper()
            pred_file_hint = repo / "results" / dataset / split / task / model_tag / ranker / f"OUTPUT-{dataset}_{split}_{task}_{model_tag}_{ranker}-{modes[0]}_k{ks[0]}.json"

    adapter_extra = []
    if not pred_file_hint:
        if len(modes) == 1 and len(ks) == 1:
            adapter_extra.extend([
                "--repo_root", str(repo),
                "--ranking_input", str(input_path),
                "--model", str(model),
                "--mode", str(modes[0]),
                "--k", str(ks[0]),
            ])

    steps, metadata = _append_unified_adapter_eval_steps(
        steps=steps,
        metadata=metadata,
        repo_root=repo_root,
        python_bin=python_bin,
        run_dir=run_dir,
        exp_name=exp_name,
        baseline_name="pgraphrag",
        cfg=cfg,
        global_cfg=global_cfg,
        pred_file_hint=pred_file_hint,
        extra_adapter_args=adapter_extra if adapter_extra else None,
    )
    
    ref_json = cfg.get("ref_json", global_cfg.get("ref_json", ""))
    if ref_json and input_path:
        ref_path = Path(ref_json).resolve()
        steps, metadata = _append_retrieval_eval_step(
            steps=steps,
            metadata=metadata,
            repo_root=repo_root,
            python_bin=python_bin,
            run_dir=run_dir,
            exp_name=exp_name,
            baseline_type="pgraphrag",
            retrieval_output=input_path,
            ref_file=ref_path,
            output_prefix=f"{exp_name}_retrieval",
        )
    
    return steps, metadata


def build_afce_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    run_dir = run_root / exp_name
    ensure_dir(run_dir)

    repo = _resolve_existing_path(
        cfg.get("repo_path", global_cfg.get("afce_repo", "./third_party_baselines/AP-Bots")),
        exp_name,
        "afce_official",
        "repo_path",
    )
    dataset = _require_field(cfg, "dataset", exp_name, "afce_official")
    cmd = [python_bin, "-m", "AP_Bots.run_exp", "-d", str(dataset)]
    if cfg.get("top_k") is not None:
        cmd.extend(["-k", str(cfg["top_k"])])
    if cfg.get("retriever") is not None:
        cmd.extend(["-r", str(cfg["retriever"])])
    if isinstance(cfg.get("features"), list) and cfg["features"]:
        cmd.extend(["-f"] + [str(x) for x in cfg["features"]])
    if cfg.get("counter_examples") is not None:
        cmd.extend(["-ce", str(cfg["counter_examples"])])
    if cfg.get("repetition_step") is not None:
        cmd.extend(["-rs", str(cfg["repetition_step"])])
    if cfg.get("openai_batch") is True:
        cmd.append("--openai_batch")
    if cfg.get("prompt_style"):
        cmd.extend(["-ps", str(cfg["prompt_style"])])

    steps = [{"name": "afce_run_exp", "cmd": cmd, "cwd": str(repo)}]
    metadata = {"repo": str(repo)}

    adapter_extra = ["--repo_root", str(repo), "--dataset_tag", str(dataset)]
    steps, metadata = _append_unified_adapter_eval_steps(
        steps=steps,
        metadata=metadata,
        repo_root=repo_root,
        python_bin=python_bin,
        run_dir=run_dir,
        exp_name=exp_name,
        baseline_name="afce",
        cfg=cfg,
        global_cfg=global_cfg,
        pred_file_hint=None,
        extra_adapter_args=adapter_extra,
    )
    
    ref_json = cfg.get("ref_json", global_cfg.get("ref_json", ""))
    if ref_json:
        ref_path = Path(ref_json).resolve()
        pred_dir = repo / "AP_Bots" / "files" / "preds"
        pred_files = sorted(pred_dir.glob(f"{dataset}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pred_files:
            steps, metadata = _append_retrieval_eval_step(
                steps=steps,
                metadata=metadata,
                repo_root=repo_root,
                python_bin=python_bin,
                run_dir=run_dir,
                exp_name=exp_name,
                baseline_type="afce",
                retrieval_output=pred_files[0],
                ref_file=ref_path,
                output_prefix=f"{exp_name}_retrieval",
            )
    
    return steps, metadata


def build_lightrag_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    run_dir = run_root / exp_name
    ensure_dir(run_dir)

    repo = _resolve_existing_path(
        cfg.get("repo_path", global_cfg.get("lightrag_repo", "./third_party_baselines/LightRAG")),
        exp_name,
        "lightrag_official",
        "repo_path",
    )

    if cfg.get("command"):
        cmd = normalize_command(cfg["command"])
    else:
        entry_script = cfg.get("entry_script", "examples/lightrag_openai_demo.py")
        cmd = [python_bin, str(entry_script)]
        if isinstance(cfg.get("entry_args"), list):
            cmd.extend([str(x) for x in cfg["entry_args"]])

    steps = [{"name": "lightrag_run", "cmd": cmd, "cwd": str(repo)}]
    metadata = {"repo": str(repo)}

    # For LightRAG, explicit adapter_pred_file is typically required because official demos do not emit a fixed benchmark output path.
    steps, metadata = _append_unified_adapter_eval_steps(
        steps=steps,
        metadata=metadata,
        repo_root=repo_root,
        python_bin=python_bin,
        run_dir=run_dir,
        exp_name=exp_name,
        baseline_name="lightrag",
        cfg=cfg,
        global_cfg=global_cfg,
        pred_file_hint=None,
        extra_adapter_args=None,
    )
    return steps, metadata


def build_llm_gt_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    run_dir = run_root / exp_name
    ensure_dir(run_dir)

    input_json = cfg.get("input_json", global_cfg.get("input_json", default_longmemeval_input("s")))
    if not input_json:
        raise ValueError(f"Experiment '{exp_name}' missing input_json.")
    in_file = _resolve_existing_path(input_json, exp_name, "llm_gt_baseline", "input_json")

    evidence_mode = str(cfg.get("evidence_mode", global_cfg.get("evidence_mode", "retrieved"))).strip().lower()
    if evidence_mode not in {"retrieved", "oracle"}:
        raise ValueError(
            f"Experiment '{exp_name}' (llm_gt_baseline) has invalid evidence_mode: {evidence_mode}."
        )

    granularity = str(cfg.get("granularity", global_cfg.get("granularity", "session"))).strip().lower()
    if granularity not in {"session", "turn"}:
        raise ValueError(
            f"Experiment '{exp_name}' (llm_gt_baseline) has invalid granularity: {granularity}."
        )

    source_for_llmgt = in_file
    retrieval_out = run_dir / f"{exp_name}_retrievallog_{granularity}_{cfg.get('retriever', global_cfg.get('retriever', 'flat-contriever'))}"
    steps = []

    if evidence_mode == "retrieved":
        retrieved_json = cfg.get("retrieved_json")
        if retrieved_json:
            source_for_llmgt = _resolve_existing_path(
                retrieved_json,
                exp_name,
                "llm_gt_baseline",
                "retrieved_json",
            )
        else:
            retriever = cfg.get("retriever", global_cfg.get("retriever", "flat-contriever"))
            outfile_prefix = cfg.get("outfile_prefix", exp_name)
            retrieval_out = run_dir / f"{outfile_prefix}_retrievallog_{granularity}_{retriever}"

            retrieval_cmd = [
                python_bin,
                "-m",
                "src.retrieval.run_retrieval",
                "--in_file",
                str(in_file),
                "--out_dir",
                str(run_dir),
                "--outfile_prefix",
                outfile_prefix,
                "--retriever",
                retriever,
                "--granularity",
                granularity,
            ]
            if cfg.get("cache_dir", global_cfg.get("cache_dir")):
                retrieval_cmd.extend(["--cache_dir", str(cfg.get("cache_dir", global_cfg.get("cache_dir")))])

            for k in (
                "index_expansion_method",
                "index_expansion_llm",
                "index_expansion_result_cache",
                "index_expansion_result_join_mode",
            ):
                if cfg.get(k, global_cfg.get(k)) not in (None, ""):
                    retrieval_cmd.extend([f"--{k}", str(cfg.get(k, global_cfg.get(k)))])

            steps.append({"name": "retrieval", "cmd": retrieval_cmd, "cwd": str(repo_root)})
            source_for_llmgt = retrieval_out

    gen = _local_gen_common(global_cfg, cfg, run_dir, exp_name)
    llmgt_input = run_dir / f"{exp_name}_llmgt_input.json"
    llmgt_build_cmd = [
        python_bin,
        "-m",
        "experiments.build_llmgt_baseline_input",
        "--in_file",
        str(source_for_llmgt),
        "--out_file",
        str(llmgt_input),
        "--evidence_mode",
        evidence_mode,
        "--granularity",
        granularity,
        "--top_k",
        str(gen["topk_context"]),
        "--max_context_tokens",
        str(cfg.get("oracle_max_context_tokens", global_cfg.get("oracle_max_context_tokens", 3000))),
    ]
    dataset_name = cfg.get("dataset_name", global_cfg.get("dataset_name", ""))
    if dataset_name:
        llmgt_build_cmd.extend(["--dataset_name", str(dataset_name)])
    if bool(cfg.get("oracle_sanity_check", global_cfg.get("oracle_sanity_check", False))):
        llmgt_build_cmd.append("--oracle_sanity_check")
    steps.append({"name": "build_llmgt_input", "cmd": llmgt_build_cmd, "cwd": str(repo_root)})

    retriever_type = cfg.get("retriever_type")
    if not retriever_type:
        retriever_type = "flat-session" if granularity == "session" else "flat-turn"

    generation_cmd = [
        python_bin,
        "-m",
        "src.generation.run_generation",
        "--in_file",
        str(llmgt_input),
        "--out_dir",
        str(run_dir),
        "--out_file",
        gen["out_file"],
        "--model_name",
        gen["model_name"],
        "--model_alias",
        gen["model_alias"],
        "--openai_key",
        gen["openai_key"],
        "--retriever_type",
        retriever_type,
        "--topk_context",
        str(gen["topk_context"]),
        "--history_format",
        gen["history_format"],
        "--useronly",
        gen["useronly"],
        "--cot",
        gen["cot"],
        "--con",
        gen["con"],
        "--merge_key_expansion_into_value",
        gen["merge_key_expansion_into_value"],
    ]
    if gen["openai_base_url"]:
        generation_cmd.extend(["--openai_base_url", str(gen["openai_base_url"])])
    if gen["openai_organization"]:
        generation_cmd.extend(["--openai_organization", str(gen["openai_organization"])])
    if gen["openai_enable_thinking"] is not None:
        generation_cmd.extend(["--openai_enable_thinking", str(gen["openai_enable_thinking"]).lower()])
    if gen["openai_extra_body_json"] not in (None, ""):
        generation_cmd.extend(["--openai_extra_body_json", str(gen["openai_extra_body_json"])])
    if gen["gen_length"] is not None:
        generation_cmd.extend(["--gen_length", str(gen["gen_length"])])

    steps.append({"name": "generation", "cmd": generation_cmd, "cwd": str(repo_root)})

    if bool(cfg.get("run_eval", global_cfg.get("run_eval", False))):
        eval_model = cfg.get("eval_model", global_cfg.get("eval_model", "gpt-4o-mini"))
        ref_file = Path(cfg.get("ref_json", global_cfg.get("ref_json", in_file))).resolve()
        eval_cmd = [
            python_bin,
            "-m",
            "src.evaluation.evaluate_qa",
            eval_model,
            gen["out_file"],
            str(ref_file),
        ]
        steps.append({"name": "eval", "cmd": eval_cmd, "cwd": str(repo_root)})

    metadata = {
        "run_dir": str(run_dir),
        "evidence_mode": evidence_mode,
        "evidence_input": str(source_for_llmgt),
        "llmgt_input": str(llmgt_input),
        "generation_output": gen["out_file"],
    }
    if evidence_mode == "retrieved" and not cfg.get("retrieved_json"):
        metadata["retrieval_output"] = str(retrieval_out)

    return steps, metadata


def build_pbr_family_steps(repo_root, python_bin, global_cfg, exp, method_defaults):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    run_dir = run_root / exp_name
    ensure_dir(run_dir)

    input_json = cfg.get("input_json", cfg.get("in_file", global_cfg.get("input_json", global_cfg.get("in_file", default_longmemeval_input("s")))))
    if not input_json:
        raise ValueError(f"Experiment '{exp_name}' missing input_json/in_file.")
    in_file = _resolve_existing_path(input_json, exp_name, exp.get("method", "pbr_family"), "input_json")

    retrieval_out = str(run_dir / f"{exp_name}_retrieval.json")

    retrieval_model_name = cfg.get(
        "retrieval_model_name",
        global_cfg.get("retrieval_model_name", resolve_default_retrieval_model_name()),
    )
    embedding_task = cfg.get(
        "embedding_task",
        global_cfg.get("embedding_task", resolve_default_embedding_task()),
    )

    retrieval_cmd = [python_bin, "-m", "src.retrieval.retrieval_PBR"]
    add_cli_arg(retrieval_cmd, "model_type", cfg.get("model_type", method_defaults.get("model_type", "PBR")))
    add_cli_arg(retrieval_cmd, "data_type", cfg.get("data_type", global_cfg.get("data_type", "s")))
    add_cli_arg(retrieval_cmd, "retrieval_model_name", retrieval_model_name)
    add_cli_arg(retrieval_cmd, "embedding_task", embedding_task)
    add_cli_arg(retrieval_cmd, "in_file", str(in_file))
    add_cli_arg(retrieval_cmd, "out_file", retrieval_out)
    add_cli_arg(retrieval_cmd, "save_suffix", cfg.get("save_suffix", f"_exp_{exp_name}"))

    temporal_enabled = bool(cfg.get("temporal_profile", method_defaults.get("temporal_profile", False)))
    cold_enabled = bool(cfg.get("cold_start_router", method_defaults.get("cold_start_router", False)))
    explicit_enabled = bool(cfg.get("explicit_profile", method_defaults.get("explicit_profile", False)))
    add_cli_arg(retrieval_cmd, "temporal_profile", temporal_enabled)
    add_cli_arg(retrieval_cmd, "cold_start_router", cold_enabled)
    add_cli_arg(retrieval_cmd, "explicit_profile", explicit_enabled)

    llm_key, llm_env = _resolve_key(cfg, global_cfg, "llm_api_key", "llm_api_key_env", "OPENAI_API_KEY")
    if llm_key:
        add_cli_arg(retrieval_cmd, "llm_api_key", llm_key)
    add_cli_arg(retrieval_cmd, "llm_api_key_env", cfg.get("llm_api_key_env", global_cfg.get("llm_api_key_env", llm_env)))
    add_cli_arg(retrieval_cmd, "llm_base_url", cfg.get("llm_base_url", global_cfg.get("llm_base_url", "")))
    add_cli_arg(retrieval_cmd, "llm_model", cfg.get("llm_model", global_cfg.get("llm_model", "gpt-4o-mini")))
    add_cli_arg(retrieval_cmd, "llm_max_tokens", cfg.get("llm_max_tokens", global_cfg.get("llm_max_tokens", 512)))
    add_cli_arg(retrieval_cmd, "llm_temperature", cfg.get("llm_temperature", global_cfg.get("llm_temperature", 0.0)))
    add_cli_arg(retrieval_cmd, "llm_enable_thinking", cfg.get("llm_enable_thinking", global_cfg.get("llm_enable_thinking", False)))
    add_cli_arg(retrieval_cmd, "llm_extra_body_json", cfg.get("llm_extra_body_json", global_cfg.get("llm_extra_body_json", None)))

    reserved = {
        "input_json", "in_file", "ref_json", "run_eval", "eval_model",
        "openai_key", "openai_key_env", "openai_base_url", "openai_organization",
        "openai_enable_thinking", "openai_extra_body_json",
        "gen_model_name", "gen_model_alias", "gen_length",
        "history_format", "useronly", "cot", "con", "topk_context", "retriever_type",
        "merge_key_expansion_into_value",
        "retrieval_model_name", "embedding_task",
        "model_type", "temporal_profile", "cold_start_router", "explicit_profile", "save_suffix",
        "llm_api_key", "llm_api_key_env", "llm_base_url", "llm_model", "llm_max_tokens", "llm_temperature", "llm_enable_thinking", "llm_extra_body_json",
    }
    for k, v in cfg.items():
        if k in reserved:
            continue
        add_cli_arg(retrieval_cmd, k, v)

    gen = _local_gen_common(global_cfg, cfg, run_dir, exp_name)
    generation_cmd = [
        python_bin,
        "-m",
        "src.generation.run_generation",
        "--in_file",
        retrieval_out,
        "--out_dir",
        str(run_dir),
        "--out_file",
        gen["out_file"],
        "--model_name",
        gen["model_name"],
        "--model_alias",
        gen["model_alias"],
        "--openai_key",
        gen["openai_key"],
        "--retriever_type",
        str(cfg.get("retriever_type", global_cfg.get("retriever_type", "flat-session"))),
        "--topk_context",
        str(gen["topk_context"]),
        "--history_format",
        gen["history_format"],
        "--useronly",
        gen["useronly"],
        "--cot",
        gen["cot"],
        "--con",
        gen["con"],
        "--merge_key_expansion_into_value",
        gen["merge_key_expansion_into_value"],
    ]
    if gen["openai_base_url"]:
        generation_cmd.extend(["--openai_base_url", str(gen["openai_base_url"])])
    if gen["openai_organization"]:
        generation_cmd.extend(["--openai_organization", str(gen["openai_organization"])])
    if gen["openai_enable_thinking"] is not None:
        generation_cmd.extend(["--openai_enable_thinking", str(gen["openai_enable_thinking"]).lower()])
    if gen["openai_extra_body_json"] not in (None, ""):
        generation_cmd.extend(["--openai_extra_body_json", str(gen["openai_extra_body_json"])])
    if gen["gen_length"] is not None:
        generation_cmd.extend(["--gen_length", str(gen["gen_length"])])

    steps = [
        {"name": "retrieval", "cmd": retrieval_cmd, "cwd": str(repo_root)},
        {"name": "generation", "cmd": generation_cmd, "cwd": str(repo_root)},
    ]

    if bool(cfg.get("run_eval", global_cfg.get("run_eval", False))):
        eval_model = cfg.get("eval_model", global_cfg.get("eval_model", "gpt-4o-mini"))
        ref_file = Path(cfg.get("ref_json", global_cfg.get("ref_json", in_file))).resolve()
        eval_cmd = [
            python_bin,
            "-m",
            "src.evaluation.evaluate_qa",
            str(eval_model),
            gen["out_file"],
            str(ref_file),
        ]
        steps.append({"name": "eval", "cmd": eval_cmd, "cwd": str(repo_root)})

    metadata = {
        "run_dir": str(run_dir),
        "in_file": str(in_file),
        "retrieval_output": retrieval_out,
        "generation_output": gen["out_file"],
        "temporal_profile": temporal_enabled,
        "cold_start_router": cold_enabled,
        "explicit_profile": explicit_enabled,
    }
    return steps, metadata


def build_pbr_pipeline_steps(repo_root, python_bin, global_cfg, exp):
    defaults = {
        "model_type": "PBR",
        "temporal_profile": False,
        "cold_start_router": False,
        "explicit_profile": False,
    }
    return build_pbr_family_steps(repo_root, python_bin, global_cfg, exp, defaults)


def build_dua_rag_steps(repo_root, python_bin, global_cfg, exp):
    defaults = {
        "model_type": "PBR++",
        "temporal_profile": True,
        "cold_start_router": True,
        "explicit_profile": True,
    }
    return build_pbr_family_steps(repo_root, python_bin, global_cfg, exp, defaults)

def build_custom_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
    raw_cmds = cfg.get("commands", [])
    if not isinstance(raw_cmds, list) or len(raw_cmds) == 0:
        raise ValueError(f"Experiment '{exp_name}' (custom) requires args.commands as a non-empty list.")
    cwd = cfg.get("cwd", str(repo_root))
    steps = []
    for idx, raw in enumerate(raw_cmds, 1):
        steps.append({"name": f"custom_{idx}", "cmd": raw, "cwd": cwd})
    return steps, {"cwd": cwd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, required=True, help="Path to baseline matrix json.")
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated experiment names.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8-sig"))
    global_cfg = matrix.get("global", {})
    exps = matrix.get("experiments", [])
    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    run_root = Path(global_cfg.get("run_root", "./experiments/runs")).resolve()
    ensure_dir(run_root)

    builders = {
        "naive_rag": build_naive_rag_steps,
        "history_rag": build_history_rag_steps,
        "pbr_pipeline": build_pbr_pipeline_steps,
        "dua_rag": build_dua_rag_steps,
        "llm_gt_baseline": build_llm_gt_steps,
        "pgraphrag_official": build_pgraphrag_steps,
        "afce_official": build_afce_steps,
        "lightrag_official": build_lightrag_steps,
        "custom": build_custom_steps,
    }

    selected_exps = [exp for exp in exps if (not only or exp["name"] in only)]

    logger = MatrixRunLogger(
        run_root=run_root,
        matrix_path=matrix_path,
        runner_name="baseline_matrix",
        dry_run=args.dry_run,
    )
    logger.info(f"[MATRIX] selected_experiments={len(selected_exps)} run_root={run_root}")
    logger.event(
        "matrix_plan",
        total_experiments=len(selected_exps),
        experiment_names=[exp["name"] for exp in selected_exps],
    )

    matrix_status = "success"
    matrix_error = ""
    try:
        for exp_index, exp in enumerate(selected_exps, 1):
            name = exp["name"]
            method = exp.get("method", "").strip()
            if method not in builders:
                raise ValueError(f"Unknown method '{method}' for experiment '{name}'.")

            logger.info(f"[RUN] {name} ({method}) [{exp_index}/{len(selected_exps)}]")
            logger.event(
                "experiment_start",
                experiment=name,
                method=method,
                experiment_index=exp_index,
                total_experiments=len(selected_exps),
            )

            exp_t0 = time.time()
            steps = []
            metadata = {}
            status = "dry_run" if args.dry_run else "success"
            error = ""
            step_results = []

            try:
                steps, metadata = builders[method](repo_root, args.python_bin, global_cfg, exp)

                ctx = {
                    "repo_root": str(repo_root),
                    "exp_name": name,
                    "python_bin": args.python_bin,
                    "run_root": str(run_root),
                    "date": datetime.now().strftime("%Y%m%d"),
                }
                steps = render_placeholders(steps, ctx)

                for step_index, step in enumerate(steps, 1):
                    result = run_step(
                        step,
                        dry_run=args.dry_run,
                        logger=logger,
                        exp_name=name,
                        exp_index=exp_index,
                        exp_total=len(selected_exps),
                        step_index=step_index,
                        step_total=len(steps),
                    )
                    step_results.append(
                        {
                            "name": step.get("name", f"step{step_index}"),
                            "status": result.get("status"),
                            "returncode": result.get("returncode"),
                            "duration_sec": result.get("duration_sec"),
                            "step_log": result.get("step_log", ""),
                        }
                    )
            except Exception as exc:
                status = "failed"
                error = repr(exc)
                matrix_status = "failed"
                matrix_error = error
                logger.event("experiment_error", experiment=name, method=method, error=error)
                raise
            finally:
                exp_elapsed = round(time.time() - exp_t0, 3)
                manifest_path = run_root / name / "manifest.json"
                ensure_dir(manifest_path.parent)
                steps_redacted = []
                for _s in steps:
                    _item = dict(_s)
                    _cmd = _item.get("cmd")
                    if isinstance(_cmd, list):
                        _item["cmd"] = redact_command(_cmd)
                    steps_redacted.append(_item)

                manifest = {
                    "name": name,
                    "method": method,
                    "matrix": str(matrix_path),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "status": status,
                    "error": error,
                    "duration_sec": exp_elapsed,
                    "steps": steps_redacted,
                    "step_results": step_results,
                    "metadata": metadata,
                    "dry_run": bool(args.dry_run),
                    "log_dir": str(logger.log_dir),
                    "run_log": str(logger.text_log),
                    "events_log": str(logger.events_log),
                    "experiment_log_dir": str(logger.experiment_log_dir(name)),
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"[MANIFEST] {manifest_path}")
                logger.event(
                    "experiment_end",
                    experiment=name,
                    method=method,
                    status=status,
                    error=error,
                    duration_sec=exp_elapsed,
                    manifest=str(manifest_path),
                )
    finally:
        logger.finalize(status=matrix_status, error=matrix_error)

if __name__ == "__main__":
    main()
