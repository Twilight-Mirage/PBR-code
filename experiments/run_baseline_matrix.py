import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.project_runtime import default_longmemeval_input, resolve_api_key as resolve_project_api_key


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


def run_step(step, dry_run=False):
    cmd = normalize_command(step["cmd"])
    cwd = step.get("cwd")
    print(f"[STEP] {step['name']}")
    print(" ".join(cmd))
    if cwd:
        print(f"[CWD]  {cwd}")
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True, env=build_subprocess_env())


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


def build_pgraphrag_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
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

    cmd = [python_bin, "master_generation.py", "--input", str(input_path), "--model", str(model)]
    if isinstance(cfg.get("mode"), list) and cfg["mode"]:
        cmd.extend(["--mode"] + [str(x) for x in cfg["mode"]])
    if isinstance(cfg.get("k"), list) and cfg["k"]:
        cmd.extend(["--k"] + [str(x) for x in cfg["k"]])
    return [{"name": "pgraphrag_generation", "cmd": cmd, "cwd": str(repo)}], {"repo": str(repo)}


def build_afce_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
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
    return [{"name": "afce_run_exp", "cmd": cmd, "cwd": str(repo)}], {"repo": str(repo)}


def build_lightrag_steps(repo_root, python_bin, global_cfg, exp):
    exp_name = exp["name"]
    cfg = exp.get("args", {})
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
    return [{"name": "lightrag_run", "cmd": cmd, "cwd": str(repo)}], {"repo": str(repo)}


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
        "llm_gt_baseline": build_llm_gt_steps,
        "pgraphrag_official": build_pgraphrag_steps,
        "afce_official": build_afce_steps,
        "lightrag_official": build_lightrag_steps,
        "custom": build_custom_steps,
    }

    for exp in exps:
        name = exp["name"]
        if only and name not in only:
            continue
        method = exp.get("method", "").strip()
        if method not in builders:
            raise ValueError(f"Unknown method '{method}' for experiment '{name}'.")

        print(f"[RUN] {name} ({method})")
        steps, metadata = builders[method](repo_root, args.python_bin, global_cfg, exp)

        ctx = {
            "repo_root": str(repo_root),
            "exp_name": name,
            "python_bin": args.python_bin,
            "run_root": str(run_root),
            "date": datetime.now().strftime("%Y%m%d"),
        }
        steps = render_placeholders(steps, ctx)

        for step in steps:
            run_step(step, dry_run=args.dry_run)

        manifest_path = run_root / name / "manifest.json"
        ensure_dir(manifest_path.parent)
        manifest = {
            "name": name,
            "method": method,
            "matrix": str(matrix_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "steps": steps,
            "metadata": metadata,
            "dry_run": bool(args.dry_run),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()


