import argparse
import json
import os
import subprocess
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


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_subprocess_env():
    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        project_key = resolve_project_api_key(env_name="OPENAI_API_KEY")
        if project_key:
            env["OPENAI_API_KEY"] = project_key
    return env


def add_cli_arg(cmd, key, value):
    flag = f"--{key}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def run_step(step, dry_run=False):
    print(f"[STEP] {step['name']}")
    print(" ".join(step["cmd"]))
    print(f"[CWD]  {step['cwd']}")
    if not dry_run:
        subprocess.run(step["cmd"], cwd=step["cwd"], check=True, env=build_subprocess_env())


def resolve_key(cfg_local, cfg_global, key_local="openai_key", key_env_local="openai_key_env", default_env="OPENAI_API_KEY"):
    env_name = cfg_local.get(key_env_local, cfg_global.get(key_env_local, default_env))
    key = cfg_local.get(key_local, os.getenv(env_name, ""))
    if not key:
        key = resolve_project_api_key(env_name=env_name)
    return key, env_name


def build_retrieval_cmd(repo_root, python_bin, global_cfg, exp_cfg, retrieval_in, retrieval_out):
    cmd = [python_bin, "-m", "src.retrieval.retrieval_PBR"]

    local_args = exp_cfg.get("args", {})
    retrieval_model_name = local_args.get(
        "retrieval_model_name",
        global_cfg.get("retrieval_model_name", resolve_default_retrieval_model_name()),
    )
    embedding_task = local_args.get(
        "embedding_task",
        global_cfg.get("embedding_task", resolve_default_embedding_task()),
    )

    add_cli_arg(cmd, "model_type", exp_cfg.get("model_type", "PBR"))
    add_cli_arg(cmd, "data_type", global_cfg.get("data_type", "s"))
    add_cli_arg(cmd, "retrieval_model_name", retrieval_model_name)
    add_cli_arg(cmd, "embedding_task", embedding_task)
    add_cli_arg(cmd, "in_file", str(retrieval_in))
    add_cli_arg(cmd, "out_file", str(retrieval_out))

    if bool(exp_cfg.get("temporal_profile", False)):
        cmd.append("--temporal_profile")
    if bool(exp_cfg.get("cold_start_router", False)):
        cmd.append("--cold_start_router")
    if bool(exp_cfg.get("explicit_profile", False)):
        cmd.append("--explicit_profile")

    save_suffix = exp_cfg.get("save_suffix")
    if not save_suffix:
        save_suffix = f"_exp_{exp_cfg['name']}"
    add_cli_arg(cmd, "save_suffix", save_suffix)

    key, key_env = resolve_key(local_args, global_cfg, key_local="llm_api_key", key_env_local="llm_api_key_env")
    if key:
        add_cli_arg(cmd, "llm_api_key", key)
    add_cli_arg(cmd, "llm_api_key_env", local_args.get("llm_api_key_env", global_cfg.get("llm_api_key_env", key_env)))
    add_cli_arg(cmd, "llm_base_url", local_args.get("llm_base_url", global_cfg.get("llm_base_url", "")))
    add_cli_arg(cmd, "llm_model", local_args.get("llm_model", global_cfg.get("llm_model", "gpt-4o-mini")))
    add_cli_arg(cmd, "llm_max_tokens", local_args.get("llm_max_tokens", global_cfg.get("llm_max_tokens", 512)))
    add_cli_arg(cmd, "llm_temperature", local_args.get("llm_temperature", global_cfg.get("llm_temperature", 0.0)))
    add_cli_arg(cmd, "llm_enable_thinking", local_args.get("llm_enable_thinking", global_cfg.get("llm_enable_thinking", False)))
    add_cli_arg(cmd, "llm_extra_body_json", local_args.get("llm_extra_body_json", global_cfg.get("llm_extra_body_json", None)))

    reserved_global = {
        "run_root",
        "in_file",
        "ref_json",
        "run_eval",
        "eval_model",
        "data_type",
        "retrieval_model_name",
        "embedding_task",
        "openai_key",
        "openai_key_env",
        "openai_base_url",
        "openai_organization",
        "openai_enable_thinking",
        "openai_extra_body_json",
        "gen_model_name",
        "gen_model_alias",
        "gen_length",
        "history_format",
        "useronly",
        "cot",
        "con",
        "topk_context",
        "retriever_type",
        "merge_key_expansion_into_value",
        "llm_model",
        "llm_api_key",
        "llm_api_key_env",
        "llm_base_url",
        "llm_max_tokens",
        "llm_temperature",
        "llm_enable_thinking",
        "llm_extra_body_json",
    }
    for gk, gv in global_cfg.items():
        if gk in reserved_global:
            continue
        add_cli_arg(cmd, gk, gv)

    for k, v in local_args.items():
        if k in {
            "openai_key",
            "openai_key_env",
            "openai_base_url",
            "openai_organization",
            "openai_enable_thinking",
            "openai_extra_body_json",
            "gen_model_name",
            "gen_model_alias",
            "gen_length",
            "history_format",
            "useronly",
            "cot",
            "con",
            "topk_context",
            "retriever_type",
            "merge_key_expansion_into_value",
            "llm_api_key",
            "llm_api_key_env",
            "llm_base_url",
            "llm_model",
            "llm_max_tokens",
            "llm_temperature",
            "llm_enable_thinking",
            "llm_extra_body_json",
            "retrieval_model_name",
            "embedding_task",
        }:
            continue
        add_cli_arg(cmd, k, v)

    return cmd


def build_generation_cmd(repo_root, python_bin, global_cfg, exp_cfg, retrieval_out, generation_out, run_dir):
    local_args = exp_cfg.get("args", {})

    model_name = local_args.get("gen_model_name", global_cfg.get("gen_model_name"))
    if not model_name:
        raise ValueError(f"Experiment '{exp_cfg['name']}' missing gen_model_name (exp args or global).")
    model_alias = local_args.get("gen_model_alias", global_cfg.get("gen_model_alias", model_name))

    openai_key, _ = resolve_key(local_args, global_cfg)
    if not openai_key:
        openai_key = "EMPTY"

    cmd = [
        python_bin,
        "-m",
        "src.generation.run_generation",
        "--in_file",
        str(retrieval_out),
        "--out_dir",
        str(run_dir),
        "--out_file",
        str(generation_out),
        "--model_name",
        str(model_name),
        "--model_alias",
        str(model_alias),
        "--openai_key",
        str(openai_key),
        "--retriever_type",
        str(local_args.get("retriever_type", global_cfg.get("retriever_type", "flat-session"))),
        "--topk_context",
        str(local_args.get("topk_context", global_cfg.get("topk_context", 10))),
        "--history_format",
        str(local_args.get("history_format", global_cfg.get("history_format", "json"))),
        "--useronly",
        str(local_args.get("useronly", global_cfg.get("useronly", "false"))).lower(),
        "--cot",
        str(local_args.get("cot", global_cfg.get("cot", "false"))).lower(),
        "--con",
        str(local_args.get("con", global_cfg.get("con", "false"))).lower(),
        "--merge_key_expansion_into_value",
        str(local_args.get("merge_key_expansion_into_value", global_cfg.get("merge_key_expansion_into_value", "none"))),
    ]

    openai_base_url = local_args.get("openai_base_url", global_cfg.get("openai_base_url"))
    openai_org = local_args.get("openai_organization", global_cfg.get("openai_organization"))
    openai_enable_thinking = local_args.get("openai_enable_thinking", global_cfg.get("openai_enable_thinking"))
    openai_extra_body_json = local_args.get("openai_extra_body_json", global_cfg.get("openai_extra_body_json"))
    gen_length = local_args.get("gen_length", global_cfg.get("gen_length"))

    if openai_base_url:
        cmd.extend(["--openai_base_url", str(openai_base_url)])
    if openai_org:
        cmd.extend(["--openai_organization", str(openai_org)])
    if openai_enable_thinking is not None:
        cmd.extend(["--openai_enable_thinking", str(openai_enable_thinking).lower()])
    if openai_extra_body_json not in (None, ""):
        cmd.extend(["--openai_extra_body_json", str(openai_extra_body_json)])
    if gen_length is not None:
        cmd.extend(["--gen_length", str(gen_length)])

    return cmd


def build_eval_cmd(repo_root, python_bin, global_cfg, exp_cfg, generation_out, ref_json):
    eval_model = exp_cfg.get("args", {}).get("eval_model", global_cfg.get("eval_model", "gpt-4o-mini"))
    return [
        python_bin,
        "-m",
        "src.evaluation.evaluate_qa",
        str(eval_model),
        str(generation_out),
        str(ref_json),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run retrieval->generation->evaluation matrix for DUA-RAG ablations.")
    parser.add_argument("--matrix", type=str, required=True, help="Path to matrix json.")
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated experiment names.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8-sig"))
    global_cfg = matrix.get("global", {})
    experiments = matrix.get("experiments", [])

    in_file = global_cfg.get("in_file")
    if not in_file:
        in_file = default_longmemeval_input(global_cfg.get("data_type", "s"))
    in_path = Path(in_file)
    if not in_path.is_absolute():
        in_path = (repo_root / in_path).resolve()
    else:
        in_path = in_path.resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    ref_json = Path(global_cfg.get("ref_json", str(in_path)))
    if not ref_json.is_absolute():
        ref_json = (repo_root / ref_json).resolve()
    else:
        ref_json = ref_json.resolve()
    run_eval = bool(global_cfg.get("run_eval", False))

    run_root = Path(global_cfg.get("run_root", "./experiments/runs/dua_e2e")).resolve()
    ensure_dir(run_root)

    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    for exp in experiments:
        name = exp["name"]
        if only and name not in only:
            continue

        print(f"[RUN] {name}")
        run_dir = run_root / name
        ensure_dir(run_dir)

        retrieval_out = run_dir / f"{name}_retrieval.json"
        generation_out = run_dir / f"{name}_hypotheses.jsonl"

        retrieval_cmd = build_retrieval_cmd(
            repo_root=repo_root,
            python_bin=args.python_bin,
            global_cfg=global_cfg,
            exp_cfg=exp,
            retrieval_in=in_path,
            retrieval_out=retrieval_out,
        )
        generation_cmd = build_generation_cmd(
            repo_root=repo_root,
            python_bin=args.python_bin,
            global_cfg=global_cfg,
            exp_cfg=exp,
            retrieval_out=retrieval_out,
            generation_out=generation_out,
            run_dir=run_dir,
        )

        steps = [
            {"name": "retrieval", "cmd": retrieval_cmd, "cwd": str(repo_root)},
            {"name": "generation", "cmd": generation_cmd, "cwd": str(repo_root)},
        ]

        cur_run_eval = bool(exp.get("args", {}).get("run_eval", run_eval))
        if cur_run_eval:
            eval_cmd = build_eval_cmd(
                repo_root=repo_root,
                python_bin=args.python_bin,
                global_cfg=global_cfg,
                exp_cfg=exp,
                generation_out=generation_out,
                ref_json=ref_json,
            )
            steps.append({"name": "eval", "cmd": eval_cmd, "cwd": str(repo_root)})

        for step in steps:
            run_step(step, dry_run=args.dry_run)

        manifest = {
            "name": name,
            "matrix": str(matrix_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "retrieval_output": str(retrieval_out),
            "generation_output": str(generation_out),
            "ref_json": str(ref_json),
            "steps": steps,
            "dry_run": bool(args.dry_run),
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()
