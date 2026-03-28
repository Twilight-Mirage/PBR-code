import argparse
import json
import os
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


def add_cli_arg(cmd, key, value):
    flag = f"--{key}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


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
    wrapped_step = dict(step)

    if logger is None:
        print(f"[STEP] {wrapped_step['name']}")
        print(" ".join(wrapped_step["cmd"]))
        print(f"[CWD]  {wrapped_step['cwd']}")
        if not dry_run:
            subprocess.run(wrapped_step["cmd"], cwd=wrapped_step["cwd"], check=True, env=build_subprocess_env())
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

    default_in_file = global_cfg.get("in_file")
    if not default_in_file:
        default_in_file = default_longmemeval_input(global_cfg.get("data_type", "s"))
    default_in_path = Path(default_in_file)
    if not default_in_path.is_absolute():
        default_in_path = (repo_root / default_in_path).resolve()
    else:
        default_in_path = default_in_path.resolve()
    if not default_in_path.exists():
        raise FileNotFoundError(f"Default input file not found: {default_in_path}")

    default_ref_json = Path(global_cfg.get("ref_json", str(default_in_path)))
    if not default_ref_json.is_absolute():
        default_ref_json = (repo_root / default_ref_json).resolve()
    else:
        default_ref_json = default_ref_json.resolve()
    run_eval = bool(global_cfg.get("run_eval", False))

    run_root = Path(global_cfg.get("run_root", "./experiments/runs/dua_e2e")).resolve()
    ensure_dir(run_root)

    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    selected_experiments = [exp for exp in experiments if (not only or exp["name"] in only)]

    logger = MatrixRunLogger(
        run_root=run_root,
        matrix_path=matrix_path,
        runner_name="dua_e2e_matrix",
        dry_run=args.dry_run,
    )
    logger.info(f"[MATRIX] selected_experiments={len(selected_experiments)} run_root={run_root}")
    logger.event(
        "matrix_plan",
        total_experiments=len(selected_experiments),
        experiment_names=[exp["name"] for exp in selected_experiments],
    )

    matrix_status = "success"
    matrix_error = ""
    try:
        for exp_index, exp in enumerate(selected_experiments, 1):
            name = exp["name"]
            logger.info(f"[RUN] {name} [{exp_index}/{len(selected_experiments)}]")
            logger.event(
                "experiment_start",
                experiment=name,
                experiment_index=exp_index,
                total_experiments=len(selected_experiments),
            )

            exp_t0 = time.time()
            status = "dry_run" if args.dry_run else "success"
            error = ""
            step_results = []
            steps = []
            cur_in_path = None
            cur_ref_json = None
            retrieval_out = None
            generation_out = None

            try:
                run_dir = run_root / name
                ensure_dir(run_dir)

                retrieval_out = run_dir / f"{name}_retrieval.json"
                generation_out = run_dir / f"{name}_hypotheses.jsonl"

                local_args = exp.get("args", {})
                cur_in_value = local_args.get("in_file", str(default_in_path))
                cur_in_path = Path(cur_in_value)
                if not cur_in_path.is_absolute():
                    cur_in_path = (repo_root / cur_in_path).resolve()
                else:
                    cur_in_path = cur_in_path.resolve()
                if not cur_in_path.exists():
                    raise FileNotFoundError(f"Experiment '{name}' input file not found: {cur_in_path}")

                cur_ref_value = local_args.get("ref_json", global_cfg.get("ref_json", str(cur_in_path)))
                cur_ref_json = Path(cur_ref_value)
                if not cur_ref_json.is_absolute():
                    cur_ref_json = (repo_root / cur_ref_json).resolve()
                else:
                    cur_ref_json = cur_ref_json.resolve()

                retrieval_cmd = build_retrieval_cmd(
                    repo_root=repo_root,
                    python_bin=args.python_bin,
                    global_cfg=global_cfg,
                    exp_cfg=exp,
                    retrieval_in=cur_in_path,
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

                cur_run_eval = bool(local_args.get("run_eval", run_eval))
                if cur_run_eval:
                    eval_cmd = build_eval_cmd(
                        repo_root=repo_root,
                        python_bin=args.python_bin,
                        global_cfg=global_cfg,
                        exp_cfg=exp,
                        generation_out=generation_out,
                        ref_json=cur_ref_json,
                    )
                    steps.append({"name": "eval", "cmd": eval_cmd, "cwd": str(repo_root)})

                for step_index, step in enumerate(steps, 1):
                    result = run_step(
                        step,
                        dry_run=args.dry_run,
                        logger=logger,
                        exp_name=name,
                        exp_index=exp_index,
                        exp_total=len(selected_experiments),
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
                logger.event("experiment_error", experiment=name, error=error)
                raise
            finally:
                exp_elapsed = round(time.time() - exp_t0, 3)
                run_dir = run_root / name
                steps_redacted = []
                for _s in steps:
                    _item = dict(_s)
                    _cmd = _item.get("cmd")
                    if isinstance(_cmd, list):
                        _item["cmd"] = redact_command(_cmd)
                    steps_redacted.append(_item)

                manifest = {
                    "name": name,
                    "matrix": str(matrix_path),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "status": status,
                    "error": error,
                    "duration_sec": exp_elapsed,
                    "in_file": str(cur_in_path) if cur_in_path else "",
                    "retrieval_output": str(retrieval_out) if retrieval_out else "",
                    "generation_output": str(generation_out) if generation_out else "",
                    "ref_json": str(cur_ref_json) if cur_ref_json else "",
                    "steps": steps_redacted,
                    "step_results": step_results,
                    "dry_run": bool(args.dry_run),
                    "log_dir": str(logger.log_dir),
                    "run_log": str(logger.text_log),
                    "events_log": str(logger.events_log),
                    "experiment_log_dir": str(logger.experiment_log_dir(name)),
                }
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"[MANIFEST] {manifest_path}")
                logger.event(
                    "experiment_end",
                    experiment=name,
                    status=status,
                    error=error,
                    duration_sec=exp_elapsed,
                    manifest=str(manifest_path),
                )
    finally:
        logger.finalize(status=matrix_status, error=matrix_error)

if __name__ == "__main__":
    main()
