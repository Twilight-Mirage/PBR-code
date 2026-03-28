import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.experiment_logging import MatrixRunLogger, redact_command, run_logged_step
from src.common.project_runtime import resolve_default_embedding_task, resolve_default_retrieval_model_name


def add_cli_arg(cmd, key, value):
    flag = f"--{key}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def build_command(python_bin, repo_root, global_cfg, exp_cfg):
    cmd = [
        python_bin,
        "-m",
        "src.retrieval.retrieval_PBR",
    ]
    exp_args = exp_cfg.get("args", {})
    retrieval_model_name = exp_args.get(
        "retrieval_model_name",
        global_cfg.get("retrieval_model_name", resolve_default_retrieval_model_name()),
    )
    embedding_task = exp_args.get(
        "embedding_task",
        global_cfg.get("embedding_task", resolve_default_embedding_task()),
    )

    add_cli_arg(cmd, "model_type", exp_cfg.get("model_type", "PBR"))
    add_cli_arg(cmd, "data_type", global_cfg.get("data_type", "s"))
    add_cli_arg(cmd, "retrieval_model_name", retrieval_model_name)
    add_cli_arg(cmd, "embedding_task", embedding_task)

    for gk, gv in global_cfg.items():
        if gk in {"data_type", "retrieval_model_name", "embedding_task", "run_root"}:
            continue
        add_cli_arg(cmd, gk, gv)

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

    for k, v in exp_args.items():
        if k in {"retrieval_model_name", "embedding_task"}:
            continue
        add_cli_arg(cmd, k, v)
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, required=True, help="Path to experiment matrix json.")
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable to use.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated experiment names to run.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without execution.")
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    matrix = json.loads(matrix_path.read_text(encoding="utf-8-sig"))
    global_cfg = matrix.get("global", {})
    exps = matrix.get("experiments", [])

    only = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    selected_exps = [exp for exp in exps if (not only or exp["name"] in only)]

    run_root = Path(global_cfg.get("run_root", "./experiments/runs/retrieval_matrix")).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    logger = MatrixRunLogger(
        run_root=run_root,
        matrix_path=matrix_path,
        runner_name="retrieval_matrix",
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
            cmd = build_command(args.python_bin, REPO_ROOT, global_cfg, exp)
            logger.info(f"[RUN] {name} [{exp_index}/{len(selected_exps)}]")
            logger.event(
                "experiment_start",
                experiment=name,
                experiment_index=exp_index,
                total_experiments=len(selected_exps),
            )
            t0 = time.time()
            status = "dry_run" if args.dry_run else "success"
            error = ""
            try:
                result = run_logged_step(
                    step={"name": "retrieval", "cmd": cmd, "cwd": str(REPO_ROOT)},
                    env=os.environ.copy(),
                    dry_run=args.dry_run,
                    logger=logger,
                    exp_name=name,
                    exp_index=exp_index,
                    exp_total=len(selected_exps),
                    step_index=1,
                    step_total=1,
                )
            except Exception as exc:
                status = "failed"
                error = repr(exc)
                matrix_status = "failed"
                matrix_error = error
                logger.event("experiment_error", experiment=name, error=error)
                raise
            finally:
                elapsed = round(time.time() - t0, 3)
                manifest_path = run_root / name / "manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "name": name,
                    "matrix": str(matrix_path),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "status": status,
                    "error": error,
                    "duration_sec": elapsed,
                    "cmd": redact_command(cmd),
                    "dry_run": bool(args.dry_run),
                    "step_result": result if status != "failed" else {},
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
                    status=status,
                    error=error,
                    duration_sec=elapsed,
                    manifest=str(manifest_path),
                )
    finally:
        logger.finalize(status=matrix_status, error=matrix_error)


if __name__ == "__main__":
    main()
