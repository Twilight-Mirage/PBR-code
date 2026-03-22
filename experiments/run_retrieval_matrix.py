import argparse
import json
import subprocess
import sys
from pathlib import Path


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
        str(repo_root / "src" / "retrieval" / "retrieval_PBR.py"),
    ]
    add_cli_arg(cmd, "model_type", exp_cfg.get("model_type", "PBR"))
    add_cli_arg(cmd, "data_type", global_cfg.get("data_type", "s"))
    add_cli_arg(cmd, "retrieval_model_name", global_cfg.get("retrieval_model_name", "multi-qa-MiniLM-L6-cos-v1"))

    for gk, gv in global_cfg.items():
        if gk in {"data_type", "retrieval_model_name"}:
            continue
        add_cli_arg(cmd, gk, gv)

    if bool(exp_cfg.get("temporal_profile", False)):
        cmd.append("--temporal_profile")
    if bool(exp_cfg.get("cold_start_router", False)):
        cmd.append("--cold_start_router")

    save_suffix = exp_cfg.get("save_suffix")
    if not save_suffix:
        save_suffix = f"_exp_{exp_cfg['name']}"
    add_cli_arg(cmd, "save_suffix", save_suffix)

    for k, v in exp_cfg.get("args", {}).items():
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

    for exp in exps:
        name = exp["name"]
        if only and name not in only:
            continue
        cmd = build_command(args.python_bin, repo_root, global_cfg, exp)
        print(f"[RUN] {name}")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
