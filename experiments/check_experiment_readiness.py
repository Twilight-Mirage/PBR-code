import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.project_runtime import default_longmemeval_input, resolve_api_key as resolve_project_api_key
from src.retrieval.evidence_source import infer_dataset_name, is_natural_oracle_dataset


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p


class ReadinessReport:
    def __init__(self):
        self.blockers = []
        self.warnings = []

    def blocker(self, msg: str):
        self.blockers.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def print(self):
        if not self.blockers and not self.warnings:
            print("[OK] no blockers or warnings.")
            return
        for m in self.blockers:
            print(f"[BLOCKER] {m}")
        for m in self.warnings:
            print(f"[WARN] {m}")


def _resolve_key(local_cfg, global_cfg, key_name, key_env_name, default_env):
    env_name = local_cfg.get(key_env_name, global_cfg.get(key_env_name, default_env))
    key = local_cfg.get(key_name, global_cfg.get(key_name, os.getenv(env_name, "")))
    if not key:
        key = resolve_project_api_key(env_name=env_name)
    return (key or "").strip(), env_name


def _check_in_file(global_cfg, repo_root: Path, report: ReadinessReport):
    if global_cfg.get("in_file"):
        in_path = _resolve_path(global_cfg["in_file"], repo_root)
    else:
        data_type = global_cfg.get("data_type", "s")
        in_path = _resolve_path(default_longmemeval_input(data_type=data_type), repo_root)
    if not in_path.exists():
        report.blocker(f"missing input file: {in_path}")
    return in_path


def _check_prototype_bank(global_cfg, exp_cfg, repo_root: Path, report: ReadinessReport):
    local_args = exp_cfg.get("args", {})
    pb = (
        local_args.get("cold_start_prototype_bank")
        or exp_cfg.get("cold_start_prototype_bank")
        or global_cfg.get("cold_start_prototype_bank")
    )
    if pb:
        pb_path = _resolve_path(str(pb), repo_root)
        if not pb_path.exists():
            report.blocker(f"{exp_cfg.get('name', '<exp>')}: missing cold-start prototype bank: {pb_path}")
    else:
        report.blocker(f"{exp_cfg.get('name', '<exp>')}: cold_start_router enabled but no cold_start_prototype_bank provided.")


def _check_explicit_encoder(global_cfg, exp_cfg, repo_root: Path, report: ReadinessReport):
    local_args = exp_cfg.get("args", {})
    ckpt = local_args.get("explicit_encoder_ckpt") or global_cfg.get("explicit_encoder_ckpt")
    if ckpt:
        ckpt_path = _resolve_path(str(ckpt), repo_root)
        if not ckpt_path.exists():
            report.blocker(f"{exp_cfg.get('name', '<exp>')}: explicit encoder checkpoint not found: {ckpt_path}")
    else:
        report.warn(
            f"{exp_cfg.get('name', '<exp>')}: explicit_profile enabled without explicit_encoder_ckpt (will use feature vectors without trained projector)."
        )


def check_retrieval_matrix(matrix_path: Path, repo_root: Path, report: ReadinessReport):
    matrix = _load_json(matrix_path)
    global_cfg = matrix.get("global", {})
    exps = matrix.get("experiments", [])

    _check_in_file(global_cfg, repo_root, report)

    for exp in exps:
        args = exp.get("args", {})
        if bool(exp.get("cold_start_router", False)):
            _check_prototype_bank(global_cfg, exp, repo_root, report)
        if bool(exp.get("explicit_profile", False)):
            _check_explicit_encoder(global_cfg, exp, repo_root, report)

        llm_key, llm_env = _resolve_key(args, global_cfg, "llm_api_key", "llm_api_key_env", "OPENAI_API_KEY")
        if not llm_key:
            report.blocker(f"{exp.get('name', '<exp>')}: missing retrieval LLM key (set args.llm_api_key, env {llm_env}, or project_settings.py).")


def check_dua_e2e_matrix(matrix_path: Path, repo_root: Path, report: ReadinessReport):
    matrix = _load_json(matrix_path)
    global_cfg = matrix.get("global", {})
    exps = matrix.get("experiments", [])

    default_in_path = _check_in_file(global_cfg, repo_root, report)

    run_eval_global = bool(global_cfg.get("run_eval", False))
    default_ref_json = _resolve_path(global_cfg.get("ref_json", str(default_in_path)), repo_root)
    if run_eval_global and not default_ref_json.exists():
        report.blocker(f"run_eval=true but ref_json not found: {default_ref_json}")

    for exp in exps:
        args = exp.get("args", {})
        exp_name = exp.get("name", "<exp>")

        in_file = args.get("in_file", str(default_in_path))
        cur_in_path = _resolve_path(str(in_file), repo_root)
        if not cur_in_path.exists():
            report.blocker(f"{exp_name}: input file not found: {cur_in_path}")

        if bool(exp.get("cold_start_router", False)):
            _check_prototype_bank(global_cfg, exp, repo_root, report)
        if bool(exp.get("explicit_profile", False)):
            _check_explicit_encoder(global_cfg, exp, repo_root, report)

        llm_key, llm_env = _resolve_key(args, global_cfg, "llm_api_key", "llm_api_key_env", "OPENAI_API_KEY")
        if not llm_key:
            report.blocker(f"{exp_name}: missing retrieval LLM key (set args.llm_api_key, env {llm_env}, or project_settings.py).")

        openai_key, openai_env = _resolve_key(args, global_cfg, "openai_key", "openai_key_env", "OPENAI_API_KEY")
        if not openai_key:
            report.warn(
                f"{exp_name}: generation openai_key unresolved (env {openai_env} or project_settings.py); only OK if model endpoint allows EMPTY key."
            )

        run_eval_cur = bool(args.get("run_eval", run_eval_global))
        cur_ref_json = _resolve_path(str(args.get("ref_json", global_cfg.get("ref_json", str(cur_in_path)))), repo_root)
        if run_eval_cur and not cur_ref_json.exists():
            report.blocker(f"{exp_name}: eval enabled but ref_json missing: {cur_ref_json}")


def _require_path(cfg, key, repo_root, report, prefix):
    if key not in cfg or cfg[key] in (None, ""):
        report.blocker(f"{prefix}: missing required field '{key}'.")
        return None
    p = _resolve_path(str(cfg[key]), repo_root)
    if not p.exists():
        report.blocker(f"{prefix}: path does not exist for '{key}': {p}")
    return p



def _check_unified_eval_requirements(prefix, method, args, global_cfg, repo_root, report):
    if not bool(args.get("unified_eval", global_cfg.get("unified_eval", False))):
        return

    ref_json = args.get("ref_json", global_cfg.get("ref_json", global_cfg.get("input_json", global_cfg.get("in_file"))))
    if not ref_json:
        report.blocker(f"{prefix}: unified_eval=true but ref_json is missing.")
    else:
        rp = _resolve_path(str(ref_json), repo_root)
        if not rp.exists():
            report.blocker(f"{prefix}: unified_eval ref_json not found: {rp}")

    adapter_pred_file = args.get("adapter_pred_file")
    if adapter_pred_file:
        pp = _resolve_path(str(adapter_pred_file), repo_root)
        if not pp.exists():
            report.blocker(f"{prefix}: adapter_pred_file not found: {pp}")
        return

    if method == "lightrag_official":
        report.blocker(f"{prefix}: unified_eval for LightRAG requires args.adapter_pred_file (no stable official benchmark output path).")
        return

    if method == "pgraphrag_official":
        modes = args.get("mode") if isinstance(args.get("mode"), list) else []
        ks = args.get("k") if isinstance(args.get("k"), list) else []
        if len(modes) != 1 or len(ks) != 1:
            report.blocker(f"{prefix}: unified_eval without adapter_pred_file requires exactly one mode and one k for auto output inference.")

def check_baseline_matrix(matrix_path: Path, repo_root: Path, report: ReadinessReport):
    matrix = _load_json(matrix_path)
    global_cfg = matrix.get("global", {})
    exps = matrix.get("experiments", [])

    for exp in exps:
        name = exp.get("name", "<exp>")
        method = exp.get("method", "")
        args = exp.get("args", {})
        prefix = f"{name} ({method})"

        if method == "naive_rag":
            input_json = args.get("input_json", global_cfg.get("input_json"))
            if not input_json:
                report.blocker(f"{prefix}: missing input_json.")
            else:
                p = _resolve_path(str(input_json), repo_root)
                if not p.exists():
                    report.blocker(f"{prefix}: input_json not found: {p}")
            if bool(args.get("run_eval", global_cfg.get("run_eval", False))):
                ref_json = args.get("ref_json", global_cfg.get("ref_json", input_json))
                rp = _resolve_path(str(ref_json), repo_root)
                if not rp.exists():
                    report.blocker(f"{prefix}: eval enabled but ref_json not found: {rp}")
            key, env_name = _resolve_key(args, global_cfg, "openai_key", "openai_key_env", "OPENAI_API_KEY")
            if not key:
                report.warn(f"{prefix}: openai_key unresolved (env {env_name} or project_settings.py); generation may fail unless endpoint accepts EMPTY key.")
            continue

        if method == "history_rag":
            input_json = args.get("input_json", global_cfg.get("input_json"))
            if not input_json:
                report.blocker(f"{prefix}: missing input_json.")
            else:
                p = _resolve_path(str(input_json), repo_root)
                if not p.exists():
                    report.blocker(f"{prefix}: input_json not found: {p}")
            key, env_name = _resolve_key(args, global_cfg, "openai_key", "openai_key_env", "OPENAI_API_KEY")
            if not key:
                report.warn(f"{prefix}: openai_key unresolved (env {env_name} or project_settings.py); generation may fail unless endpoint accepts EMPTY key.")
            continue
        if method in {"pbr_pipeline", "dua_rag"}:
            input_json = args.get("input_json", args.get("in_file", global_cfg.get("input_json", global_cfg.get("in_file"))))
            if not input_json:
                data_type = global_cfg.get("data_type", "s")
                input_json = default_longmemeval_input(data_type=data_type)
            p = _resolve_path(str(input_json), repo_root)
            if not p.exists():
                report.blocker(f"{prefix}: input_json/in_file not found: {p}")

            if method == "dua_rag":
                if bool(args.get("cold_start_router", True)):
                    _check_prototype_bank(global_cfg, {"name": name, "args": args}, repo_root, report)
                if bool(args.get("explicit_profile", True)):
                    _check_explicit_encoder(global_cfg, {"name": name, "args": args}, repo_root, report)
            else:
                if bool(args.get("cold_start_router", False)):
                    _check_prototype_bank(global_cfg, {"name": name, "args": args}, repo_root, report)
                if bool(args.get("explicit_profile", False)):
                    _check_explicit_encoder(global_cfg, {"name": name, "args": args}, repo_root, report)

            llm_key, llm_env = _resolve_key(args, global_cfg, "llm_api_key", "llm_api_key_env", "OPENAI_API_KEY")
            if not llm_key:
                report.blocker(f"{prefix}: missing retrieval LLM key (set llm_api_key, env {llm_env}, or project_settings.py).")

            key, env_name = _resolve_key(args, global_cfg, "openai_key", "openai_key_env", "OPENAI_API_KEY")
            if not key:
                report.warn(f"{prefix}: openai_key unresolved (env {env_name} or project_settings.py); generation may fail unless endpoint accepts EMPTY key.")

            if bool(args.get("run_eval", global_cfg.get("run_eval", False))):
                ref_json = args.get("ref_json", global_cfg.get("ref_json", input_json))
                rp = _resolve_path(str(ref_json), repo_root)
                if not rp.exists():
                    report.blocker(f"{prefix}: eval enabled but ref_json not found: {rp}")
            continue
        if method == "llm_gt_baseline":
            input_json = args.get("input_json", global_cfg.get("input_json"))
            if not input_json:
                report.blocker(f"{prefix}: missing input_json.")
                input_path = None
            else:
                input_path = _resolve_path(str(input_json), repo_root)
                if not input_path.exists():
                    report.blocker(f"{prefix}: input_json not found: {input_path}")

            evidence_mode = str(args.get("evidence_mode", global_cfg.get("evidence_mode", "retrieved"))).strip().lower()
            if evidence_mode not in {"retrieved", "oracle"}:
                report.blocker(f"{prefix}: evidence_mode must be retrieved/oracle, got '{evidence_mode}'.")

            granularity = str(args.get("granularity", global_cfg.get("granularity", "session"))).strip().lower()
            if granularity not in {"session", "turn"}:
                report.blocker(f"{prefix}: granularity must be session/turn, got '{granularity}'.")

            if evidence_mode == "retrieved" and args.get("retrieved_json"):
                rp = _resolve_path(str(args.get("retrieved_json")), repo_root)
                if not rp.exists():
                    report.blocker(f"{prefix}: retrieved_json not found: {rp}")

            if evidence_mode == "oracle":
                dataset_name = str(args.get("dataset_name", global_cfg.get("dataset_name", ""))).strip()
                inferred = infer_dataset_name(dataset_name, str(input_path or input_json))
                if (not is_natural_oracle_dataset(inferred)) and input_path and input_path.exists():
                    try:
                        data_obj = _load_json(input_path)
                        sample0 = data_obj[0] if isinstance(data_obj, list) and data_obj else None
                    except Exception:
                        sample0 = None
                    inferred = infer_dataset_name(dataset_name, str(input_path), sample0)
                if not is_natural_oracle_dataset(inferred):
                    report.blocker(
                        f"{prefix}: oracle mode is limited to natural-oracle datasets (LongMemEval/PersonaBench), got '{inferred or 'unknown'}'."
                    )

            if bool(args.get("run_eval", global_cfg.get("run_eval", False))):
                ref_json = args.get("ref_json", global_cfg.get("ref_json", input_json))
                rp = _resolve_path(str(ref_json), repo_root)
                if not rp.exists():
                    report.blocker(f"{prefix}: eval enabled but ref_json not found: {rp}")

            key, env_name = _resolve_key(args, global_cfg, "openai_key", "openai_key_env", "OPENAI_API_KEY")
            if not key:
                report.warn(f"{prefix}: openai_key unresolved (env {env_name} or project_settings.py); generation may fail unless endpoint accepts EMPTY key.")
            continue

        if method == "pgraphrag_official":
            _check_unified_eval_requirements(prefix, method, args, global_cfg, repo_root, report)
            repo = _resolve_path(
                str(args.get("repo_path", global_cfg.get("pgraphrag_repo", "./third_party_baselines/PGraphRAG"))), repo_root
            )
            if not repo.exists():
                report.blocker(f"{prefix}: repo_path not found: {repo}")
            _require_path(args, "input", repo_root, report, prefix)
            continue

        if method == "afce_official":
            _check_unified_eval_requirements(prefix, method, args, global_cfg, repo_root, report)
            repo = _resolve_path(
                str(args.get("repo_path", global_cfg.get("afce_repo", "./third_party_baselines/AP-Bots"))), repo_root
            )
            if not repo.exists():
                report.blocker(f"{prefix}: repo_path not found: {repo}")
            if not args.get("dataset"):
                report.blocker(f"{prefix}: missing dataset.")
            key, env_name = _resolve_key(args, global_cfg, "openai_key", "openai_key_env", "OPENAI_API_KEY")
            if not key:
                report.warn(f"{prefix}: openai_key unresolved (env {env_name} or project_settings.py); AF+CE official run usually needs it.")
            continue

        if method == "lightrag_official":
            _check_unified_eval_requirements(prefix, method, args, global_cfg, repo_root, report)
            repo = _resolve_path(
                str(args.get("repo_path", global_cfg.get("lightrag_repo", "./third_party_baselines/LightRAG"))), repo_root
            )
            if not repo.exists():
                report.blocker(f"{prefix}: repo_path not found: {repo}")
            if not args.get("command"):
                entry_script = args.get("entry_script", "examples/lightrag_openai_demo.py")
                entry_path = (repo / entry_script).resolve()
                if not entry_path.exists():
                    report.blocker(f"{prefix}: entry_script not found: {entry_path}")
            key, env_name = _resolve_key(args, global_cfg, "openai_key", "openai_key_env", "OPENAI_API_KEY")
            if not key:
                report.warn(f"{prefix}: openai_key unresolved (env {env_name} or project_settings.py); LightRAG OpenAI demo needs it.")
            continue

        if method == "custom":
            commands = args.get("commands", [])
            if not isinstance(commands, list) or len(commands) == 0:
                report.blocker(f"{prefix}: custom method requires non-empty args.commands.")
            continue

        report.blocker(f"{prefix}: unknown method '{method}'.")


def main():
    parser = argparse.ArgumentParser(description="Check whether experiment matrix config is ready to run.")
    parser.add_argument("--kind", type=str, required=True, choices=["retrieval", "baseline", "dua"])
    parser.add_argument("--matrix", type=str, required=True, help="Matrix json path.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    matrix_path = _resolve_path(args.matrix, repo_root)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    report = ReadinessReport()
    if args.kind == "retrieval":
        check_retrieval_matrix(matrix_path, repo_root, report)
    elif args.kind == "baseline":
        check_baseline_matrix(matrix_path, repo_root, report)
    else:
        check_dua_e2e_matrix(matrix_path, repo_root, report)

    report.print()
    if report.blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
