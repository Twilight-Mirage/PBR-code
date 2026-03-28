import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unnamed"


_SECRET_FLAGS = {
    "--openai_key",
    "--llm_api_key",
    "--api_key",
    "--token",
}


def redact_command(cmd: List[str]) -> List[str]:
    redacted = []
    mask_next = False
    for token in [str(x) for x in cmd]:
        lower = token.lower()
        if mask_next:
            redacted.append("***")
            mask_next = False
            continue
        if lower in _SECRET_FLAGS:
            redacted.append(token)
            mask_next = True
            continue
        if lower.startswith("--") and "key=" in lower:
            key_part = token.split("=", 1)[0]
            redacted.append(f"{key_part}=***")
            continue
        if token.startswith("sk-"):
            redacted.append("***")
            continue
        redacted.append(token)
    return redacted


class MatrixRunLogger:
    def __init__(self, run_root: Path, matrix_path: Path, runner_name: str, dry_run: bool):
        self.run_root = Path(run_root).resolve()
        self.matrix_path = Path(matrix_path).resolve()
        self.runner_name = runner_name
        self.dry_run = bool(dry_run)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_id = f"{self.matrix_path.stem}_{runner_name}_{ts}"
        self.log_dir = (self.run_root / "_logs" / self.session_id).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.text_log = self.log_dir / "run.log"
        self.events_log = self.log_dir / "events.jsonl"
        self._write_text(f"[START] runner={runner_name} dry_run={self.dry_run} matrix={self.matrix_path}")
        self.event(
            "matrix_start",
            matrix=str(self.matrix_path),
            runner=self.runner_name,
            dry_run=self.dry_run,
            log_dir=str(self.log_dir),
        )

    def _write_text(self, line: str):
        with self.text_log.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")

    def info(self, msg: str):
        print(msg)
        self._write_text(msg)

    def event(self, etype: str, **payload: Any):
        data = {"ts": _now_iso(), "type": etype}
        data.update(payload)
        with self.events_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def experiment_log_dir(self, exp_name: str) -> Path:
        p = self.log_dir / _safe_name(exp_name)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def step_log_file(self, exp_name: str, step_index: int, step_name: str) -> Path:
        exp_dir = self.experiment_log_dir(exp_name)
        return exp_dir / f"step{step_index:02d}_{_safe_name(step_name)}.log"

    def finalize(self, status: str, error: str = ""):
        self.event("matrix_end", status=status, error=error)
        self._write_text(f"[END] status={status} error={error}")


def run_logged_step(
    *,
    step: Dict[str, Any],
    env: Dict[str, str],
    dry_run: bool,
    logger: MatrixRunLogger,
    exp_name: str,
    exp_index: int,
    exp_total: int,
    step_index: int,
    step_total: int,
):
    cmd = step["cmd"]
    cwd = step.get("cwd")
    step_name = str(step.get("name", f"step{step_index}"))

    redacted_cmd = redact_command(cmd)
    progress = f"exp {exp_index}/{exp_total}, step {step_index}/{step_total}"
    logger.info(f"[PROGRESS] {progress}")
    logger.info(f"[STEP] {step_name}")
    logger.info("[CMD]  " + " ".join(redacted_cmd))
    if cwd:
        logger.info(f"[CWD]  {cwd}")

    step_log = logger.step_log_file(exp_name, step_index, step_name)
    t0 = time.time()
    logger.event(
        "step_start",
        experiment=exp_name,
        progress=progress,
        step=step_name,
        cmd=redacted_cmd,
        cwd=cwd,
        step_log=str(step_log),
    )

    if dry_run:
        elapsed = round(time.time() - t0, 3)
        logger.event(
            "step_end",
            experiment=exp_name,
            step=step_name,
            status="dry_run",
            duration_sec=elapsed,
            returncode=0,
            step_log=str(step_log),
        )
        logger.info(f"[STEP-END] {step_name} status=dry_run duration={elapsed:.3f}s")
        return {
            "status": "dry_run",
            "duration_sec": elapsed,
            "returncode": 0,
            "step_log": str(step_log),
        }

    with step_log.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fout.write(line)

    rc = proc.wait()
    elapsed = round(time.time() - t0, 3)
    status = "success" if rc == 0 else "failed"
    logger.event(
        "step_end",
        experiment=exp_name,
        step=step_name,
        status=status,
        duration_sec=elapsed,
        returncode=rc,
        step_log=str(step_log),
    )
    logger.info(f"[STEP-END] {step_name} status={status} duration={elapsed:.3f}s returncode={rc}")

    if rc != 0:
        raise subprocess.CalledProcessError(returncode=rc, cmd=cmd)

    return {
        "status": status,
        "duration_sec": elapsed,
        "returncode": rc,
        "step_log": str(step_log),
    }
