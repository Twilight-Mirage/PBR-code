import os
import json
from pathlib import Path
from typing import Any, Dict

import project_settings


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def get_setting(name: str, default: Any = None) -> Any:
    value = getattr(project_settings, name, default)
    value = _clean(value)
    if isinstance(value, str) and value == "":
        return default
    return value


def _iter_unique(items):
    seen = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        yield item


def resolve_api_key(explicit_key: str = "", env_name: str = "OPENAI_API_KEY") -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    key_candidates = list(_iter_unique([env_name, "OPENAI_API_KEY", "DASHSCOPE_API_KEY"]))
    for key_name in key_candidates:
        env_value = os.getenv(key_name, "").strip()
        if env_value:
            return env_value
    for key_name in key_candidates:
        setting_value = get_setting(key_name, "")
        if isinstance(setting_value, str) and setting_value.strip():
            return setting_value.strip()
    return ""


def resolve_base_url(explicit_base: str = "", env_name: str = "OPENAI_BASE_URL") -> str:
    if explicit_base and explicit_base.strip():
        return explicit_base.strip()

    base_candidates = list(_iter_unique([env_name, "OPENAI_BASE_URL", "DASHSCOPE_BASE_URL"]))
    for key_name in base_candidates:
        env_value = os.getenv(key_name, "").strip()
        if env_value:
            return env_value
    for key_name in base_candidates:
        setting_value = get_setting(key_name, "")
        if isinstance(setting_value, str) and setting_value.strip():
            return setting_value.strip()
    return "https://api.openai.com/v1"


def resolve_organization(explicit_org: str = "", env_name: str = "OPENAI_ORGANIZATION") -> str:
    if explicit_org and explicit_org.strip():
        return explicit_org.strip()
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    return get_setting("OPENAI_ORGANIZATION", "")


def resolve_enable_thinking(explicit_enable: Any = None, env_name: str = "LLM_ENABLE_THINKING") -> bool:
    if isinstance(explicit_enable, bool):
        return explicit_enable
    if isinstance(explicit_enable, str) and explicit_enable.strip():
        return explicit_enable.strip().lower() in {"1", "true", "yes", "on"}

    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value.lower() in {"1", "true", "yes", "on"}

    setting_value = get_setting("LLM_ENABLE_THINKING", False)
    if isinstance(setting_value, bool):
        return setting_value
    if isinstance(setting_value, str):
        return setting_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(setting_value)


def resolve_default_retrieval_model_name(
    explicit_model_name: str = "",
    env_name: str = "RETRIEVAL_MODEL_NAME",
) -> str:
    if explicit_model_name and explicit_model_name.strip():
        return explicit_model_name.strip()
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    return get_setting("RETRIEVAL_MODEL_NAME", "multi-qa-MiniLM-L6-cos-v1")


def resolve_default_embedding_task(
    explicit_task: str = "",
    env_name: str = "EMBEDDING_TASK",
) -> str:
    if explicit_task and explicit_task.strip():
        return explicit_task.strip()
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    return get_setting("EMBEDDING_TASK", "auto")


def default_longmemeval_input(data_type: str = "s") -> str:
    by_split: Dict[str, str] = get_setting("LONGMEMEVAL_INPUT_BY_SPLIT", {}) or {}
    configured = None
    if isinstance(by_split, dict):
        value = by_split.get(data_type)
        if isinstance(value, str) and value.strip():
            configured = value.strip()

    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(
        [
            f"./data/longmemeval_data/longmemeval_{data_type}.json",
            f"./data/longmemeval_data/longmemeval_{data_type}_cleaned.json",
        ]
    )

    for path_str in candidates:
        if Path(path_str).exists():
            return path_str
    return candidates[0]


def _looks_like_jina_v5_model(model_name_or_path: str) -> bool:
    model_ref = str(model_name_or_path).strip().lower()
    if "jina-embeddings-v5" in model_ref:
        return True

    p = Path(model_name_or_path)
    if p.exists() and p.is_dir():
        if (p / "configuration_jina_embeddings_v5.py").exists() or (p / "modeling_jina_embeddings_v5.py").exists():
            return True
        cfg_path = p / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                model_type = str(cfg.get("model_type", "")).lower()
                if "jina" in model_type:
                    return True
            except Exception:
                pass
    return False


def resolve_embedding_task(model_name_or_path: str, embedding_task: str = "auto") -> str:
    task = (embedding_task or "").strip().lower()
    if task and task != "auto":
        return task
    if _looks_like_jina_v5_model(model_name_or_path):
        return "retrieval"
    return ""


def load_sentence_transformer(model_name_or_path: str, embedding_task: str = "auto", trust_remote_code: bool = True):
    from sentence_transformers import SentenceTransformer

    resolved_task = resolve_embedding_task(model_name_or_path=model_name_or_path, embedding_task=embedding_task)
    if resolved_task:
        print(f"[embedding] loading with task='{resolved_task}' from {model_name_or_path}")
        return SentenceTransformer(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            model_kwargs={"default_task": resolved_task},
        )
    return SentenceTransformer(model_name_or_path, trust_remote_code=trust_remote_code)
