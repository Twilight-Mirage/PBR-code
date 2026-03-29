from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json


@dataclass
class EvalConfig:
    evidence_mode: str = "retrieved"  # "retrieved" or "oracle"
    top_k: int = 5
    max_context_tokens: int = 3000
    granularity: str = "turn"  # for LongMemEval: "turn" or "session"
    dataset_name: str = ""


NATURAL_ORACLE_DATASETS = {"longmemeval", "personabench"}


def normalize_dataset_name(name: str) -> str:
    return str(name or "").strip().lower()


def is_natural_oracle_dataset(name: str) -> bool:
    return normalize_dataset_name(name) in NATURAL_ORACLE_DATASETS


def infer_dataset_name(explicit_name: str, in_file: str, sample: Optional[Dict[str, Any]] = None) -> str:
    explicit = normalize_dataset_name(explicit_name)
    if explicit:
        return explicit

    path_lc = str(in_file or "").lower()
    if "longmemeval" in path_lc:
        return "longmemeval"
    if "personabench" in path_lc or "persona_bench" in path_lc:
        return "personabench"

    if isinstance(sample, dict):
        if "haystack_sessions" in sample and "answer_session_ids" in sample:
            return "longmemeval"
        if any(k in sample for k in ("relevant_chunks", "supporting_chunks", "oracle_evidence")):
            return "personabench"
    return ""


def _safe_text(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ("text", "content", "chunk", "evidence", "passage"):
            if key in x and isinstance(x[key], str):
                return x[key].strip()
    return ""


def _turn_ids_from_session_id(session_id: str, turn_index_1based: int) -> str:
    return f"{session_id}_{turn_index_1based}"


def _build_longmemeval_session_items(sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    items: List[Dict[str, Any]] = []
    correct_ids: List[str] = []

    session_ids = sample.get("haystack_session_ids", [])
    sessions = sample.get("haystack_sessions", [])
    dates = sample.get("haystack_dates", [])

    answer_sid_set = set(str(x) for x in sample.get("answer_session_ids", []) if x is not None)

    for idx, (sid, sess) in enumerate(zip(session_ids, sessions)):
        sid_str = str(sid)
        user_turns = []
        if isinstance(sess, list):
            for turn in sess:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    txt = str(turn.get("content", "")).strip()
                    if txt:
                        user_turns.append(txt)

        payload = {
            "date": dates[idx] if idx < len(dates) else None,
            "conversation": user_turns,
        }
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        items.append({"corpus_id": sid_str, "text": text})

        if answer_sid_set and sid_str in answer_sid_set:
            correct_ids.append(sid_str)

    if not correct_ids:
        for item in items:
            if "answer" in item["corpus_id"]:
                correct_ids.append(item["corpus_id"])

    return items, correct_ids


def _build_longmemeval_turn_items(sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    items: List[Dict[str, Any]] = []
    correct_ids: List[str] = []

    session_ids = sample.get("haystack_session_ids", [])
    sessions = sample.get("haystack_sessions", [])

    for sid, sess in zip(session_ids, sessions):
        sid_str = str(sid)
        if not isinstance(sess, list):
            continue
        for i_turn, turn in enumerate(sess, start=1):
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            txt = str(turn.get("content", "")).strip()
            if not txt:
                continue
            tid = _turn_ids_from_session_id(sid_str, i_turn)
            items.append({"corpus_id": tid, "text": txt})
            if bool(turn.get("has_answer", False)):
                correct_ids.append(tid)

    if not correct_ids:
        for item in items:
            if "answer" in item["corpus_id"]:
                correct_ids.append(item["corpus_id"])

    return items, correct_ids


def build_oracle_evidence_longmemeval(sample: Dict[str, Any], config: EvalConfig):
    gran = str(config.granularity or "session").strip().lower()
    if gran == "turn":
        items, correct_ids = _build_longmemeval_turn_items(sample)
    elif gran == "session":
        items, correct_ids = _build_longmemeval_session_items(sample)
    else:
        raise ValueError(f"Unknown LongMemEval granularity: {config.granularity}")

    if not items:
        raise ValueError("No evidence items can be built for LongMemEval sample.")

    correct_set = set(correct_ids)
    ranked = []
    rest = []
    for idx, item in enumerate(items):
        row = {
            "corpus_id": item["corpus_id"],
            "corpus_index": idx,
            "text": item["text"],
            "evidence_source": "oracle",
        }
        if item["corpus_id"] in correct_set:
            ranked.append(row)
        else:
            rest.append(row)

    ranked_items = (ranked + rest)[: max(1, int(config.top_k))]
    corpus_ids = [x["corpus_id"] for x in items]
    rankings = []
    id2idx = {cid: i for i, cid in enumerate(corpus_ids)}
    for row in ranked + rest:
        cid = row["corpus_id"]
        if cid in id2idx:
            rankings.append(id2idx[cid])

    return {
        "ranked_items": ranked_items,
        "corpus_ids": corpus_ids,
        "rankings": rankings,
        "correct_docs": list(correct_set),
    }


def build_oracle_evidence_personabench(sample: Dict[str, Any], config: EvalConfig):
    candidates = None
    for key in ("relevant_chunks", "supporting_chunks", "oracle_evidence"):
        val = sample.get(key)
        if isinstance(val, list):
            candidates = val
            break

    if candidates is None:
        raise ValueError("PersonaBench sample has no relevant_chunks/supporting_chunks/oracle_evidence field.")

    ranked_items = []
    for i, c in enumerate(candidates):
        txt = _safe_text(c)
        if not txt:
            continue
        ranked_items.append(
            {
                "corpus_id": f"oracle_{i}",
                "corpus_index": i,
                "text": txt,
                "evidence_source": "oracle",
            }
        )
        if len(ranked_items) >= max(1, int(config.top_k)):
            break

    if not ranked_items:
        raise ValueError("PersonaBench oracle evidence exists but all chunks are empty.")

    return {
        "ranked_items": ranked_items,
        "corpus_ids": [x["corpus_id"] for x in ranked_items],
        "rankings": list(range(len(ranked_items))),
        "correct_docs": [x["corpus_id"] for x in ranked_items],
    }


def build_oracle_evidence(sample: Dict[str, Any], config: EvalConfig):
    dname = normalize_dataset_name(config.dataset_name)
    if dname == "longmemeval":
        return build_oracle_evidence_longmemeval(sample, config)
    if dname == "personabench":
        return build_oracle_evidence_personabench(sample, config)
    raise NotImplementedError(
        f"Oracle evidence is not naturally available for dataset '{config.dataset_name}'."
    )


def build_evidence(sample: Dict[str, Any], config: EvalConfig, retrieve_fn):
    mode = str(config.evidence_mode or "retrieved").strip().lower()
    if mode == "retrieved":
        return retrieve_fn(sample, config)
    if mode == "oracle":
        return build_oracle_evidence(sample, config)
    raise ValueError(f"Unknown evidence_mode: {config.evidence_mode}")
