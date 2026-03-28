import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "you", "your", "i", "we", "they",
    "this", "those", "these", "my", "our", "their", "me", "us", "them",
}

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}

TOPIC_LEXICONS = {
    "work": {"project", "deadline", "meeting", "team", "manager", "report", "client", "office", "task"},
    "learning": {"study", "course", "learn", "research", "paper", "exam", "university", "class", "thesis"},
    "lifestyle": {"food", "travel", "movie", "music", "sport", "health", "family", "weekend", "home"},
    "finance": {"budget", "cost", "price", "salary", "investment", "bank", "expense", "payment", "revenue"},
    "technology": {"model", "code", "api", "system", "database", "server", "python", "algorithm", "deploy"},
}


def _tokenize(text):
    return re.findall(r"[A-Za-z][A-Za-z']+", str(text).lower())


def _safe_float(v):
    return float(v) if v is not None else 0.0


def _truncate_words(text, max_words=90):
    toks = str(text).split()
    if len(toks) <= max_words:
        return str(text).strip()
    return " ".join(toks[:max_words]).strip() + " ..."


def _extract_external_contrastive_examples(test_item, top_k=2, max_words=90):
    if top_k <= 0:
        return []
    for key in ("contrastive_examples", "negative_examples", "counter_examples"):
        val = test_item.get(key)
        if not isinstance(val, list):
            continue
        out = []
        for item in val:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or item.get("example")
            else:
                txt = item
            if not isinstance(txt, str) or not txt.strip():
                continue
            out.append({"source": "external", "text": _truncate_words(txt, max_words=max_words)})
            if len(out) >= top_k:
                break
        if out:
            return out
    return []


def _extract_entity_stats(raw_turns):
    text = "\n".join(raw_turns)
    cap_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    email_entities = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    date_like = re.findall(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", text)
    number_like = re.findall(r"\b\d+(?:\.\d+)?\b", text)

    unique_cap = len(set(cap_entities))
    return {
        "named_entity_count": int(len(cap_entities) + len(email_entities) + len(date_like)),
        "unique_named_entity_count": int(unique_cap + len(set(email_entities)) + len(set(date_like))),
        "email_count": int(len(email_entities)),
        "date_count": int(len(date_like)),
        "number_count": int(len(number_like)),
    }


def _infer_topic_cluster(filtered_tokens):
    if not filtered_tokens:
        return "general", {}
    token_counter = Counter(filtered_tokens)
    score = {}
    for topic, lex in TOPIC_LEXICONS.items():
        score[topic] = sum(token_counter.get(tok, 0) for tok in lex)
    best_topic = max(score, key=score.get) if score else "general"
    if score.get(best_topic, 0) <= 0:
        best_topic = "general"
    total = sum(score.values())
    if total <= 0:
        dist = {k: 0.0 for k in score.keys()}
    else:
        dist = {k: float(v) / float(total) for k, v in score.items()}
    return best_topic, dist


def _estimate_dependency_patterns(raw_turns):
    patterns = {
        "first_person_modal": 0,
        "first_person_verb": 0,
        "wh_question": 0,
        "negation": 0,
    }
    for turn in raw_turns:
        t = str(turn)
        low = t.lower()
        if re.search(r"\b(i|we)\s+(can|should|will|must|might|could|would)\b", low):
            patterns["first_person_modal"] += 1
        if re.search(r"\b(i|we)\s+(think|feel|need|want|prefer|plan|believe|know)\b", low):
            patterns["first_person_verb"] += 1
        if re.search(r"\b(what|why|how|when|where|who|which)\b", low) and "?" in t:
            patterns["wh_question"] += 1
        if re.search(r"\b(not|never|no|can't|cannot|won't|don't|didn't|isn't|aren't)\b", low):
            patterns["negation"] += 1
    return patterns


def build_explicit_feature_profile(history_turns_per_session, top_k_keywords=12, top_k_bigrams=6):
    sessions = [x for x in history_turns_per_session if isinstance(x, list) and len(x) > 0]
    all_turns = [t for sess in sessions for t in sess if isinstance(t, str) and t.strip()]
    if not all_turns:
        return {
            "session_count": 0,
            "turn_count": 0,
            "top_keywords": [],
            "top_bigrams": [],
            "semantic": {},
            "style": {},
            "syntax": {},
        }

    all_tokens = []
    filtered_tokens = []
    first_person_cnt = 0
    question_cnt = 0
    exclaim_cnt = 0
    turn_lengths = []
    session_lengths = []
    punct_counter = Counter()

    for sess in sessions:
        sess_tokens = []
        for turn in sess:
            toks = _tokenize(turn)
            if not toks:
                continue
            turn_lengths.append(len(toks))
            all_tokens.extend(toks)
            sess_tokens.extend(toks)
            first_person_cnt += sum(1 for x in toks if x in FIRST_PERSON)
            filtered_tokens.extend([x for x in toks if len(x) >= 3 and x not in STOPWORDS])
            if "?" in turn:
                question_cnt += 1
            if "!" in turn:
                exclaim_cnt += 1
            punct_counter["comma"] += turn.count(",")
            punct_counter["period"] += turn.count(".")
            punct_counter["question"] += turn.count("?")
            punct_counter["exclaim"] += turn.count("!")
        if sess_tokens:
            session_lengths.append(len(sess_tokens))

    token_counter = Counter(filtered_tokens)
    top_keywords = [w for w, _ in token_counter.most_common(top_k_keywords)]

    bigrams = Counter()
    for i in range(len(filtered_tokens) - 1):
        a, b = filtered_tokens[i], filtered_tokens[i + 1]
        if a == b:
            continue
        bigrams[(a, b)] += 1
    top_bigrams = [" ".join(bg) for bg, _ in bigrams.most_common(top_k_bigrams)]

    turn_count = len(all_turns)
    token_count = len(all_tokens)
    style = {
        "avg_turn_words": _safe_float(np.mean(turn_lengths)) if turn_lengths else 0.0,
        "avg_session_words": _safe_float(np.mean(session_lengths)) if session_lengths else 0.0,
        "question_turn_ratio": _safe_float(question_cnt / max(turn_count, 1)),
        "exclaim_turn_ratio": _safe_float(exclaim_cnt / max(turn_count, 1)),
        "first_person_ratio": _safe_float(first_person_cnt / max(token_count, 1)),
        "lexical_diversity": _safe_float(len(set(all_tokens)) / max(token_count, 1)),
        "comma_per_turn": _safe_float(punct_counter["comma"] / max(turn_count, 1)),
        "period_per_turn": _safe_float(punct_counter["period"] / max(turn_count, 1)),
    }

    topic_label, topic_dist = _infer_topic_cluster(filtered_tokens)
    semantic = {
        "topic_cluster_label": topic_label,
        "topic_distribution": topic_dist,
        "topic_keywords": top_keywords[: min(5, len(top_keywords))],
        "entity_stats": _extract_entity_stats(all_turns),
    }
    syntax = {
        "dependency_patterns": _estimate_dependency_patterns(all_turns),
    }

    return {
        "session_count": len(sessions),
        "turn_count": turn_count,
        "top_keywords": top_keywords,
        "top_bigrams": top_bigrams,
        "semantic": semantic,
        "style": style,
        "syntax": syntax,
    }


def render_explicit_feature_block(feature_profile):
    if not feature_profile:
        return ""

    style = feature_profile.get("style", {})
    semantic = feature_profile.get("semantic", {})
    syntax = feature_profile.get("syntax", {})
    keywords = feature_profile.get("top_keywords", [])
    bigrams = feature_profile.get("top_bigrams", [])
    if not keywords and not bigrams:
        return ""

    lines = [
        "Explicit User Feature Summary",
        f"- sessions: {feature_profile.get('session_count', 0)}, turns: {feature_profile.get('turn_count', 0)}",
        (
            "- style: "
            f"avg_turn_words={style.get('avg_turn_words', 0.0):.2f}, "
            f"question_ratio={style.get('question_turn_ratio', 0.0):.2f}, "
            f"exclaim_ratio={style.get('exclaim_turn_ratio', 0.0):.2f}, "
            f"first_person_ratio={style.get('first_person_ratio', 0.0):.2f}, "
            f"lexical_diversity={style.get('lexical_diversity', 0.0):.2f}"
        ),
    ]

    topic_label = semantic.get("topic_cluster_label", "general")
    entity_stats = semantic.get("entity_stats", {})
    dep = syntax.get("dependency_patterns", {})
    lines.append(f"- semantic: topic={topic_label}, entity_count={entity_stats.get('named_entity_count', 0)}")
    lines.append(
        "- syntax: "
        f"first_person_modal={dep.get('first_person_modal', 0)}, "
        f"wh_question={dep.get('wh_question', 0)}, "
        f"negation={dep.get('negation', 0)}"
    )

    if keywords:
        lines.append("- keywords: " + ", ".join(keywords))
    if bigrams:
        lines.append("- key phrases: " + ", ".join(bigrams))
    return "\n".join(lines)


def select_contrastive_examples(
    test_item,
    history_plain_texts,
    history_ids,
    history_embeddings,
    anchor_embedding,
    top_k=2,
    max_words=90,
    external_only=False,
):
    if top_k <= 0:
        return []
    external = _extract_external_contrastive_examples(test_item, top_k=top_k, max_words=max_words)
    if external_only:
        return external[:top_k]
    if len(external) >= top_k:
        return external[:top_k]

    remain = max(top_k - len(external), 0)
    if remain == 0:
        return external
    if history_embeddings is None or anchor_embedding is None:
        return external
    if len(history_plain_texts) <= 1:
        return external

    sims = cosine_similarity(history_embeddings, anchor_embedding[None, :]).squeeze()
    order = np.argsort(sims)[:remain]
    out = list(external)
    for idx in order.tolist():
        txt = history_plain_texts[idx] if idx < len(history_plain_texts) else ""
        if not isinstance(txt, str) or not txt.strip():
            continue
        out.append(
            {
                "source": "history_dissimilar",
                "session_id": str(history_ids[idx]) if idx < len(history_ids) else str(idx),
                "similarity": float(sims[idx]),
                "text": _truncate_words(txt, max_words=max_words),
            }
        )
    return out[:top_k]


def render_contrastive_block(contrastive_examples):
    if not contrastive_examples:
        return ""
    lines = [
        "Contrastive References (less target-aligned)",
        "Use them as negative signals: avoid imitating these styles or preferences.",
    ]
    for i, ex in enumerate(contrastive_examples, 1):
        txt = ex.get("text", "")
        src = ex.get("source", "unknown")
        lines.append(f"[{i}] source={src}: {txt}")
    return "\n".join(lines)


def chunk_to_plain_text(chunk):
    if not isinstance(chunk, str):
        return str(chunk)
    try:
        obj = json.loads(chunk)
    except Exception:
        return chunk
    if isinstance(obj, dict) and isinstance(obj.get("conversation"), list):
        turns = [str(x) for x in obj["conversation"] if isinstance(x, str)]
        if turns:
            return " ".join(turns)
    return chunk
