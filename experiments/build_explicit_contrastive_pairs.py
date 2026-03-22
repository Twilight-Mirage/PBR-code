import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def l2_normalize(v):
    arr = np.asarray(v, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


def resolve_user_id(item, idx):
    for key in ("user_id", "profile_id", "author_id"):
        if key in item and item[key] is not None:
            return str(item[key])
    qid = item.get("question_id")
    if qid is not None:
        return str(qid)
    return f"sample_{idx}"


def collect_user_session_texts(item):
    texts = []
    for sess in item.get("haystack_sessions", []):
        if not isinstance(sess, list):
            continue
        user_turns = [x.get("content", "") for x in sess if isinstance(x, dict) and x.get("role") == "user"]
        user_turns = [x for x in user_turns if isinstance(x, str) and x.strip()]
        if user_turns:
            texts.append(" ".join(user_turns))
    return texts


def collect_user_labels(item):
    out = {}
    if isinstance(item.get("user_labels"), dict):
        for k, v in item["user_labels"].items():
            if isinstance(v, (str, int, float)):
                out[str(k)] = str(v)
    if isinstance(item.get("user_profile"), dict):
        for k, v in item["user_profile"].items():
            if isinstance(v, (str, int, float)):
                out[str(k)] = str(v)
    return out


def is_label_compatible(a_labels, b_labels, label_keys):
    if not label_keys:
        return True
    for key in label_keys:
        av = a_labels.get(key)
        bv = b_labels.get(key)
        if av is not None and bv is not None and av == bv:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Build triplets for explicit-user-representation contrastive training.")
    parser.add_argument("--input_json", type=str, required=True, help="Input dataset json path.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output jsonl path for triplets.")
    parser.add_argument(
        "--retrieval_model_name",
        type=str,
        default="multi-qa-MiniLM-L6-cos-v1",
        help="Embedding model used for hard-negative mining.",
    )
    parser.add_argument("--max_pairs_per_user", type=int, default=12, help="Max (anchor,pos,neg) triplets per user.")
    parser.add_argument(
        "--negative_diff_label_keys",
        type=str,
        default="",
        help="Comma-separated label keys requiring user-level mismatch for negatives (e.g., department,role).",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap on loaded examples.")
    args = parser.parse_args()

    in_path = Path(args.input_json)
    out_path = Path(args.output_jsonl)
    label_keys = [x.strip() for x in args.negative_diff_label_keys.split(",") if x.strip()]

    data = json.loads(in_path.read_text(encoding="utf-8-sig"))
    if args.max_items > 0:
        data = data[: args.max_items]

    user_sessions = defaultdict(list)
    user_labels = defaultdict(dict)
    for idx, item in enumerate(data):
        uid = resolve_user_id(item, idx)
        texts = collect_user_session_texts(item)
        user_sessions[uid].extend(texts)
        labels = collect_user_labels(item)
        if labels:
            user_labels[uid].update(labels)

    user_ids = []
    flat_texts = []
    owners = []
    for uid, texts in user_sessions.items():
        texts = [x for x in texts if isinstance(x, str) and x.strip()]
        if len(texts) < 2:
            continue
        user_ids.append(uid)
        for txt in texts:
            flat_texts.append(txt)
            owners.append(uid)

    if len(user_ids) < 2:
        raise ValueError("Need at least 2 users (each with >=2 sessions) to build contrastive pairs.")

    model = SentenceTransformer(args.retrieval_model_name, trust_remote_code=True)
    emb = model.encode(flat_texts, convert_to_numpy=True)
    emb = np.asarray([l2_normalize(x) for x in emb], dtype=np.float32)

    by_user_emb = defaultdict(list)
    by_user_text = defaultdict(list)
    for idx, uid in enumerate(owners):
        by_user_emb[uid].append(emb[idx])
        by_user_text[uid].append(flat_texts[idx])

    user_centroids = {}
    for uid in user_ids:
        user_centroids[uid] = l2_normalize(np.mean(np.stack(by_user_emb[uid]), axis=0))

    triplets = []
    for uid in tqdm(user_ids, desc="Building triplets"):
        pos_texts = by_user_text[uid]
        pos_embs = np.stack(by_user_emb[uid])

        cand_uids = [x for x in user_ids if x != uid and is_label_compatible(user_labels[uid], user_labels[x], label_keys)]
        if not cand_uids:
            cand_uids = [x for x in user_ids if x != uid]
        if not cand_uids:
            continue

        user_sim = cosine_similarity(user_centroids[uid][None, :], np.stack([user_centroids[x] for x in cand_uids])).squeeze()
        hard_uid = cand_uids[int(np.argmax(user_sim))]

        neg_texts = by_user_text[hard_uid]
        neg_embs = np.stack(by_user_emb[hard_uid])
        max_pairs = min(args.max_pairs_per_user, len(pos_texts))
        for i in range(max_pairs):
            pos_t = pos_texts[i]
            pos_e = pos_embs[i]
            neg_sim = cosine_similarity(pos_e[None, :], neg_embs).squeeze()
            neg_idx = int(np.argmax(neg_sim))
            neg_t = neg_texts[neg_idx]
            triplets.append(
                {
                    "anchor_user_id": uid,
                    "hard_negative_user_id": hard_uid,
                    "anchor_profile_text": " ".join(pos_texts[: min(len(pos_texts), 3)]),
                    "positive_text": pos_t,
                    "negative_text": neg_t,
                    "meta": {
                        "anchor_to_negative_user_sim": float(np.max(user_sim)),
                        "positive_to_negative_sim": float(neg_sim[neg_idx]),
                    },
                }
            )

    if not triplets:
        raise ValueError("No triplets generated. Try relaxing label mismatch constraints.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in triplets:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(triplets)} triplets to: {out_path}")


if __name__ == "__main__":
    main()
