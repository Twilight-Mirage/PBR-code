import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm


def l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


def parse_label_keys(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def extract_labels(item, label_keys):
    labels = {}

    user_labels = item.get("user_labels")
    if isinstance(user_labels, dict):
        for k, v in user_labels.items():
            if isinstance(v, (str, int, float)):
                labels[str(k)] = str(v)

    user_profile = item.get("user_profile")
    if isinstance(user_profile, dict):
        for k in label_keys:
            if k in user_profile and isinstance(user_profile[k], (str, int, float)):
                labels[k] = str(user_profile[k])

    return labels


def resolve_user_id(item, idx):
    for key in ("user_id", "profile_id", "author_id"):
        if key in item and item[key] is not None:
            return str(item[key])
    qid = item.get("question_id")
    if qid is not None:
        return str(qid)
    return f"sample_{idx}"


def build_user_texts(item):
    texts = []
    sessions = item.get("haystack_sessions")
    dates = item.get("haystack_dates")

    if isinstance(sessions, list):
        for sid, sess in enumerate(sessions):
            if not isinstance(sess, list):
                continue
            user_turns = [x.get("content", "") for x in sess if isinstance(x, dict) and x.get("role") == "user"]
            user_turns = [x for x in user_turns if isinstance(x, str) and x.strip()]
            if not user_turns:
                continue
            payload = {
                "date": dates[sid] if isinstance(dates, list) and sid < len(dates) else None,
                "conversation": user_turns,
            }
            texts.append(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

    # Fallback for datasets without haystack sessions.
    if not texts:
        q = item.get("question")
        if isinstance(q, str) and q.strip():
            texts.append(q)

    return texts


def aggregate_user_vectors(data, model, label_keys):
    user_vectors = defaultdict(list)
    user_labels = defaultdict(dict)

    for idx, item in enumerate(tqdm(data, desc="Building user embeddings")):
        uid = resolve_user_id(item, idx)
        texts = build_user_texts(item)
        if not texts:
            continue

        emb = model.encode(texts, convert_to_numpy=True)
        user_vec = l2_normalize(np.mean(emb, axis=0))
        user_vectors[uid].append(user_vec)

        labels = extract_labels(item, label_keys)
        if labels:
            user_labels[uid].update(labels)

    user_ids = []
    vectors = []
    for uid, vecs in user_vectors.items():
        user_ids.append(uid)
        vectors.append(l2_normalize(np.mean(np.stack(vecs), axis=0)))

    if not vectors:
        raise ValueError("No user vectors built from input data. Check data format and fields.")

    X = np.stack(vectors).astype(np.float32)
    return user_ids, X, user_labels


def build_unsupervised_bank(user_ids, X, num_clusters, random_seed):
    n_users = len(user_ids)
    k = max(1, min(num_clusters, n_users))

    if k == 1:
        labels = np.zeros(n_users, dtype=np.int64)
        centroids = np.array([l2_normalize(np.mean(X, axis=0))], dtype=np.float32)
    else:
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = np.asarray([l2_normalize(c) for c in kmeans.cluster_centers_], dtype=np.float32)

    user_to_cluster = {uid: int(cid) for uid, cid in zip(user_ids, labels)}
    cluster_sizes = Counter(labels.tolist())
    cluster_size_list = [int(cluster_sizes.get(i, 0)) for i in range(len(centroids))]

    return {
        "centroids": centroids.tolist(),
        "cluster_sizes": cluster_size_list,
    }, user_to_cluster


def build_supervised_bank(user_ids, X, user_labels, label_keys, min_group_size):
    uid_to_vec = {uid: X[idx] for idx, uid in enumerate(user_ids)}
    grouped = {k: defaultdict(list) for k in label_keys}

    for uid, labels in user_labels.items():
        if uid not in uid_to_vec:
            continue
        vec = uid_to_vec[uid]
        for key in label_keys:
            value = labels.get(key)
            if value:
                grouped[key][str(value)].append(vec)

    out = {}
    for key in label_keys:
        out[key] = {}
        for label_value, vecs in grouped[key].items():
            if len(vecs) < min_group_size:
                continue
            out[key][label_value] = l2_normalize(np.mean(np.stack(vecs), axis=0)).tolist()

    return out


def main():
    parser = argparse.ArgumentParser(description="Build prototype bank for cold-start routing.")
    parser.add_argument("--input_json", type=str, required=True, help="Dataset json path.")
    parser.add_argument("--output_json", type=str, required=True, help="Output prototype-bank json path.")
    parser.add_argument(
        "--retrieval_model_name",
        type=str,
        default="multi-qa-MiniLM-L6-cos-v1",
        help="SentenceTransformer model used to encode user history.",
    )
    parser.add_argument("--label_keys", type=str, default="department,role,team", help="Comma-separated supervised label keys.")
    parser.add_argument("--num_clusters", type=int, default=8, help="Number of unsupervised user clusters.")
    parser.add_argument(
        "--min_supervised_group_size",
        type=int,
        default=2,
        help="Minimum samples per label group to keep a supervised prototype.",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for KMeans.")
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap on loaded items (0 means all).")
    args = parser.parse_args()

    label_keys = parse_label_keys(args.label_keys)
    in_path = Path(args.input_json)
    out_path = Path(args.output_json)

    data = json.loads(in_path.read_text(encoding="utf-8-sig"))
    if args.max_items > 0:
        data = data[: args.max_items]

    model = SentenceTransformer(args.retrieval_model_name, trust_remote_code=True)
    user_ids, X, user_labels = aggregate_user_vectors(data, model, label_keys)

    unsupervised_bank, user_to_cluster = build_unsupervised_bank(
        user_ids=user_ids,
        X=X,
        num_clusters=args.num_clusters,
        random_seed=args.random_seed,
    )
    supervised_bank = build_supervised_bank(
        user_ids=user_ids,
        X=X,
        user_labels=user_labels,
        label_keys=label_keys,
        min_group_size=args.min_supervised_group_size,
    )

    output = {
        "meta": {
            "created_by": "experiments/build_coldstart_prototype_bank.py",
            "retrieval_model_name": args.retrieval_model_name,
            "num_users": len(user_ids),
            "embedding_dim": int(X.shape[1]),
            "num_clusters": int(len(unsupervised_bank["centroids"])),
            "label_keys": label_keys,
            "min_supervised_group_size": int(args.min_supervised_group_size),
        },
        "global_mean": l2_normalize(np.mean(X, axis=0)).tolist(),
        "supervised": supervised_bank,
        "unsupervised": unsupervised_bank,
        "user_to_cluster": user_to_cluster,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved prototype bank to: {out_path}")


if __name__ == "__main__":
    main()
