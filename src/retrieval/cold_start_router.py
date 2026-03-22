import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32)
    denom = np.linalg.norm(arr) + 1e-8
    return arr / denom


def load_prototype_bank(path):
    if path is None or str(path).strip() == "":
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prototype bank not found: {path}")
    return json.loads(p.read_text(encoding="utf-8-sig"))


class ColdStartRouter:
    """
    Router for Algorithm B:
    - m == 0      -> cohort_only
    - 0 < m < m0  -> blend
    - m >= m0     -> individual

    Cohort prototype is built from:
    1) supervised labels (department / role / team, etc.)
    2) unsupervised centroids
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enable = bool(cfg.get("enable_cold_start_router", False))
        self.m0 = int(cfg.get("cold_start_m0", 3))
        self.tau = float(cfg.get("cold_start_tau", 2.0))
        self.supervised_weight = float(cfg.get("cold_start_supervised_weight", 0.6))
        self.prefer_supervised = bool(cfg.get("cold_start_prefer_supervised", True))
        self.label_keys = list(cfg.get("cold_start_label_keys", ["department", "role", "team"]))
        self.prototype_bank = cfg.get("cold_start_prototype_bank_obj")
        self.debug = bool(cfg.get("cold_start_debug", False))

    @property
    def available(self):
        return self.enable and (self.prototype_bank is not None)

    def _extract_user_labels(self, test_item):
        labels = {}
        user_labels = test_item.get("user_labels")
        if isinstance(user_labels, dict):
            for k, v in user_labels.items():
                if isinstance(v, (str, int, float)):
                    labels[str(k)] = str(v)

        user_profile = test_item.get("user_profile")
        if isinstance(user_profile, dict):
            for k in self.label_keys:
                if k in user_profile and isinstance(user_profile[k], (str, int, float)):
                    labels[str(k)] = str(user_profile[k])
        return labels

    def _pick_supervised_prototype(self, labels):
        supervised = (self.prototype_bank or {}).get("supervised", {})
        if not isinstance(supervised, dict):
            return None, None

        for key in self.label_keys:
            if key not in labels:
                continue
            val = labels[key]
            key_bank = supervised.get(key, {})
            if isinstance(key_bank, dict) and val in key_bank:
                return np.asarray(key_bank[val], dtype=np.float32), {"label_key": key, "label_value": val}
        return None, None

    def _pick_unsupervised_prototype(self, anchor_embedding, user_id=None):
        unsup = (self.prototype_bank or {}).get("unsupervised", {})
        centroids = unsup.get("centroids", [])
        if not centroids:
            return None, None
        centroids = np.asarray(centroids, dtype=np.float32)

        user_to_cluster = (self.prototype_bank or {}).get("user_to_cluster", {})
        if user_id is not None and isinstance(user_to_cluster, dict) and str(user_id) in user_to_cluster:
            cid = int(user_to_cluster[str(user_id)])
            if 0 <= cid < len(centroids):
                return centroids[cid], {"cluster_id": cid, "cluster_source": "user_to_cluster"}

        sims = cosine_similarity(anchor_embedding[None, :], centroids).squeeze()
        cid = int(np.argmax(sims))
        return centroids[cid], {"cluster_id": cid, "cluster_source": "nearest_centroid"}

    def _compose_cohort_prototype(self, test_item, anchor_embedding):
        labels = self._extract_user_labels(test_item)
        user_id = test_item.get("user_id")

        sup_proto, sup_meta = self._pick_supervised_prototype(labels)
        unsup_proto, unsup_meta = self._pick_unsupervised_prototype(anchor_embedding, user_id=user_id)

        out_meta = {
            "labels": labels,
            "supervised_hit": sup_meta,
            "unsupervised_hit": unsup_meta,
        }

        if sup_proto is not None and unsup_proto is not None:
            if self.prefer_supervised:
                proto = l2_normalize(self.supervised_weight * sup_proto + (1 - self.supervised_weight) * unsup_proto)
            else:
                proto = l2_normalize((1 - self.supervised_weight) * sup_proto + self.supervised_weight * unsup_proto)
            out_meta["prototype_source"] = "hybrid"
            return proto, out_meta

        if sup_proto is not None:
            out_meta["prototype_source"] = "supervised"
            return l2_normalize(sup_proto), out_meta

        if unsup_proto is not None:
            out_meta["prototype_source"] = "unsupervised"
            return l2_normalize(unsup_proto), out_meta

        global_mean = (self.prototype_bank or {}).get("global_mean")
        if global_mean is not None:
            out_meta["prototype_source"] = "global_mean"
            return l2_normalize(np.asarray(global_mean, dtype=np.float32)), out_meta

        out_meta["prototype_source"] = "none"
        return None, out_meta

    def route(self, test_item, query_embedding, user_history_embedding, user_history_size):
        if not self.available:
            out = user_history_embedding if user_history_embedding is not None else query_embedding
            return l2_normalize(out), {"mode": "individual", "reason": "router_disabled_or_no_bank"}

        anchor = user_history_embedding if user_history_embedding is not None else query_embedding
        cohort_proto, meta = self._compose_cohort_prototype(test_item, anchor)

        if cohort_proto is None:
            out = user_history_embedding if user_history_embedding is not None else query_embedding
            return l2_normalize(out), {"mode": "individual", "reason": "no_prototype", "meta": meta}

        m = int(user_history_size)
        if m <= 0 or user_history_embedding is None:
            return cohort_proto, {"mode": "cohort_only", "blend_beta": 0.0, "meta": meta}

        if m < self.m0:
            beta = 1.0 - np.exp(-m / max(self.tau, 1e-8))
            blended = l2_normalize((1 - beta) * cohort_proto + beta * user_history_embedding)
            return blended, {"mode": "blend", "blend_beta": float(beta), "meta": meta}

        return l2_normalize(user_history_embedding), {"mode": "individual", "blend_beta": 1.0, "meta": meta}
