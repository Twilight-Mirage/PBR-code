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


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ColdStartRouter:
    """
    Router for Algorithm B:
    - m == 0      -> cohort_only
    - 0 < m < t   -> blend
    - m >= t      -> individual

    Cohort prototype is built from:
    1) supervised labels (department / role / team, etc.)
    2) unsupervised centroids

    Blending schedule:
    - linear: lambda = min(1, m / K)
    - exp:    lambda = 1 - exp(-m / tau)
    - auto:   if K is explicitly provided -> linear; otherwise if tau is provided -> exp;
              else fallback to linear with m0.
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enable = bool(cfg.get("enable_cold_start_router", False))

        self.cold_start_m0 = int(cfg.get("cold_start_m0", 3))
        self.cold_start_k = _optional_float(cfg.get("cold_start_k", None))
        self.cold_start_tau = _optional_float(cfg.get("cold_start_tau", None))

        if self.cold_start_k is not None and self.cold_start_k <= 0:
            self.cold_start_k = None
        if self.cold_start_tau is not None and self.cold_start_tau <= 0:
            self.cold_start_tau = None

        gate = str(cfg.get("cold_start_gate", "auto")).strip().lower()
        self.cold_start_gate = gate if gate in {"auto", "linear", "exp"} else "auto"

        self.supervised_weight = float(cfg.get("cold_start_supervised_weight", 0.6))
        self.prefer_supervised = bool(cfg.get("cold_start_prefer_supervised", True))
        self.label_keys = list(cfg.get("cold_start_label_keys", ["department", "role", "team"]))
        self.min_cluster_size = int(cfg.get("cold_start_min_cluster_size", 2))
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
        cluster_sizes = unsup.get("cluster_sizes", [])
        if not isinstance(cluster_sizes, list) or len(cluster_sizes) != len(centroids):
            cluster_sizes = [0] * len(centroids)

        valid_mask = np.asarray([int(x) >= self.min_cluster_size for x in cluster_sizes], dtype=bool)
        if not np.any(valid_mask):
            return None, {"cluster_source": "none_valid_cluster", "min_cluster_size": self.min_cluster_size}

        user_to_cluster = (self.prototype_bank or {}).get("user_to_cluster", {})
        if user_id is not None and isinstance(user_to_cluster, dict) and str(user_id) in user_to_cluster:
            cid = int(user_to_cluster[str(user_id)])
            if 0 <= cid < len(centroids) and valid_mask[cid]:
                return centroids[cid], {"cluster_id": cid, "cluster_source": "user_to_cluster"}
            if 0 <= cid < len(centroids) and (not valid_mask[cid]):
                return None, {
                    "cluster_id": cid,
                    "cluster_source": "user_to_cluster_too_small",
                    "cluster_size": int(cluster_sizes[cid]),
                    "min_cluster_size": self.min_cluster_size,
                }

        valid_centroids = centroids[valid_mask]
        valid_ids = np.where(valid_mask)[0]
        sims = cosine_similarity(anchor_embedding[None, :], valid_centroids).squeeze()
        if np.ndim(sims) == 0:
            chosen_pos = 0
        else:
            chosen_pos = int(np.argmax(sims))
        cid = int(valid_ids[chosen_pos])
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

    def _resolve_history_lambda(self, m):
        gate = self.cold_start_gate
        if gate == "auto":
            if self.cold_start_k is not None:
                gate = "linear"
            elif self.cold_start_tau is not None:
                gate = "exp"
            else:
                gate = "linear"

        if gate == "exp":
            tau = self.cold_start_tau
            if tau is None:
                tau = float(max(self.cold_start_m0, 1))
            lam = 1.0 - float(np.exp(-float(m) / max(tau, 1e-8)))
            return float(np.clip(lam, 0.0, 1.0)), "exp", {"tau": float(tau)}

        k = self.cold_start_k
        if k is None:
            k = float(max(self.cold_start_m0, 1))
        lam = float(min(1.0, float(m) / max(k, 1e-8)))
        return float(np.clip(lam, 0.0, 1.0)), "linear", {"k": float(k)}

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
            return cohort_proto, {
                "mode": "cohort_only",
                "history_lambda": 0.0,
                "history_gate": None,
                "meta": meta,
            }

        lam, gate, gate_meta = self._resolve_history_lambda(m)
        if lam < 1.0:
            blended = l2_normalize(lam * user_history_embedding + (1 - lam) * cohort_proto)
            out = {"mode": "blend", "history_lambda": lam, "history_gate": gate, "meta": meta}
            out.update(gate_meta)
            return blended, out

        out = {
            "mode": "individual",
            "history_lambda": 1.0,
            "history_gate": gate,
            "meta": meta,
        }
        out.update(gate_meta)
        return l2_normalize(user_history_embedding), out
