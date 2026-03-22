import argparse
import asyncio
import json
import time
from datetime import datetime

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from async_llm import run_async
from src.retrieval.cold_start_router import ColdStartRouter, l2_normalize, load_prototype_bank
from src.retrieval.eval_utils import evaluate_retrieval


def load_json_res(res):
    import re

    match = re.search(r"(?:json)?\s*(\{.*?\})\s*", res, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            data = {}
            print("error in transform json", e)
    else:
        print("Output is not in JSON format")
        data = {}
    return data


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        if isinstance(data, (dict, list)):
            json.dump(data, f, indent=4, ensure_ascii=False)
        elif isinstance(data, str):
            try:
                json.loads(data)
                f.write(data)
            except json.JSONDecodeError:
                raise ValueError("Provided string is not valid JSON")
        else:
            raise TypeError("Data must be a list/dictionary or a JSON string")


def gen_retrieval_prompt_fake_ada_reason_10(query, doc, model):
    prompt_template = """You are to generate 10 natural candidate utterances the user might say, inspired by the dialogue history and the current question.

Context
------------
User dialogue history (for style imitation):  
{history}

Current question (to inspire the utterances):  
{query}
------------

Guidelines
1. Generate 10 fluent, natural utterances the user might plausibly say.
2. Do NOT just paraphrase; include variations in tone, emphasis, or context.
3. Each > 25 words.
4. Reflect the style and tone consistent with the document.
5. Return ONLY valid JSON in this format (no comments, no markdown):
   {{
     "candidates": [
       "...",
       "...",
       "...",
     ]
   }}
"""
    instruction = prompt_template.format(query=query, history=doc)
    p_utt_prompt = instruction
    prompt_template = """Solve the question step-by-step, inspired by the user dialogue history.

Context
------------
User dialogue history (for style imitation):  
{history}

Current question (to inspire the utterances):  
{query}
------------
Output (step-by-step):
"""
    instruction = prompt_template.format(query=query, history=doc)
    p_rea_prompt = instruction

    return p_utt_prompt, p_rea_prompt


def remove_key(json_data, key_to_remove):
    if isinstance(json_data, dict):
        return {k: remove_key(v, key_to_remove) for k, v in json_data.items() if k != key_to_remove}
    if isinstance(json_data, list):
        return [remove_key(item, key_to_remove) for item in json_data]
    return json_data


def convert_json_to_plain_text(json_data, exclude=None):
    exclude = exclude or []
    for key_to_remove in exclude:
        json_data = remove_key(json_data, key_to_remove)
    return json.dumps(json_data, separators=(",", ":"))


def safe_parse_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


class RAGRetriever:
    def __init__(self, retriever_model, retriever_model_name, data_type="s", temporal_cfg=None, coldstart_cfg=None):
        self.retriever_model = retriever_model
        self.index = None
        self.chunks = []
        self.segment_ids = []
        self.user_embd_mean = None

        self.memory_embeddings = None
        self.adjacency_matrix = None
        self.memory_graph = None
        self.damping_factor = 0.85
        if data_type == "m":
            self.sim_threshold = 0.75
            self.top_k_neighbors = 50
        else:
            self.sim_threshold = 0.75
            self.top_k_neighbors = 10
        self.pi_step = 50

        temporal_cfg = temporal_cfg or {}
        self.enable_temporal_profile = bool(temporal_cfg.get("enable_temporal_profile", False))
        self.temporal_decay_lambda = float(temporal_cfg.get("temporal_decay_lambda", 0.05))
        self.temporal_util_alpha = float(temporal_cfg.get("temporal_util_alpha", 4.0))
        self.temporal_util_bias = float(temporal_cfg.get("temporal_util_bias", 0.0))
        self.temporal_rerank_beta = float(temporal_cfg.get("temporal_rerank_beta", 0.4))
        self.temporal_graph_decay = float(temporal_cfg.get("temporal_graph_decay", 0.02))
        self.short_term_blend = float(temporal_cfg.get("short_term_blend", 0.5))
        self.seed_candidate_multiplier = int(temporal_cfg.get("seed_candidate_multiplier", 4))

        self.session_datetimes = []
        self.session_ordinals = None
        self.session_utilities = None
        self.recency_weights = None
        self.util_weights = None
        self.temporal_weights = None
        self.user_profile_embedding = None
        self.short_term_center = None
        self.long_term_center = None

        coldstart_cfg = coldstart_cfg or {}
        self.cold_router = ColdStartRouter(coldstart_cfg)
        self.enable_cold_start_router = bool(coldstart_cfg.get("enable_cold_start_router", False))
        self.cold_start_anchor_mix = float(coldstart_cfg.get("cold_start_anchor_mix", 0.35))
        self.cold_start_seed_profile_alpha = float(coldstart_cfg.get("cold_start_seed_profile_alpha", 0.2))
        self.cold_start_rerank_alpha = float(coldstart_cfg.get("cold_start_rerank_alpha", 0.25))
        self.cold_start_route_info = {
            "mode": "individual",
            "reason": "router_not_used",
        }

    def _estimate_session_utility(self, sess_entry):
        user_turns = [x.get("content", "") for x in sess_entry if x.get("role") == "user"]
        if not user_turns:
            return 0.1
        avg_len = np.mean([len(t.split()) for t in user_turns])
        depth_score = min(len(user_turns) / 6.0, 1.0)
        richness_score = min(avg_len / 40.0, 1.0)
        return float(0.5 * depth_score + 0.5 * richness_score)

    def _compute_temporal_weights(self, question_date):
        n = len(self.segment_ids)
        if n == 0:
            self.recency_weights = np.array([], dtype=np.float32)
            self.util_weights = np.array([], dtype=np.float32)
            self.temporal_weights = np.array([], dtype=np.float32)
            return

        ref_dt = safe_parse_datetime(question_date)
        if ref_dt is None:
            valid_dts = [x for x in self.session_datetimes if x is not None]
            ref_dt = max(valid_dts) if valid_dts else datetime.utcnow()

        ages = []
        ordinals = []
        for dt in self.session_datetimes:
            if dt is None:
                ages.append(365.0)
                ordinals.append(np.nan)
            else:
                ages.append(float(max((ref_dt.date() - dt.date()).days, 0)))
                ordinals.append(float(dt.toordinal()))

        ordinals = np.array(ordinals, dtype=np.float32)
        if np.any(np.isnan(ordinals)):
            max_ord = np.nanmax(ordinals) if np.any(~np.isnan(ordinals)) else float(ref_dt.toordinal())
            ordinals = np.where(np.isnan(ordinals), max_ord - 365.0, ordinals)
        self.session_ordinals = ordinals

        ages = np.array(ages, dtype=np.float32)
        self.recency_weights = np.exp(-self.temporal_decay_lambda * ages)

        self.util_weights = 1.0 / (1.0 + np.exp(-self.temporal_util_alpha * (self.session_utilities - self.temporal_util_bias)))

        temporal = self.recency_weights * self.util_weights
        temporal = np.clip(temporal, 1e-6, None)
        self.temporal_weights = temporal / (temporal.sum() + 1e-8)

    def query_seed_weighted(self, questions, top_k=10):
        """
        P-PRF++ seed retrieval following Algorithm A:
        score_i = sim(q, doc_i) * exp(-lambda * age_i) * sigmoid(alpha * util_i)
        """
        if self.index is None:
            raise ValueError("Index has not been built. Please call build_index() first.")
        if len(questions) != 1:
            raise ValueError("query_seed_weighted currently supports single-query input.")

        top_k = min(top_k, len(self.chunks))
        q_embedding = self.retriever_model.encode(questions, convert_to_numpy=True)[0]
        sim_scores = np.dot(self.memory_embeddings, q_embedding)

        if self.enable_temporal_profile and self.temporal_weights is not None:
            weighted_scores = sim_scores * self.temporal_weights
        else:
            weighted_scores = sim_scores

        if self.enable_cold_start_router and self.user_profile_embedding is not None:
            profile_sim = cosine_similarity(self.memory_embeddings, self.user_profile_embedding[None, :]).squeeze()
            if self.cold_start_route_info.get("mode") in {"cohort_only", "blend"}:
                weighted_scores = weighted_scores * (1 + self.cold_start_seed_profile_alpha * profile_sim)

        idx = np.argsort(weighted_scores)[::-1][:top_k]
        d_scores = weighted_scores[idx][None, :].astype(np.float32)
        i_rankings = idx[None, :].astype(np.int64)
        rankings_id = self.segment_ids[i_rankings].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[i_rankings].tolist()[0]
        return d_scores, i_rankings, retrieved_chunks, rankings_id

    def _temporal_mix_scores(self, idx, base_scores):
        if (not self.enable_temporal_profile) or self.temporal_weights is None:
            return base_scores
        temporal_scores = self.temporal_weights[idx]
        return base_scores * ((1 - self.temporal_rerank_beta) + self.temporal_rerank_beta * temporal_scores)

    def _temporal_rerank(self, D, I, top_k):
        if (not self.enable_temporal_profile) or self.temporal_weights is None:
            return D[:, :top_k], I[:, :top_k]

        reranked_D = []
        reranked_I = []
        for row in range(I.shape[0]):
            valid_mask = I[row] >= 0
            valid_idx = I[row][valid_mask]
            valid_scores = D[row][valid_mask]
            if len(valid_idx) == 0:
                reranked_D.append(np.array([], dtype=np.float32))
                reranked_I.append(np.array([], dtype=np.int64))
                continue
            mixed_scores = self._temporal_mix_scores(valid_idx, valid_scores)
            order = np.argsort(mixed_scores)[::-1][:top_k]
            reranked_D.append(mixed_scores[order].astype(np.float32))
            reranked_I.append(valid_idx[order].astype(np.int64))

        return np.stack(reranked_D), np.stack(reranked_I)

    def _profile_rerank(self, D, I, top_k):
        if (not self.enable_cold_start_router) or (self.user_profile_embedding is None):
            return D[:, :top_k], I[:, :top_k]
        if self.cold_start_route_info.get("mode") not in {"cohort_only", "blend"}:
            return D[:, :top_k], I[:, :top_k]

        reranked_D = []
        reranked_I = []
        for row in range(I.shape[0]):
            valid_mask = I[row] >= 0
            valid_idx = I[row][valid_mask]
            valid_scores = D[row][valid_mask]
            if len(valid_idx) == 0:
                reranked_D.append(np.array([], dtype=np.float32))
                reranked_I.append(np.array([], dtype=np.int64))
                continue
            profile_scores = cosine_similarity(self.memory_embeddings[valid_idx], self.user_profile_embedding[None, :]).squeeze()
            mixed_scores = (1 - self.cold_start_rerank_alpha) * valid_scores + self.cold_start_rerank_alpha * profile_scores
            order = np.argsort(mixed_scores)[::-1][:top_k]
            reranked_D.append(mixed_scores[order].astype(np.float32))
            reranked_I.append(valid_idx[order].astype(np.int64))
        return np.stack(reranked_D), np.stack(reranked_I)

    def build_index(self, test_item):
        corpus, corpus_ids = [], []
        self.session_datetimes = []
        utility_map = test_item.get("session_utilities", {})
        utilities = []

        for cur_sess_id, sess_entry, ts in zip(
            test_item["haystack_session_ids"],
            test_item["haystack_sessions"],
            test_item["haystack_dates"],
        ):
            user_data = []
            for item in sess_entry:
                if item["role"] == "user":
                    user_data.append(item["content"])
            tmp_data = {"date": ts, "conversation": user_data}
            segment_id = cur_sess_id
            plain_text = json.dumps(tmp_data, separators=(",", ":"))
            corpus.append(plain_text)
            corpus_ids.append(segment_id)
            self.session_datetimes.append(safe_parse_datetime(ts))

            util = utility_map.get(cur_sess_id) if isinstance(utility_map, dict) else None
            if util is None:
                util = self._estimate_session_utility(sess_entry)
            utilities.append(float(util))

        if len(corpus) == 0:
            raise ValueError("No user history found in haystack_sessions; cannot build retrieval index.")

        embeddings = self.retriever_model.encode(corpus, convert_to_numpy=True)
        self.user_embd_mean = np.mean(embeddings, axis=0)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.chunks = corpus
        self.segment_ids = np.array(corpus_ids)
        self.session_utilities = np.array(utilities, dtype=np.float32)
        self._compute_temporal_weights(test_item.get("question_date"))

        if self.enable_temporal_profile:
            weighted_profile = self.temporal_weights / (self.temporal_weights.sum() + 1e-8)
            base_profile_embedding = np.dot(weighted_profile, embeddings)
        else:
            base_profile_embedding = np.mean(embeddings, axis=0)

        query_embedding = self.retriever_model.encode([test_item.get("question", "")], convert_to_numpy=True)[0]
        self.user_profile_embedding, self.cold_start_route_info = self.cold_router.route(
            test_item=test_item,
            query_embedding=query_embedding,
            user_history_embedding=base_profile_embedding,
            user_history_size=len(corpus),
        )

        self.memory_embeddings = embeddings
        t2 = time.perf_counter()
        self._build_memory_graph(embeddings)
        t3 = time.perf_counter()
        print(f"[TIME] memory graph building stage took {t3 - t2:.4f} seconds")

        if self.enable_cold_start_router and self.cold_start_route_info.get("mode") in {"cohort_only", "blend"}:
            self.graph_center = l2_normalize(
                (1 - self.cold_start_anchor_mix) * self.graph_center
                + self.cold_start_anchor_mix * self.user_profile_embedding
            )

    def _build_memory_graph(self, embeddings):
        n = len(embeddings)
        adjacency = np.zeros((n, n), dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        sim_matrix = np.dot(normalized, normalized.T)
        mask = sim_matrix >= self.sim_threshold
        sim_filtered = np.where(mask, sim_matrix, -np.inf)

        for i in range(n):
            sims_i = sim_filtered[i]
            valid_idx = np.where(np.isfinite(sims_i))[0]
            if valid_idx.size == 0:
                continue
            if valid_idx.size > self.top_k_neighbors:
                top_idx = np.argpartition(sims_i[valid_idx], -self.top_k_neighbors)[-self.top_k_neighbors :]
                keep_idx = valid_idx[top_idx]
            else:
                keep_idx = valid_idx

            edge_weight = sim_matrix[i, keep_idx]
            if self.enable_temporal_profile and self.session_ordinals is not None:
                time_gap = np.abs(self.session_ordinals[i] - self.session_ordinals[keep_idx])
                time_weight = np.exp(-self.temporal_graph_decay * time_gap)
                edge_weight = edge_weight * time_weight
            adjacency[i, keep_idx] = edge_weight

        self.adjacency_matrix = csr_matrix(adjacency)

        n_components = connected_components(self.adjacency_matrix, directed=False)[0]
        if n_components > 1:
            print(f"Warning: Memory graph has {n_components} disconnected components")

        row_sums = adjacency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-8
        transition_matrix = adjacency / row_sums

        pi = np.ones(n) / n
        for _ in range(self.pi_step):
            pi_new = transition_matrix.T @ pi
            if np.linalg.norm(pi_new - pi) < 1e-6:
                break
            pi = pi_new
        self.pi = pi / pi.sum()
        self.long_term_center = np.dot(self.pi, embeddings)

        if self.enable_temporal_profile and self.recency_weights is not None:
            recency_norm = self.recency_weights / (self.recency_weights.sum() + 1e-8)
            self.short_term_center = np.dot(recency_norm, embeddings)
            self.graph_center = (1 - self.short_term_blend) * self.long_term_center + self.short_term_blend * self.short_term_center
        else:
            self.short_term_center = self.long_term_center
            self.graph_center = self.long_term_center

    def _mem_pagerank(self, query_embedding, max_iter=20, tol=1e-6):
        n = self.memory_embeddings.shape[0]

        s_q_raw = np.dot(self.memory_embeddings, query_embedding)
        s_q = np.exp(s_q_raw / 0.1)
        if self.enable_temporal_profile and self.temporal_weights is not None:
            s_q = s_q * self.temporal_weights
        s_q /= s_q.sum() + 1e-8

        row, col = self.adjacency_matrix.nonzero()
        base_weights = self.adjacency_matrix.data
        n_edges = len(base_weights)

        t_q = np.zeros_like(base_weights)
        for idx in range(n_edges):
            i, j = row[idx], col[idx]
            path_vec = self.memory_embeddings[i] + self.memory_embeddings[j]
            path_sim = np.dot(query_embedding, path_vec)
            t_q[idx] = 1 / (1 + np.exp(-path_sim / (np.linalg.norm(query_embedding) + 1e-8)))

        out_degree = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        d_inv_sqrt = 1.0 / np.sqrt(out_degree + 1e-8)
        t_q_scaled = d_inv_sqrt[row] * base_weights * t_q * d_inv_sqrt[col]

        transition = csr_matrix((t_q_scaled, (row, col)), shape=(n, n))

        pr = np.ones(n) / n
        for _ in range(max_iter):
            new_pr = (1 - self.damping_factor) * s_q + self.damping_factor * transition.T.dot(pr)
            if np.linalg.norm(new_pr - pr) < tol:
                break
            pr = new_pr

        return pr / (pr.sum() + 1e-8)

    def query(self, questions, top_k=5):
        if self.index is None:
            raise ValueError("Index has not been built. Please call build_index() first.")

        q_embeddings = self.retriever_model.encode(questions, convert_to_numpy=True)
        candidate_k = min(len(self.chunks), max(top_k, top_k * self.seed_candidate_multiplier))
        d_scores, i_rankings = self.index.search(q_embeddings, candidate_k)
        d_scores, i_rankings = self._temporal_rerank(d_scores, i_rankings, top_k)
        d_scores, i_rankings = self._profile_rerank(d_scores, i_rankings, top_k)

        rankings_id = self.segment_ids[i_rankings].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[i_rankings].tolist()[0]
        return d_scores, i_rankings, retrieved_chunks, rankings_id

    def query_fake_ada_reason(self, fake, reason, question, top_k=5):
        if self.index is None:
            raise ValueError("Index has not been built. Please call build_index() first.")

        q_embeddings = self.retriever_model.encode(question, convert_to_numpy=True)[0, :]
        g_embeddings = self.graph_center
        u_embeddings = self.user_profile_embedding

        reason_embd = self.retriever_model.encode([reason], convert_to_numpy=True)[0]
        fake = fake if isinstance(fake, list) else [str(fake)]
        prf_embd = self.retriever_model.encode(fake, convert_to_numpy=True)
        combined_vec = prf_embd.mean(0)

        if self.enable_temporal_profile or self.enable_cold_start_router:
            gate_logits = np.array(
                [
                    cosine_similarity(q_embeddings[None, :], combined_vec[None, :]).squeeze(),
                    cosine_similarity(q_embeddings[None, :], reason_embd[None, :]).squeeze(),
                    cosine_similarity(q_embeddings[None, :], u_embeddings[None, :]).squeeze(),
                ]
            )
            gate = softmax(gate_logits)
            q_embeddings = q_embeddings + g_embeddings + gate[0] * combined_vec + gate[1] * reason_embd + gate[2] * u_embeddings
        else:
            w1 = 1 + cosine_similarity((q_embeddings[None, :] + g_embeddings[None, :]) / 2, combined_vec[None, :]).squeeze()
            w2 = 1 + cosine_similarity((q_embeddings[None, :] + g_embeddings[None, :]) / 2, reason_embd[None, :]).squeeze()
            q_embeddings = q_embeddings + g_embeddings + w1 * combined_vec + w2 * reason_embd

        candidate_k = min(len(self.chunks), max(top_k, top_k * self.seed_candidate_multiplier))
        d_scores, i_rankings = self.index.search(q_embeddings[None, :], candidate_k)
        d_scores, i_rankings = self._temporal_rerank(d_scores, i_rankings, top_k)
        d_scores, i_rankings = self._profile_rerank(d_scores, i_rankings, top_k)

        rankings_id = self.segment_ids[i_rankings].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[i_rankings].tolist()[0]
        return d_scores, i_rankings, retrieved_chunks, rankings_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="PBR", help="Which model to use.")
    parser.add_argument("--data_type", type=str, default="s", help="Which model to use.")
    parser.add_argument("--retrieval_model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", help="Which model to use.")

    parser.add_argument("--temporal_profile", action="store_true", help="Enable temporal-evolving profile for PBR.")
    parser.add_argument("--temporal_decay_lambda", type=float, default=0.05, help="Lambda in exp(-lambda * age_days).")
    parser.add_argument("--temporal_util_alpha", type=float, default=4.0, help="Alpha for sigmoid(alpha * utility).")
    parser.add_argument("--temporal_util_bias", type=float, default=0.0, help="Bias term in sigmoid(alpha * (utility-bias)).")
    parser.add_argument("--temporal_rerank_beta", type=float, default=0.4, help="Mix ratio for temporal reranking.")
    parser.add_argument("--temporal_graph_decay", type=float, default=0.02, help="Temporal proximity decay in graph edges.")
    parser.add_argument("--short_term_blend", type=float, default=0.5, help="Blend ratio of short-term and long-term anchor.")
    parser.add_argument("--seed_candidate_multiplier", type=int, default=4, help="Over-fetch multiplier before temporal rerank.")

    parser.add_argument("--cold_start_router", action="store_true", help="Enable cohort prototype router for cold-start users.")
    parser.add_argument("--cold_start_prototype_bank", type=str, default="", help="Path to prototype-bank json.")
    parser.add_argument("--cold_start_m0", type=int, default=3, help="History-size threshold for switching to individual profile.")
    parser.add_argument("--cold_start_tau", type=float, default=2.0, help="Blend scheduler tau in beta=1-exp(-m/tau).")
    parser.add_argument(
        "--cold_start_supervised_weight",
        type=float,
        default=0.6,
        help="Weight for supervised prototype in supervised/unsupervised hybrid.",
    )
    parser.add_argument(
        "--cold_start_prefer_supervised",
        dest="cold_start_prefer_supervised",
        action="store_true",
        default=True,
        help="Prefer supervised labels when both supervised and unsupervised prototypes exist.",
    )
    parser.add_argument(
        "--cold_start_prefer_unsupervised",
        dest="cold_start_prefer_supervised",
        action="store_false",
        help="Prefer unsupervised centroid when both supervised and unsupervised prototypes exist.",
    )
    parser.add_argument(
        "--cold_start_label_keys",
        type=str,
        default="department,role,team",
        help="Comma-separated label keys used for supervised prototype lookup.",
    )
    parser.add_argument("--cold_start_anchor_mix", type=float, default=0.35, help="Mix weight of routed profile into graph center.")
    parser.add_argument(
        "--cold_start_seed_profile_alpha",
        type=float,
        default=0.2,
        help="Profile similarity boost in seed retrieval for cohort-only/blend mode.",
    )
    parser.add_argument(
        "--cold_start_rerank_alpha",
        type=float,
        default=0.25,
        help="Profile rerank blend ratio in final retrieval for cohort-only/blend mode.",
    )
    parser.add_argument("--cold_start_debug", action="store_true", help="Enable debug metadata in cold-start routing.")

    parser.add_argument("--k_seed", type=int, default=10, help="K_seed for P-PRF++ seed history sampling.")
    parser.add_argument("--top_k_retrieval", type=int, default=10, help="Top-K used for final retrieval evaluation.")
    parser.add_argument("--save_suffix", type=str, default="", help="Optional suffix for output json file.")

    args = parser.parse_args()
    retriever_model_name = args.retrieval_model_name
    print(retriever_model_name)
    method = args.model_type
    print(method)

    enable_temporal_profile = args.temporal_profile or method.lower() in {"pbr_temporal", "pbr++", "pbr-dynamic"}
    temporal_cfg = {
        "enable_temporal_profile": enable_temporal_profile,
        "temporal_decay_lambda": args.temporal_decay_lambda,
        "temporal_util_alpha": args.temporal_util_alpha,
        "temporal_util_bias": args.temporal_util_bias,
        "temporal_rerank_beta": args.temporal_rerank_beta,
        "temporal_graph_decay": args.temporal_graph_decay,
        "short_term_blend": args.short_term_blend,
        "seed_candidate_multiplier": args.seed_candidate_multiplier,
    }
    print("temporal_profile:", enable_temporal_profile)

    method_lc = method.lower()
    enable_cold_start_router = args.cold_start_router or method_lc in {
        "pbr_cold",
        "pbr_coldstart",
        "pbr-coldstart",
        "pbr++_cold",
        "pbr++_coldstart",
    }
    cold_start_label_keys = [x.strip() for x in args.cold_start_label_keys.split(",") if x.strip()]
    if enable_cold_start_router and not args.cold_start_prototype_bank.strip():
        raise ValueError("cold-start router is enabled, but --cold_start_prototype_bank is empty.")
    if enable_cold_start_router:
        prototype_bank_obj = load_prototype_bank(args.cold_start_prototype_bank)
    else:
        prototype_bank_obj = None
    coldstart_cfg = {
        "enable_cold_start_router": enable_cold_start_router,
        "cold_start_m0": args.cold_start_m0,
        "cold_start_tau": args.cold_start_tau,
        "cold_start_supervised_weight": args.cold_start_supervised_weight,
        "cold_start_prefer_supervised": args.cold_start_prefer_supervised,
        "cold_start_label_keys": cold_start_label_keys,
        "cold_start_anchor_mix": args.cold_start_anchor_mix,
        "cold_start_seed_profile_alpha": args.cold_start_seed_profile_alpha,
        "cold_start_rerank_alpha": args.cold_start_rerank_alpha,
        "cold_start_debug": args.cold_start_debug,
        "cold_start_prototype_bank": args.cold_start_prototype_bank,
        "cold_start_prototype_bank_obj": prototype_bank_obj,
    }
    coldstart_cfg_for_log = {k: v for k, v in coldstart_cfg.items() if k != "cold_start_prototype_bank_obj"}
    print("cold_start_router:", enable_cold_start_router)

    data_type = args.data_type
    in_file = f"./data/longmemeval_data/longmemeval_{data_type}.json"
    print(in_file)

    tag_parts = ["PBR"]
    if enable_temporal_profile:
        tag_parts.append("temporal")
    if enable_cold_start_router:
        tag_parts.append("coldstart")
    model_tag = "_".join(tag_parts)
    save_path = in_file.replace(".json", f"_{model_tag}{args.save_suffix}.json")
    print(save_path)

    in_data = json.load(open(in_file, encoding="utf-8"))
    retriever_model = SentenceTransformer(retriever_model_name, trust_remote_code=True)

    results = []
    out_json = []
    uttr_prompt = []
    rea_prompt = []

    for test_item in tqdm(in_data):
        question = test_item["question"]
        retriever = RAGRetriever(
            retriever_model,
            retriever_model_name,
            data_type,
            temporal_cfg=temporal_cfg,
            coldstart_cfg=coldstart_cfg,
        )
        retriever.build_index(test_item)
        questions = [question]
        if enable_temporal_profile or enable_cold_start_router:
            d_scores, i_rankings, retrieved_chunks, rankings_id = retriever.query_seed_weighted(questions, top_k=args.k_seed)
        else:
            d_scores, i_rankings, retrieved_chunks, rankings_id = retriever.query(questions, top_k=args.k_seed)
        docs = "\n".join(retrieved_chunks[: args.k_seed])
        p_uttr_prompt, p_rea_prompt = gen_retrieval_prompt_fake_ada_reason_10(question, docs, "")
        uttr_prompt.append(p_uttr_prompt)
        rea_prompt.append(p_rea_prompt)

    print("begin_generation")
    async_uttr_responses = asyncio.run(run_async(uttr_prompt, model="gpt-4o-mini"))
    async_res_responses = asyncio.run(run_async(rea_prompt, model="gpt-4o-mini"))
    print("end_generation")

    for idx, test_item in tqdm(enumerate(in_data)):
        question = test_item["question"]
        retriever = RAGRetriever(
            retriever_model,
            retriever_model_name,
            data_type,
            temporal_cfg=temporal_cfg,
            coldstart_cfg=coldstart_cfg,
        )
        retriever.build_index(test_item)
        questions = [question]
        try:
            fake_10 = load_json_res(async_uttr_responses[idx])["candidates"]
        except Exception:
            print(async_uttr_responses[idx])
            fake_10 = [async_uttr_responses[idx]]

        ada_reason = async_res_responses[idx]
        d_scores, i_rankings, retrieved_chunks, rankings_id = retriever.query_fake_ada_reason(
            fake_10,
            ada_reason,
            questions,
            top_k=args.top_k_retrieval,
        )
        corpus_ids = [item for item in test_item["haystack_session_ids"]]

        ret_res = []
        rankings = []
        for res, ids in zip(retrieved_chunks, i_rankings[0].tolist()):
            ret_res.append({"corpus_id": ids, "text": res})
            rankings.append(ids)

        cur_results = {
            "question_id": test_item["question_id"],
            "question_type": test_item["question_type"],
            "question": test_item["question"],
            "answer": test_item["answer"],
            "question_date": test_item["question_date"],
            "haystack_dates": test_item["haystack_dates"],
            "haystack_sessions": test_item["haystack_sessions"],
            "haystack_session_ids": test_item["haystack_session_ids"],
            "answer_session_ids": test_item["answer_session_ids"],
            "retrieval_results": {
                "query": question,
                "ranked_items": ret_res,
                "metrics": {"session": {}, "turn": {}},
            },
            "fake_10": fake_10,
            "reason": ada_reason,
        }
        if enable_temporal_profile:
            cur_results["temporal_profile_cfg"] = temporal_cfg
        if enable_cold_start_router:
            cur_results["cold_start_cfg"] = coldstart_cfg_for_log
            cur_results["cold_start_route_info"] = retriever.cold_start_route_info

        correct_docs = list(set([doc_id for doc_id in corpus_ids if "answer" in doc_id]))
        for k in [1, 3, 5, 10]:
            recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
            cur_results["retrieval_results"]["metrics"]["session"].update(
                {
                    f"recall_any@{k}": recall_any,
                    f"recall_all@{k}": recall_all,
                    f"ndcg_any@{k}": ndcg_any,
                }
            )

        out_json.append(cur_results)
        results.append(cur_results)

    averaged_results = {"session": {}, "turn": {}}
    ignored_qs_abstention, ignored_qs_no_target = set(), set()
    for k in results[0]["retrieval_results"]["metrics"]["session"]:
        try:
            results_list = []
            for eval_entry in results:
                if "_abs" in eval_entry["question_id"]:
                    ignored_qs_abstention.add(eval_entry["question_id"])
                    continue
                has_target = any(
                    ("has_answer" in turn) and (turn["has_answer"])
                    for turn in [x for y in eval_entry["haystack_sessions"] for x in y if x["role"] == "user"]
                )
                if not has_target:
                    ignored_qs_no_target.add(eval_entry["question_id"])
                    continue
                results_list.append(eval_entry["retrieval_results"]["metrics"]["session"][k])
            averaged_results["session"][k] = np.mean(results_list)
        except Exception:
            continue

    print(json.dumps(averaged_results))
    save_json(out_json, save_path)
