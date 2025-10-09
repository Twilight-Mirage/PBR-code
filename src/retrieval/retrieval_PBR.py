import json
import numpy as np

import os
import sys
from pathlib import Path

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import time
import argparse
import faiss
from sentence_transformers import SentenceTransformer
from src.retrieval.eval_utils import evaluate_retrieval, evaluate_retrieval_turn2session
from tqdm import tqdm
import asyncio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import cvxpy as cp
import ot
from scipy.special import softmax

from async_llm import run_async
import asyncio



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
        print("Outpu t is not in JSON format")
        data = {}
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        if isinstance(data, dict) or isinstance(data, list):
            json.dump(data, f, indent=4)
        elif isinstance(data, str):
            try:
                # Verify if it's a valid JSON string
                json.loads(data)
                f.write(data)
            except json.JSONDecodeError:
                raise ValueError("Provided string is not valid JSON")
        else:
            raise TypeError("Data must be a list/dictionary or a JSON string")
        


def gen_retrieval_prompt_fake_ada_reason_10(query,doc,model):
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
    P_utt_prompt = instruction
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

    return P_utt_prompt,p_rea_prompt


def remove_key(json_data, key_to_remove):
    if isinstance(json_data, dict):
        return {k: remove_key(v, key_to_remove) for k, v in json_data.items() if k != key_to_remove}
    elif isinstance(json_data, list):
        return [remove_key(item, key_to_remove) for item in json_data]
    else:
        return json_data
    
def convert_json_to_plain_text(json_data, exclude=None):
    """
    conver json data to plain text for RAG
    exclude: the keys that should not be included
    """
    
    for key_to_remove in exclude:
        json_data = remove_key(json_data, key_to_remove)
    plian_text = json.dumps(json_data, separators=(',', ':'))
    return plian_text


class RAGRetriever:
    def __init__(self, retriever_model, retriever_model_name,data_type = 's'):
        self.retriever_model = retriever_model
        self.index = None
        self.chunks = []
        self.segment_ids = []
        self.user_embd_mean = None

        self.memory_embeddings = None  
        self.adjacency_matrix = None  
        self.memory_graph = None  
        self.damping_factor = 0.85  
        if data_type == 'm':
            self.sim_threshold  = 0.75 
            self.top_k_neighbors = 50  
        else:
            self.sim_threshold  = 0.75 
            self.top_k_neighbors = 10  
        self.pi_step = 50

    def build_index(self, test_item):
        corpus, corpus_ids, corpus_timestamps = [], [], []
        for cur_sess_id, sess_entry, ts in zip(test_item['haystack_session_ids'], test_item['haystack_sessions'], test_item['haystack_dates']):
            user_data = []
            for item in sess_entry:
                if item['role']=='user':
                    user_data.append(item['content'])
            tmp_data = {
                'date': ts,
                'conversation':user_data
            }
            segment_id = cur_sess_id
            plian_text = json.dumps(tmp_data, separators=(',', ':'))
            corpus.append(plian_text)
            corpus_ids.append(segment_id)

        embeddings = self.retriever_model.encode(corpus, convert_to_numpy=True)
        self.user_embd_mean = np.mean(embeddings, axis=0)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.chunks = corpus
        self.segment_ids = np.array(corpus_ids)
        # 构建记忆图
        self.memory_embeddings = embeddings
        t2 = time.perf_counter()
        self._build_memory_graph(embeddings)
        t3 = time.perf_counter()
        print(f"[TIME] first memory graph building stage took {t3 - t2:.4f} seconds")
    
    def _build_memory_graph(self, embeddings):
        n = len(embeddings)
        adjacency = np.zeros((n, n))

        # Normalize to unit vectors for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        sim_matrix = np.dot(normalized, normalized.T)
        mask = sim_matrix >= self.sim_threshold
        sim_filtered = np.where(mask, sim_matrix, -np.inf) 
        adjacency = np.zeros_like(sim_matrix, dtype=np.float32)
        for i in range(n):
            sims_i = sim_filtered[i]
            valid_idx = np.where(np.isfinite(sims_i))[0]
            if valid_idx.size == 0:
                continue
            keep_idx = valid_idx

            if valid_idx.size > self.top_k_neighbors:
                top_idx = np.argpartition(sims_i[valid_idx],
                                        -self.top_k_neighbors)[-self.top_k_neighbors:]
                keep_idx = valid_idx[top_idx]
            else:
                keep_idx = valid_idx

            adjacency[i, keep_idx] = sim_matrix[i, keep_idx]

        self.adjacency_matrix = csr_matrix(adjacency)

        # Warn if disconnected
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
        self.pi = pi/pi.sum()  
        self.graph_center = np.dot(self.pi, embeddings) 


    def _mem_pagerank(self, query_embedding, max_iter=20, tol=1e-6):

        n = self.memory_embeddings.shape[0]

        S_q_raw = np.dot(self.memory_embeddings, query_embedding)
        S_q = np.exp(S_q_raw / 0.1)  
        S_q /= S_q.sum()  

        row, col = self.adjacency_matrix.nonzero()
        base_weights = self.adjacency_matrix.data  
        n_edges = len(base_weights)

        T_q = np.zeros_like(base_weights)
        for idx in range(n_edges):
            i, j = row[idx], col[idx]
            path_vec = self.memory_embeddings[i] + self.memory_embeddings[j]
            path_sim = np.dot(query_embedding, path_vec)
            T_q[idx] = 1 / (1 + np.exp(-path_sim / np.linalg.norm(query_embedding)))  

        out_degree = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        D_inv_sqrt = 1.0 / np.sqrt(out_degree + 1e-8)
        T_q_scaled = D_inv_sqrt[row] * base_weights * T_q * D_inv_sqrt[col]

        transition = csr_matrix((T_q_scaled, (row, col)), shape=(n, n))  

        pr = np.ones(n) / n
        for _ in range(max_iter):
            new_pr = (1 - self.damping_factor) * S_q + self.damping_factor * transition.T.dot(pr)
            if np.linalg.norm(new_pr - pr) < tol:
                break
            pr = new_pr

        return pr / pr.sum()


    def query(self, questions, top_k=5):
        
        if self.index is None:
            raise ValueError("Index has not been built. Please call build_index() first.")

        q_embeddings = self.retriever_model.encode(questions, convert_to_numpy=True)
        D, I = self.index.search(q_embeddings, top_k)
        # print(I.shape)
        # print(self.segment_ids)
        rankings_id =self.segment_ids[I].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[I].tolist()[0]
        return D, I, retrieved_chunks, rankings_id

    def query_fake_ada_reason(self, fake, reason, question, top_k=5):
        if self.index is None:
            raise ValueError("Index has not been built. Please call build_index() first.")
        # best version
        q_embeddings = self.retriever_model.encode(question, convert_to_numpy=True)[0,:]
        g_embeddings = self.graph_center
        reason_embd = self.retriever_model.encode(reason, convert_to_numpy=True)
        prf_embd = self.retriever_model.encode(fake, convert_to_numpy=True)
        combined_vec = prf_embd.mean(0)
        w1 =1+cosine_similarity((q_embeddings[None,:]+g_embeddings)/2, combined_vec[None,:]).squeeze()
        w2 =1+cosine_similarity((q_embeddings[None,:]+g_embeddings)/2, reason_embd[None,:]).squeeze()
        q_embeddings = q_embeddings + g_embeddings + w1 * combined_vec + w2*reason_embd
        
        D, I = self.index.search(q_embeddings[None,:], top_k)
        rankings_id =self.segment_ids[I].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[I].tolist()[0]
        return D, I, retrieved_chunks, rankings_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="PBR", help="Which model to use.")
    parser.add_argument("--data_type", type=str, default="s", help="Which model to use.")
    parser.add_argument("--retrieval_model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", help="Which model to use.")

    
    args = parser.parse_args()
    retriever_model_name = args.retrieval_model_name
    print(retriever_model_name)
    method = args.model_type
    print(method)
    data_type = args.data_type
    in_file = f"./data/longmemeval_data/longmemeval_{data_type}.json"
    print(in_file)
    save_path = in_file.replace('.json','_PBR.json')
    print(save_path)
    in_data = json.load(open(in_file))
    retriever_model = SentenceTransformer(retriever_model_name, trust_remote_code=True)
    
    results= []
    out_json = []
    uttr_prompt = []
    rea_prompt = []
    for test_item in tqdm(in_data):
        question = test_item['question']
        retriever = RAGRetriever(retriever_model,retriever_model_name, data_type)
        retriever.build_index(test_item)
        questions = [question]
        D, I, retrieved_chunks,rankings_id = retriever.query(questions, top_k=10)
        docs = '\n'.join(retrieved_chunks[:10])
        p_uttr_prompt,p_rea_prompt = gen_retrieval_prompt_fake_ada_reason_10(question, docs, '')
        uttr_prompt.append(p_uttr_prompt)
        rea_prompt.append(p_rea_prompt)
    print('begin_generation')
    async_uttr_responses = asyncio.run(run_async(uttr_prompt,model="gpt-4o-mini"))
    async_res_responses = asyncio.run(run_async(rea_prompt,model="gpt-4o-mini"))
    print('end_generation')

    for idx, test_item in tqdm(enumerate(in_data)):
        question = test_item['question']
        retriever = RAGRetriever(retriever_model,retriever_model_name, data_type)
        retriever.build_index(test_item)
        questions = [question]
        try:
            fake_10 = load_json_res(async_uttr_responses[idx])
            fake_10 = fake_10['candidates']
        except:
            print(async_uttr_responses[idx])
            fake_10 = [async_uttr_responses[idx]]
        ada_reason = async_res_responses[idx]
        D, I, retrieved_chunks,rankings_id = retriever.query_fake_ada_reason(fake_10, ada_reason, questions, top_k=10)
        corpus_ids = [item for item in test_item['haystack_session_ids']]
        ret_res = []
        rankings = []
        for res,ids in zip(retrieved_chunks,I[0].tolist()):
            tmp_rank = {
                    'corpus_id': ids,
                    'text': res,
                }
            rankings.append(ids)
            ret_res.append(tmp_rank)
        cur_results = {
            'question_id': test_item['question_id'],
            'question_type': test_item['question_type'],
            'question': test_item['question'],
            'answer': test_item['answer'],
            'question_date': test_item['question_date'],
            'haystack_dates': test_item['haystack_dates'],
            'haystack_sessions': test_item['haystack_sessions'],
            'haystack_session_ids': test_item['haystack_session_ids'],
            'answer_session_ids': test_item['answer_session_ids'],
            'retrieval_results': {
                'query': question,
                'ranked_items': ret_res,
                'metrics': {
                    'session': {},
                    'turn': {}
                    }
                }
            }

        cur_results['fake_10'] = fake_10
        cur_results['reason'] = ada_reason

        correct_docs = list(set([doc_id for doc_id in corpus_ids if "answer" in doc_id]))

        for k in [1,3,5,10]:
            recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
            cur_results['retrieval_results']['metrics']['session'].update({
                'recall_any@{}'.format(k): recall_any,
                'recall_all@{}'.format(k): recall_all,
                'ndcg_any@{}'.format(k): ndcg_any
            })

        out_json.append(cur_results)
        results.append(cur_results)
    averaged_results = {
        'session': {},
        'turn': {}
    }
    ignored_qs_abstention, ignored_qs_no_target = set(), set()
    for k in results[0]['retrieval_results']['metrics']['session']:
        try:
            results_list = []
            for eval_entry in results:
                # will skip abstention instances for reporting the metric
                if '_abs' in eval_entry['question_id']:
                    ignored_qs_abstention.add(eval_entry['question_id'])
                    continue
                # will also skip instances with no target labels
                if not any(('has_answer' in turn) and (turn['has_answer']) for turn in [x for y in eval_entry['haystack_sessions'] for x in y if x['role'] == 'user']):
                    ignored_qs_no_target.add(eval_entry['question_id'])
                    continue
                results_list.append(eval_entry['retrieval_results']['metrics']['session'][k])
                
            averaged_results['session'][k] = np.mean(results_list)
        except:
            continue
    print(json.dumps(averaged_results))
    save_json(out_json,save_path)
