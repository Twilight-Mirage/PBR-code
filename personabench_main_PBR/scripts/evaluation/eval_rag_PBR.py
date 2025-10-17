import os
import sys
from pathlib import Path

# 自动获取 personabench 的绝对路径（跨平台兼容）
personabench_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, personabench_path) 

import argparse
import faiss
import tqdm
import logging
import numpy as np
import json
from personabench.utils.shared_args import add_shared_arguments
from personabench.utils.utils import get_api_key, load_json, convert_json_to_plain_text, save_json, set_random_seed, setup_logger
from sentence_transformers import SentenceTransformer
from personabench.models.openai_yy import OpenAIGPT
import asyncio
import time
from eval import eval as final_eval
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import cvxpy as cp
import ot
from scipy.special import softmax

def safe_generate(model, prompt, max_retries=10):
    for attempt in range(max_retries):
        try:
            return model.generate(prompt)
        except:
            print(f"Attempt {attempt+1} failed  Retrying...")
            time.sleep(2 * (attempt + 1))  # 指数退避
    raise RuntimeError("Failed to generate after retries.")

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
    prompt = {"instruction": instruction}
    response = safe_generate(model, prompt, max_retries=3)
    query_dict = load_json_res(response)
    # print(query_dict)
    try:
        query_ls = query_dict['candidates']
    except:
        response = safe_generate(model, prompt, max_retries=3)
        query_dict = load_json_res(response)
        query_ls = query_dict['candidates']
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
    prompt = {"instruction": instruction}
    response = safe_generate(model, prompt, max_retries=3)
    # print(response)

    return query_ls,response



def build_rag_corpus(community_path,noise,model):
    conversation_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "conversation_data_all.json"))
    user_ai_interaction_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "user_ai_interaction_data_all.json"))
    purchase_history_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "purchase_history_data_all.json"))
    conversation_data_all_lookup = {entry["Name"]: entry["Data"] for entry in conversation_data_all}
    user_ai_interaction_data_all_lookup = {entry["Name"]: entry["Data"] for entry in user_ai_interaction_data_all}
    purchase_history_data_all_lookup = {entry["Name"]: entry["Data"] for entry in purchase_history_data_all}
    return conversation_data_all_lookup, user_ai_interaction_data_all_lookup,purchase_history_data_all_lookup

def extract_gt_segment_ids(QA, qa_gt_context_all_lookup, num_chunks):
    gt_segment_ids_retrieval = []
    for qa in QA:
        q_id = qa["q_id"]
        question_gt_ids = []
        for individual_ids in qa_gt_context_all_lookup[q_id]["segment_id"].values():
            question_gt_ids.append(individual_ids[0])
        if len(question_gt_ids) < num_chunks:
            additional_count = num_chunks - len(question_gt_ids)
            # Repeat the list to fill up the required length
            repeat_part = question_gt_ids * (additional_count // len(question_gt_ids))
            remaining_part = question_gt_ids[:additional_count % len(question_gt_ids)]
            question_gt_ids += repeat_part + remaining_part
        elif len(question_gt_ids) > num_chunks:
            raise ValueError(f"find {len(question_gt_ids)} ground truth, which is larger than {num_chunks}")
        gt_segment_ids_retrieval.append(question_gt_ids)
    return np.array(gt_segment_ids_retrieval)

def create_vector_base(documents_dict, retriever_model=None):
    chunks = []
    segment_ids = []
    for document_type, document_data in documents_dict.items():
        if document_type != "conversation_data":
            for data_entry in document_data:
                segment_id = data_entry["segment_id"]
                chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                chunks.append(chunk)
                segment_ids.append(segment_id)
        else:
            # deal conversation data
            for sub_conversation_data in document_data:
                for data_entry in sub_conversation_data["Conversations"]:
                    segment_id = data_entry["segment_id"]
                    chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                    chunks.append(chunk)
                    segment_ids.append(segment_id)
    if retriever_model:
        embeddings = retriever_model.encode(chunks, convert_to_numpy=True)
        # print(type(embeddings))
        # print(embeddings.shape)
        user_embd_mean = np.mean(embeddings,axis = 0)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    else:
        index = None
    return chunks, index, np.array(segment_ids), user_embd_mean

class RAGRetriever:
    def __init__(self, retriever_model, retriever_model_name,num_neighbor,threshold):
        """
        retriever_model: 一个具有 encode(texts, convert_to_numpy=True) 接口的向量化模型
        """
        self.retriever_model = retriever_model
        self.index = None
        self.chunks = []
        self.segment_ids = []
        self.user_embd_mean = None

        # 新增MemPageRank相关属性
        self.memory_embeddings = None  # 存储所有记忆片段的嵌入
        self.adjacency_matrix = None  # 邻接矩阵
        self.memory_graph = None  # 图结构表示
        self.damping_factor = 0.85  # PageRank阻尼系数
        print(num_neighbor,threshold)
        self.sim_threshold  = float(threshold)
        self.top_k_neighbors = int(num_neighbor)

    def build_index(self, documents_dict,user_name):
        """
        从 documents_dict 构建向量索引。
        """
        chunks = []
        segment_ids = []
        for document_type, document_data in documents_dict.items():
            if document_type != "conversation_data":
                for data_entry in document_data:
                    segment_id = data_entry["segment_id"]
                    chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                    chunks.append(chunk)
                    segment_ids.append(segment_id)
            else:
                # deal conversation data
                for sub_conversation_data in document_data:
                    for data_entry in sub_conversation_data["Conversations"]:
                        segment_id = data_entry["segment_id"]
                        chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                        chunks.append(chunk)
                        segment_ids.append(segment_id)

        corpus, corpus_ids = chunks, segment_ids

        embeddings = self.retriever_model.encode(corpus, convert_to_numpy=True)

        user_text = []
        for document_type, document_data in documents_dict.items():
            user_content = ''
            if document_type != "conversation_data":
                for data_entry in document_data:
                    segment_id = data_entry["segment_id"]
                    if 'user_ai_interaction' in data_entry.keys():
                        for data_conv in data_entry["user_ai_interaction"]:
                            if data_conv['role']=='user':
                                user_content+=passage['content']
                                user_content+='\n'
                        user_text.append(user_content)
                    else:
                        chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                        user_text.append(chunk)

            else:
                # deal conversation data
                for sub_conversation_data in document_data:
                    
                    for data_entry in sub_conversation_data["Conversations"]:
                        segment_id = data_entry["segment_id"]
                        for passage in data_entry['conversation']:
                            if passage['role']==user_name:
                                user_content+=passage['content']
                                user_content+='\n'
                        user_text.append(user_content)
            
        user_embd = self.retriever_model.encode(user_text, convert_to_numpy=True)

        self.user_embd_mean = np.mean(embeddings, axis=0)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.chunks = corpus
        self.segment_ids = np.array(corpus_ids)
        self.memory_embeddings = embeddings
        self._build_memory_graph(embeddings)
    
    def _build_memory_graph(self, embeddings):
        """构建记忆图邻接矩阵并计算结构感知图中心"""
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
            # 取出满足阈值的相似度
            sims_i = sim_filtered[i]
            valid_idx = np.where(np.isfinite(sims_i))[0]
            if valid_idx.size == 0:
                continue
            keep_idx = valid_idx

            # 若超过 top_k_neighbors，就进一步筛 K 个最大
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
        transition_matrix = adjacency / row_sums  # shape (n, n)
        pi = np.ones(n) / n
        for _ in range(50):  # fixed max iterations
            pi_new = transition_matrix.T @ pi
            if np.linalg.norm(pi_new - pi) < 1e-6:
                break
            pi = pi_new
        self.pi = pi/pi.sum() 
        self.graph_center = np.dot(self.pi, embeddings)  # shape (d,)


        
    def query(self, questions, top_k=5):
        """
        输入 questions (list of str)，返回 top_k 检索结果。
        """
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
        q_embeddings = g_embeddings +  w1 * combined_vec+ w2 * reason_embd + q_embeddings

        D, I = self.index.search(q_embeddings[None,:], top_k)
        rankings_id =self.segment_ids[I].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[I].tolist()[0]
        return D, I, retrieved_chunks, rankings_id

def eval_rag(args):
    model = args.model_name
    retriever = args.retrieval_model
    noise = args.noise
    num_neighbor = args.num_neighbor
    threshold = args.threshold
    test_result_dir = os.path.join(args.log_dir, args.save_dir)
    community_ids = [r.strip() for r in args.test_community_ids.split(',')]
    results_all_community = []
    save_path = os.path.join(test_result_dir, f"results_all_community_chunk_{args.num_chunks}")
    os.makedirs(save_path, exist_ok=True)
    results_all_community.append({"Model": f"{model}+{retriever}", "Results": []})
    retriever_model = SentenceTransformer(retriever, trust_remote_code=True)
    gen_model = OpenAIGPT('', model="gpt-4o-mini")
    print('Currnet method:',args.model_type)
    ret_name = args.retrieval_model
    if '/' in retriever:
        ret_name = args.retrieval_model.split('/')[1]
    for community_id in community_ids:
        community_path = os.path.join(args.data_dir, "synthetic_data", community_id)
        if args.model_type=='PBR':
            eval_info_all = os.path.join(community_path, 'eval_info', f'eval_info_all_PBR_{ret_name}.json')
        else:
            eval_info_all = os.path.join(community_path, 'eval_info', f'eval_info_all.json')
        if os.path.exists(eval_info_all):
            path = eval_info_all
            eval_info_all = load_json(path)
        else:
            path = os.path.join(community_path, 'eval_info', f'eval_info_all.json')
            eval_info_all = load_json(path)
        if args.model_type=='PBR':
            save_eval_path = os.path.join(community_path, 'eval_info', f'eval_info_all_{args.model_type}_{ret_name}.json')
        else:
            save_eval_path = os.path.join(community_path, 'eval_info', f'eval_info_all_{args.model_type}.json')
        print(path)
        print(save_eval_path)
        qa_gt_context_all = load_json(os.path.join(community_path, 'eval_info', f'qa_gt_context_all_noise_{noise}.json'))
        qa_gt_context_all_lookup = {entry["q_id"]: entry for entry in qa_gt_context_all}
        conver_all, user_ai_all,purchase_his_all = build_rag_corpus(community_path,noise,retriever_model)
        eval_info_all_res = []
        for eval_info_dict in eval_info_all[:args.num_people]:
            name = eval_info_dict["Name"]
            eval_info = eval_info_dict["Eval_Info"]
            conversation_data = conver_all[name]
            user_ai_interaction_data = user_ai_all[name]
            purchase_history_data = purchase_his_all[name]
            documents_dict = {"conversation_data": conversation_data, "user_ai_interaction_data": user_ai_interaction_data, "purchase_history_data": purchase_history_data}
            QA = eval_info["qa"]
            retriever = RAGRetriever(retriever_model,args.retrieval_model,num_neighbor,threshold )
            retriever.build_index(documents_dict,name)
            gt_segment_ids_retrieval = extract_gt_segment_ids(QA, qa_gt_context_all_lookup, args.num_chunks)
            for i,entry in enumerate(QA):
                question = entry['question']
                questions = [question]
                D, I, retrieved_chunks,rankings_id = retriever.query(questions, top_k=5)
                if args.model_type=="base":
                    ques = question
                elif "PBR" in args.model_type:
                    docs = '\n'.join(retrieved_chunks[:5])
                    if "fake_10" in entry.keys():
                        fake_ans = entry['fake_10']
                        reason = entry['reason']
                    else:
                        fake_ans,reason = gen_retrieval_prompt_fake_ada_reason_10(question, docs, gen_model)
                    
                    entry['fake_10'] = fake_ans
                    entry['reason'] = reason
                    ques = str(fake_ans) + reason
                    D, I, retrieved_chunks,rankings_id = retriever.query_fake_ada_reason(fake_ans,reason, questions, top_k=5)
                else:
                    ques = entry['general_response']['candidates']
                logging.info("Retrieve finished")
                prediction = ''
                results_all_community[-1]["Results"].append({"q_id": entry["q_id"],
                                                                "question": entry["question"],
                                                                "answer": entry["answer"],
                                                                "prediction":prediction,
                                                                "type": entry["type"],
                                                                "difficulty": entry["difficulty"],
                                                                "retrieved_segment_ids": list(rankings_id),
                                                                "ground_truth_segment_ids": qa_gt_context_all_lookup[entry["q_id"]]["segment_id"],
                                                                }
                                                                )
            eval_info_all_res.append(eval_info_dict)
            
            save_json(eval_info_all_res, save_eval_path)
    save_json(results_all_community, os.path.join(save_path, f"result_noise_{noise}.json"))
    final_eval(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_shared_arguments(parser)
    parser.add_argument("--model_type", type=str, default="fake_3", help="Path to save eval results.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to save eval results.")
    parser.add_argument("--save_dir", type=str, default="eval_data_v1")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunks for retrieval tasks")

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--retrieval_model", type=str)
    parser.add_argument("--test_community_ids", type=str, default="community_0,community_1", help="Comma-separated list of community ids to test")
    parser.add_argument('--num_people', type=int, default=3)
    parser.add_argument('--noise', type=str)
    parser.add_argument('--num_neighbor', default="5", type=str)
    parser.add_argument('--threshold', default="0.5", type=str)

    args = parser.parse_args()
    setup_logger(args.verbose)
    set_random_seed(args.seed)
    eval_rag(args)