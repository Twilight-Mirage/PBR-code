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
from eval import eval


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


def eval_rag(args):
    model = args.model_name
    retriever = args.retrieval_model
    noise = args.noise
    test_result_dir = os.path.join(args.log_dir, args.save_dir)
    community_ids = [r.strip() for r in args.test_community_ids.split(',')]
    results_all_community = []
    save_path = os.path.join(test_result_dir, f"results_all_community_chunk_{args.num_chunks}")
    os.makedirs(save_path, exist_ok=True)
    results_all_community.append({"Model": f"{model}+{retriever}", "Results": []})
    retriever_model = SentenceTransformer(retriever, trust_remote_code=True)
    for community_id in community_ids:
        community_path = os.path.join(args.data_dir, "synthetic_data", community_id)
        eval_info_all = load_json(os.path.join(community_path, 'eval_info', f'eval_info_all_{model}_v10.json'))
        qa_gt_context_all = load_json(os.path.join(community_path, 'eval_info', f'qa_gt_context_all_noise_{noise}.json'))
        qa_gt_context_all_lookup = {entry["q_id"]: entry for entry in qa_gt_context_all}
        conver_all, user_ai_all,purchase_his_all = build_rag_corpus(community_path,noise,retriever_model)
        for eval_info_dict in eval_info_all[:args.num_people]:
            name = eval_info_dict["Name"]
            eval_info = eval_info_dict["Eval_Info"]
            conversation_data = conver_all[name]
            user_ai_interaction_data = user_ai_all[name]
            purchase_history_data = purchase_his_all[name]
            documents_dict = {"conversation_data": conversation_data, "user_ai_interaction_data": user_ai_interaction_data, "purchase_history_data": purchase_history_data}
            chunks, index, segment_ids, user_embd_mean = create_vector_base(documents_dict, retriever_model)
            QA = eval_info["qa"]
            questions = []
            for entry in QA:
                if args.eval_SAT_key == "all":
                    query_ls = list(entry['SAT_query'].values())
                    questions.append('\n'.join(query_ls))
                elif args.eval_SAT_key == "all_add":
                    query_ls = list(entry['SAT_query'].values())
                    questions.append(query_ls)
                elif args.eval_SAT_key == "base":
                    questions.append(entry['question'])
                elif args.eval_SAT_key == "add_origin":
                    query_ls = list(entry['SAT_query'].values())
                    query_ls = [entry['question']] + query_ls
                    questions.append('\n'.join(query_ls))
                elif args.eval_SAT_key == "no":
                    query_ls =[]
                    questions.append(''.join(query_ls))
                else:
                    questions.append(entry['SAT_query'][args.eval_SAT_key])
            # questions = [entry['SAT_query'][args.eval_SAT_key] for entry in QA]
            gt_segment_ids_retrieval = extract_gt_segment_ids(QA, qa_gt_context_all_lookup, args.num_chunks)
            if retriever == "gt-context":  # use ground-truth context
                id_to_index = {id_: idx for idx, id_ in enumerate(segment_ids)}
                I = np.vectorize(id_to_index.get)(gt_segment_ids_retrieval)
            else:
                if args.eval_SAT_key == "all_add":
                    # w= [0.2,0.3,0.5]
                    loc = [item[0] for item in questions]
                    iloc = [item[1] for item in questions]
                    perloc = [item[2] for item in questions]
                    q_embeddings = retriever_model.encode(loc, convert_to_numpy=True) + retriever_model.encode(iloc, convert_to_numpy=True)+ retriever_model.encode(perloc, convert_to_numpy=True)
                    # q_embeddings = w[0]*retriever_model.encode(loc, convert_to_numpy=True) + w[1]*retriever_model.encode(iloc, convert_to_numpy=True)+ w[2]*retriever_model.encode(perloc, convert_to_numpy=True)
                    if args.use_base == "yes":
                        base = [entry['question'] for entry in QA]
                        q_embeddings = q_embeddings+ retriever_model.encode(base, convert_to_numpy=True)
                elif args.eval_SAT_key == "no":
                    if args.use_base == "yes":
                        base = [entry['question'] for entry in QA]
                        q_embeddings = retriever_model.encode(base, convert_to_numpy=True)

                else:
                    q_embeddings = retriever_model.encode(questions, convert_to_numpy=True)
                if args.add_user_mean == "yes":
                    if args.eval_SAT_key == "no" and args.use_base == "no":
                        user_embd_mean_all = user_embd_mean[np.newaxis,: ] # 新增一维
                        user_embd_mean_all = np.tile(user_embd_mean_all, (len(questions), 1)) # 复制k次
                        q_embeddings = user_embd_mean_all
                    else:
                        q_embeddings = q_embeddings + user_embd_mean

                D, I = index.search(q_embeddings, args.num_chunks)
            retrival_res = np.array(chunks)[I]  # (# questions, # retrieve)
            logging.info("Retrieve finished")
            for i, q_dict in enumerate(QA):
                prediction = ''
                results_all_community[-1]["Results"].append({"q_id": q_dict["q_id"],
                                                                "question": q_dict["question"],
                                                                "answer": q_dict["answer"],
                                                                "prediction":prediction,
                                                                "type": q_dict["type"],
                                                                "difficulty": q_dict["difficulty"],
                                                                "retrieved_segment_ids": list(segment_ids[I[i]]),
                                                                "ground_truth_segment_ids": qa_gt_context_all_lookup[q_dict["q_id"]]["segment_id"],
                                                                "rewrite_query": q_dict['rewrite_query']
                                                                }
                                                                )
    save_json(results_all_community, os.path.join(save_path, f"result_noise_{noise}.json"))
    eval(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_shared_arguments(parser)
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to save eval results.")
    parser.add_argument("--save_dir", type=str, default="eval_data_v1")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunks for retrieval tasks")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--retrieval_model", type=str)
    parser.add_argument("--test_community_ids", type=str, default="community_0,community_1", help="Comma-separated list of community ids to test")
    parser.add_argument('--num_people', type=int, default=3)
    parser.add_argument('--noise', type=str)
    parser.add_argument('--eval_SAT_key', type=str)
    parser.add_argument('--use_base', type=str)
    parser.add_argument('--add_user_mean', type=str)
    parser.add_argument("--meta_train", type=str, default="no")
    args = parser.parse_args()
    setup_logger(args.verbose)
    set_random_seed(args.seed)
    eval_rag(args)