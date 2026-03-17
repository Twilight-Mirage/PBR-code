# PBR: Personalize-Before-Retrieve Framework for User-Centric Retrieval

This is repository provides the official implementation of the `AAAI 2026 Oral` paper:

**Personalize Before Retrieve: LLM-based Personalized Query Expansion for User-Centric Retrieval**  

![璁烘枃鍥剧墖](./assets/main_framework.png)

---

## 馃専 Overview

This project implements **PBR (Personalize Before Retrieve)** 鈥?a novel framework for personalized query expansion in Retrieval-Augmented Generation (RAG) systems. Unlike traditional expansion methods, PBR adapts to **individual user styles and corpus structures** before retrieval.

PBR consists of two core modules:

- **P-PRF**: simulates personalized query rewrites (utterances + reasoning) based on user history
- **P-Anchor**: builds a graph over user corpus and applies Personalized PageRank to ground queries structurally

We evaluate PBR on two personalized benchmarks: **PersonaBench** and **LongMemEval**, where it outperforms strong baselines such as HyDE, Query2Term, MILL, CoT, and ThinkQE across multiple retrievers.

---

## 馃 Key Features

- 馃攣 **LLM-based query expansion** with style-aware feedback and reasoning simulation
- 馃З **Graph-enhanced memory retrieval** via PageRank and embedding fusion
- 馃И Full evaluation pipeline with ablation, baselines, and parameter sensitivity
- 馃搳 Compatible with retrievers like `multi-qa-MiniLM`, `all-MiniLM`, `bge-base-en`

---

## 馃摝 Installation

Install required packages:

```bash
pip install -r requirements.txt
```

You鈥檒l need:
	鈥?sentence-transformers, faiss-cpu
	鈥?openai, cvxpy, scikit-learn, ot, numpy, scipy
	鈥?a valid OpenAI API Key (for gpt-4o-mini)


## 馃摎 Usage - LongMemEval
### 1. download data
you need to download LongMemEval data to this dictionary form https://github.com/xiaowu0162/LongMemEval. 

For example: "./data/longmemeval_data/longmemeval_s.json".

### 2. run the code
```bash
python -u ./src/retrieval/retrieval_PBR.py \
    --model_type="PBR" \
    --retrieval_model_name="multi-qa-MiniLM-L6-cos-v1" \
    --data_type='s'

```

## 馃摎 Usage - personabench
### 1. run the code
```bash
cd ./personabench_main_PBR
SEED=2024
MODEL_TYPE="PBR" # fake_ada_reason_fake_10
LOG_DIR="PBR_all-mpnet-base-v2_new"
DATA_DIR="eval_data/eval_data_v1"
SAVE_DIR="PBR" 
TEST_COMMUNITY_IDS="community_0,community_1"
NUM_CHUNKS=5
BASE_MODELS="gpt-4o-mini"
RETRIEVERS="multi-qa-MiniLM-L6-cos-v1"
TEST_NOISES="0.0"
VERBOSE="--verbose"



CUDA_VISIBLE_DEVICES="0" python scripts/evaluation/eval_rag_PBR.py \
    --seed $SEED \
    --log_dir $LOG_DIR \
    --model_type $MODEL_TYPE \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --test_community_ids $TEST_COMMUNITY_IDS \
    --num_chunks $NUM_CHUNKS \
    --model_name $BASE_MODELS \
    --retrieval_model $RETRIEVERS \
    --noise $TEST_NOISES \
    $VERBOSE

```

### 馃摉 Citation
If you find this work helpful, please cite:
```bibtex
@article{zhang2025personalize,
  title={Personalize Before Retrieve: LLM-based Personalized Query Expansion for User-Centric Retrieval},
  author={Zhang, Yingyi and Jia, Pengyue and Xu, Derong and Wen, Yi and Li, Xianneng and Wang, Yichao and Zhang, Wenlin and Li, Xiaopeng and Gan, Weinan and Guo, Huifeng and others},
  journal={arXiv preprint arXiv:2510.08935},
  year={2025}
}
```
## Temporal Profile Extension (PBR++)

The retrieval pipeline now supports a temporal-evolving user profile:

- time decay: exp(-lambda * age_days)
- usage-aware weighting: sigmoid(alpha * utility)
- temporal graph center: short-term and long-term anchor blending
- temporal rerank: rerank over FAISS candidates with temporal weights

Example:

```bash
python -u ./src/retrieval/retrieval_PBR.py \
    --model_type="PBR++" \
    --temporal_profile \
    --temporal_decay_lambda=0.05 \
    --temporal_util_alpha=4.0 \
    --temporal_util_bias=0.0 \
    --temporal_rerank_beta=0.4 \
    --temporal_graph_decay=0.02 \
    --short_term_blend=0.5 \
    --k_seed=10 \
    --top_k_retrieval=10 \
    --retrieval_model_name="multi-qa-MiniLM-L6-cos-v1" \
    --data_type='s' \
    --save_suffix="_exp_temporal"
```

## Fast Comparison Workflow

Use `experiments/` for reproducible ablations:

```bash
python experiments/run_retrieval_matrix.py --matrix experiments/configs/longmemeval_temporal_matrix.json
python experiments/summarize_retrieval_results.py --glob "data/longmemeval_data/*_PBR*.json"
```


