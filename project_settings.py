"""
Project-local runtime settings.

Edit this file directly to manage defaults for this repository.
Priority in code paths is generally:
1) explicit CLI arg
2) environment variable
3) values from this file
"""

# OpenAI-compatible settings (also used by DashScope compatible-mode)
OPENAI_API_KEY = "sk-e5ae65daa99b44009c008e6a7087177e"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_ORGANIZATION = ""

# DashScope aliases (optional; runtime falls back between OPENAI_* and DASHSCOPE_*)
DASHSCOPE_API_KEY = "sk-e5ae65daa99b44009c008e6a7087177e"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Provider-specific generation knobs
LLM_ENABLE_THINKING = False

# Retrieval embedding defaults
RETRIEVAL_MODEL_NAME = r"D:\研三\毕设\temporal-rag\embedding_model"
EMBEDDING_TASK = "auto"

# Default LongMemEval inputs used by experiment scripts and retrieval fallback.
LONGMEMEVAL_INPUT_BY_SPLIT = {
    "s": "./data/longmemeval_data/longmemeval_s_cleaned.json",
    "oracle": "./data/longmemeval_data/longmemeval_oracle.json",
}
