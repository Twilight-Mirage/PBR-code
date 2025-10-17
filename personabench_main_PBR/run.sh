SEED=2024
MODEL_TYPE="PBR" # fake_ada_reason_fake_10
LOG_DIR="PBR_all-mpnet-base-v2_new"
DATA_DIR="eval_data/eval_data_v1"
SAVE_DIR="PBR" 
TEST_COMMUNITY_IDS="community_0,community_1"
NUM_CHUNKS=5
BASE_MODELS="gpt-4o-mini"
RETRIEVERS="multi-qa-MiniLM-L6-cos-v1" # "all-mpnet-base-v2,gt-context,all-MiniLM-L6-v2,BAAI/bge-m3", "multi-qa-MiniLM-L6-cos-v1"
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