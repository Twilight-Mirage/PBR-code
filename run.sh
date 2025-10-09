#!/bin/bash
#SBATCH --job-name=my_job             # 作业名称-请修改为自己任务名字 
#SBATCH --output=/home/xiaopli2/test/output_%j.txt        # 标准输出文件名 (%j 表示作业ID)-请修改为自己路径
#SBATCH --error=/home/xiaopli2/test/error_%j.txt          # 标准错误文件名-请修改为自己路径
#SBATCH --cpus-per-task=4             # 每个任务使用的CPU核心数
#SBATCH --mem=100G                      # 申请100GB内存
#SBATCH --time=12:00:00               # 运行时间限制，格式为hh:mm:ss
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="3"
python -u ./src/retrieval/retrieval_PBR.py \
    --model_type="PBR" \
    --retrieval_model_name="multi-qa-MiniLM-L6-cos-v1" \
    --data_type='s'
