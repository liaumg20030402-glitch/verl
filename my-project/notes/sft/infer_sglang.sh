#!/bin/bash
# SFT 推理脚本 - 使用 sglang 原生后端
# 离线批量推理模式
source /home3/medcog/jycai6/.bashrc

module use /opt/tool/modulefiles/
module load cuda/12.9

tmp_cache=/tmp/jycai6_swift_cache_$(hostname)_${RANK:-0}
export MODELSCOPE_CACHE=$tmp_cache/modelscope
export HF_HOME=$tmp_cache/huggingface
export TRITON_CACHE_DIR=$tmp_cache/triton_cache

# Conda 环境配置
CONDA_ENV_NAME="sglang_infer"
conda activate ${CONDA_ENV_NAME}

# 获取 conda 环境路径
# CONDA_ENV_PATH=$(conda env list | grep "^${CONDA_ENV_NAME} " | awk '{print $2}')
# export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:$LD_LIBRARY_PATH


PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# single node
export RANK=0
export NODE_RANK=$RANK
export WORLD_SIZE=1
export NNODES=$WORLD_SIZE
export MASTER_PORT=6223
export MASTER_ADDR=localhost

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 可配置参数
# MODEL_PATH 读取 ckpt
# MODEL_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1042_20260330/model_output/qwen_27b_sft_med_1042_32k_20260330_epoch/v0-20260407-202322/checkpoint-882-resave"
# MODEL_ID="qwen_27b_sft_med_1042_32k_20260330_epoch1"

# MODEL_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/pretrained_ckpts/Qwen3.5-27B"
# MODEL_ID="qwen_27b_baseline"

# MODEL_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1042_20260330/model_output/qwen_27b_sft_med_1042_32k_20260413_epoch/v0-20260413-101232/checkpoint-882-resave"
# MODEL_ID="qwen_27b_sft_med_1042_32k_20260413_epoch1"

# MODEL_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1042_20260330/model_output/qwen_27b_sft_med_1042_32k_20260413_epoch/v0-20260413-101232/checkpoint-1764-resave"
# MODEL_ID="qwen_27b_sft_med_1042_32k_20260413_epoch2"

# MODEL_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1042_20260330/model_output/qwen_27b_sft_med_1042_32k_20260413_epoch_resume/v0-20260416-170313/checkpoint-2646-resave"
# MODEL_ID="qwen_27b_sft_med_1042_32k_20260413_epoch3"


# MODEL_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1042_20260330/model_output/qwen_27b_sft_med_1042_32k_20260424_epoch/v0-20260425-235304/checkpoint-882-resave"
# MODEL_ID="qwen_27b_sft_med_1042_32k_20260424_epoch1"

MODEL_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1043_20260501/model_output/qwen_27b_sft_med_1043_32k_20260502_epoch_v2/v0-20260507-215556/checkpoint-224-resave"
MODEL_ID="qwen_27b_sft_med_1043_32k_20260502_v2_epoch1"


# 输出配置
OUTPUT_DIR="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1043_20260501/test_output"

# 解析命令行参数
THINKING_MODE="all"   # 默认运行两种模式："all", "fast" 或 "slow"
while [[ $# -gt 0 ]]; do
    case $1 in
        --thinking-mode)
            THINKING_MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

TP_SIZE=1
DP_SIZE=8
DTYPE="bfloat16"

# 推理参数
MAX_NEW_TOKENS=32768
TEMPERATURE=0.7
TOP_P=1.0
TOP_K=-1
BATCH_SIZE=128

# 显存利用 mem_fraction
# 控制 GPU 显存中用于静态分配（模型权重 + KV cache 内存池）的比例。

# 计算公式：
#     mem_fraction_static = (模型权重 + KV cache 池) / GPU 总显存

# 示例（以 80GB A100 为例）:
#     0.9 → 72GB 用于模型权重 + KV cache, 剩余  8GB 用于激活值/CUDA graph 等
#     0.7 → 56GB 用于模型权重 + KV cache, 剩余 24GB

# 调参建议：
#     - 值越大：KV cache 越多，可同时处理更多请求，吞吐量更高，但更容易 OOM
#     - 值越小：显存余量更大，更不容易 OOM，但并发能力受限
#     - 推理时遇到 OOM，尝试调小至 0.7~0.8
#     - 可以从 0.9 开始，以 0.01 递减，找到当前负载下的最大可用值
MEM_FRACTION=0.9

# 数据集路径（支持多个输入文件）
INPUT_FILES=(
    # "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/医考公开集600题/cmnlu.jsonl"
    # "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/医考公开集600题/medqa.jsonl"
    # "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/医考公开集600题/mlec.jsonl"
    # "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/医考公开集1100题/医考公开集1100题.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/医考等级集610题/医考等级集610题.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/病历质控-无锡集/病历质控-无锡集.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/IFEval/ifeval.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/KIE-科研分析/省立线上数据开发集.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/病历生成/病历生成精简测试_顺序2.jsonl"
    "/train21/medcog/permanent/jycai6/med_sft_train_swift/data/test_data/病历生成/门诊新验收集130.jsonl"
)

# 执行推理
# thinking_mode: fast（无思考），slow（带思考），或 all（两种都运行）
SCRIPT_DIR=$(dirname "$0")

python ${SCRIPT_DIR}/infer_sglang.py \
    --model ${MODEL_PATH} \
    --input "${INPUT_FILES[@]}" \
    --output-dir ${OUTPUT_DIR} \
    --model-id ${MODEL_ID} \
    --thinking-mode ${THINKING_MODE} \
    --tp ${TP_SIZE} \
    --dp ${DP_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    --mem-fraction ${MEM_FRACTION}