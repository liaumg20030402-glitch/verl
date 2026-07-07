#!/usr/bin/env bash
# 提交脚本：单节点离线 SGLang 跑 check_genrm_consistency_sglang.py，
# 测 blsc 病历审查裁判在不同温度下的**判分一致率 / 解码重复率**。
#
# 直接用 blsc 原始 messages 数据里的 (user=医患对话, assistant=参考病历) 当被审样本，
# 无需先跑 actor rollout。引擎只加载一次，内部依次扫描全部温度档（不用手动改参数）。
#
# 用法（集群单节点）：
#   bash run_check_genrm_consistency.sh                                  # 默认温度 1.0/0.9/0.8/0.7/0.6，N=5
#   MAX_SAMPLES=200 bash run_check_genrm_consistency.sh                  # 先小样本冒烟
#   TEMPERATURES="1.0 0.8 0.6" N_RUNS=3 bash run_check_genrm_consistency.sh
#   INPUT_FILES=/path/to/blsc_val.jsonl bash run_check_genrm_consistency.sh
#
# 看 OUT/summary.json 的 per_temperature：label_exact_match_rate / score_exact_match_rate /
#   mean_pairwise_jaccard 越高越稳；rep10_rate_mean / truncation_rate 越低越好。

set -xeuo pipefail

# ============================ 环境（同 run_check_genrm_truncation.sh）============================
source /home3/medcog/jycai6/.bashrc
module use /opt/tool/modulefiles/ || true
module load cuda/12.9 || true
conda activate "${CONDA_ENV:-sglang_infer}"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# 编译缓存放本地 /tmp（dp=8 并发编译 triton kernel，NFS 上会 race）
cache_root="/tmp/jmli27_genrmconsist_cache_$(hostname -s)"
export TRITON_CACHE_DIR="${cache_root}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor_cache"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${cache_root}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${cache_root}/xdg_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}"

# single node
export RANK=0
export NODE_RANK=$RANK
export WORLD_SIZE=1
export NNODES=$WORLD_SIZE
export MASTER_PORT=34572
export MASTER_ADDR=localhost

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ============================ 参数 ============================
# GenRM 模型：必须和线上 server 同一个
MODEL=${MODEL:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}

# 输入：blsc 原始 messages 数据（jsonl/parquet）。用 INPUT_FILES 直接给文件，或 VAL_DATA_DIR 收集 *.jsonl。
VAL_DATA_DIR=${VAL_DATA_DIR:-"/train21/medcog/permanent/jycai6/jmli27/dataset/blsc"}
shopt -s nullglob
INPUT_FILES=( ${INPUT_FILES:-${VAL_DATA_DIR}/blsc_val.jsonl} )
shopt -u nullglob
if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "[ERR] 没找到输入文件：${VAL_DATA_DIR}/blsc_val.jsonl（用 VAL_DATA_DIR 或 INPUT_FILES 指定）"; exit 1
fi

REWARD_FN_PATH=${REWARD_FN_PATH:-"/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_blsc_genrm_remote.py"}

# 一致性实验主参数
TEMPERATURES=${TEMPERATURES:-"1.0 0.9 0.8 0.7 0.6"}   # 引擎只加载一次、内部依次扫描
N_RUNS=${N_RUNS:-5}                                   # 同参数重复推理次数

# 固定采样（默认 = 实验4 约定：top_p=0.95 top_k=20 min_p=0 presence=1.5 rep=1.0）
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-0.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
ENABLE_THINKING=${ENABLE_THINKING:-true}
MAX_SAMPLES=${MAX_SAMPLES:--1}
BATCH_SIZE=${BATCH_SIZE:-256}

# SGLang 引擎：27B 单卡放得下 → tp=1/dp=8 吞吐最高
TP=${TP:-1}
DP=${DP:-8}
MEM_FRACTION=${MEM_FRACTION:-0.9}
MAMBA_BACKEND=${MAMBA_BACKEND:-triton}
OUT=${OUT:-"/train21/medcog/permanent/jycai6/jmli27/genrm_check/blsc_consistency_N${N_RUNS}"}
mkdir -p "${OUT}"

echo "[Consistency] 模型=${MODEL}"
echo "[Consistency] 输入文件数=${#INPUT_FILES[@]}  N=${N_RUNS}  温度=${TEMPERATURES}  out=${OUT}"

# 温度列表展开成多个 --temperatures 参数
TEMPS=( ${TEMPERATURES} )

python "${SCRIPT_DIR}/check_genrm_consistency_sglang.py" \
    --model "${MODEL}" \
    --input "${INPUT_FILES[@]}" \
    --reward-fn-path "${REWARD_FN_PATH}" \
    --temperatures "${TEMPS[@]}" --n-runs "${N_RUNS}" \
    --top-p "${TOP_P}" --top-k "${TOP_K}" --min-p "${MIN_P}" \
    --presence-penalty "${PRESENCE_PENALTY}" --repetition-penalty "${REPETITION_PENALTY}" \
    --frequency-penalty "${FREQUENCY_PENALTY}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" --enable-thinking "${ENABLE_THINKING}" \
    --max-samples "${MAX_SAMPLES}" --batch-size "${BATCH_SIZE}" \
    --tp "${TP}" --dp "${DP}" --mem-fraction "${MEM_FRACTION}" --mamba-backend "${MAMBA_BACKEND}" \
    --out "${OUT}"

echo "完成。汇总：${OUT}/summary.json （每个温度/第几次跑的明细：${OUT}/T<温度>_run<次>/outputs.jsonl）"
