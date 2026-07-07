#!/usr/bin/env bash
# 提交脚本：用 HTTP 打你**已起好的 SGLang server**，测 GenRM 判分的截断率/复读率，
# 并验证 server 是否支持各惩罚参数（这条路径 == 实际 RL 训练）。
#
# 纯 HTTP 客户端，不占 GPU，在任何能访问 BASE_URL 的机器上跑即可。
#
# 用法：
#   BASE_URL=http://100.85.97.73:8000 bash run_check_genrm_truncation_server.sh           # 基线（=训练请求）
#   # 惩罚参数对比（看 summary 的 error_rate / first_error 判断 server 支不支持该键）：
#   TAG=prespen1.5 PRESENCE_PENALTY=1.5 bash run_check_genrm_truncation_server.sh
#   TAG=reppen1.1  REPETITION_PENALTY=1.1 bash run_check_genrm_truncation_server.sh
#   MAX_SAMPLES=200 bash run_check_genrm_truncation_server.sh                              # 先小样本

set -xeuo pipefail

source /home3/medcog/jycai6/.bashrc
conda activate "${CONDA_ENV:-sglang_infer}"   # 只需有 aiohttp/pandas/numpy/json_repair 的环境即可

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ============================ 必填：server 地址 ============================
BASE_URL=${BASE_URL:-"http://100.85.97.73:8000"}   # = 训练侧 GENRM_BASE_URL
MODEL_NAME=${MODEL_NAME:-"genrm_remote"}            # = server 的 --served-model-name / GENRM_MODEL_NAME

# ============================ 数据 ============================
VAL_DATA_DIR=${VAL_DATA_DIR:-"/train21/medcog/permanent/jycai6/jmli27/output/verl_grpo_qwen3_5_27b_multitask/multitask_sftinit_v1_multi/validation_data"}
shopt -s nullglob
INPUT_FILES=( ${INPUT_FILES:-${VAL_DATA_DIR}/*.parquet} )
shopt -u nullglob
if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "[ERR] 没找到输入文件：${VAL_DATA_DIR}/*.parquet（用 VAL_DATA_DIR 或 INPUT_FILES 指定）"; exit 1
fi

DATA_SOURCE=${DATA_SOURCE:-"med-exam"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_medexam_genrm_remote.py"}

# ============================ 采样（默认对齐训练 multi 分支）============================
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
ENABLE_THINKING=${ENABLE_THINKING:-true}
MAX_SAMPLES=${MAX_SAMPLES:--1}

# ============================ 防复读/防崩 可选项（默认中性=不发；server 不支持的键会 400 暴露）====
MIN_P=${MIN_P:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-0.0}

# ============================ HTTP 并发 ============================
CONCURRENCY=${CONCURRENCY:-64}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
MAX_RETRIES=${MAX_RETRIES:-2}

TAG=${TAG:-"T${TEMPERATURE}"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-"/train21/medcog/permanent/jycai6/jmli27/genrm_trunc_check_server/${DATA_SOURCE}_${TAG}_${RUN_ID}"}
mkdir -p "${OUT}"

echo "[Check-server] BASE_URL=${BASE_URL} model=${MODEL_NAME}  data_source=${DATA_SOURCE}  out=${OUT}"

python "${SCRIPT_DIR}/check_genrm_truncation_server.py" \
    --base-url "${BASE_URL}" --model-name "${MODEL_NAME}" \
    --input "${INPUT_FILES[@]}" \
    --data-source "${DATA_SOURCE}" \
    --reward-fn-path "${REWARD_FN_PATH}" \
    --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" --enable-thinking "${ENABLE_THINKING}" \
    --min-p "${MIN_P}" --repetition-penalty "${REPETITION_PENALTY}" \
    --presence-penalty "${PRESENCE_PENALTY}" --frequency-penalty "${FREQUENCY_PENALTY}" \
    --concurrency "${CONCURRENCY}" --request-timeout "${REQUEST_TIMEOUT}" --max-retries "${MAX_RETRIES}" \
    --max-samples "${MAX_SAMPLES}" \
    --out "${OUT}"

echo "完成。结果：${OUT}/summary.json （每条裁判输出：${OUT}/judge_outputs.jsonl）"
