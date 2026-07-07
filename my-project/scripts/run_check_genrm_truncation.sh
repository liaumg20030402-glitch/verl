#!/usr/bin/env bash
# 提交脚本：单节点跑 check_genrm_truncation_sglang.py，
# 用离线 SGLang 复现 med-exam 的 GenRM 裁判截断率/解析失败率，判断超长是不是 server 调用方式导致。
#
# 用法（集群单节点）：
#   bash run_check_genrm_truncation.sh                          # 基线（temp 0.8，无其它手段）
#   MAX_SAMPLES=200 bash run_check_genrm_truncation.sh          # 先小样本快速对比
#   VAL_DATA_DIR=/path/to/rollout_data bash run_check_genrm_truncation.sh
#
# 防复读对比实验（每组改 TAG，结果落到不同 OUT 目录，互不覆盖）：
#   TAG=temp0.6      TEMPERATURE=0.6                       bash run_check_genrm_truncation.sh
#   TAG=temp1.0      TEMPERATURE=1.0                       bash run_check_genrm_truncation.sh
#   TAG=reppen1.1    REPETITION_PENALTY=1.1                bash run_check_genrm_truncation.sh
#   TAG=prespen1.5   PRESENCE_PENALTY=1.5                  bash run_check_genrm_truncation.sh
#   TAG=freqpen0.5   FREQUENCY_PENALTY=0.5                 bash run_check_genrm_truncation.sh
#   TAG=minp0.05     MIN_P=0.05                            bash run_check_genrm_truncation.sh
#   TAG=nothink      ENABLE_THINKING=false                 bash run_check_genrm_truncation.sh   # 关思考对照
#
# 看每组 OUT/summary.json 里的：truncation_rate / heavy_repeat_rate(rep10>0.5) / rep10_rate_mean /
#   json_parse_fail_rate / completion_tokens.mean，横向对比哪种手段最有效。
# summary.json 自带本次 sampling 配置，方便汇成对比表。
#
# 解读（基线）：
#   - 离线截断率也很高 → 不是 server 的锅，是模型/采样/prompt（已确认）。
#   - 离线明显低、线上高 → 问题在 server 调用路径（多半 chat_template_kwargs 没生效）。

set -xeuo pipefail

# ============================ 环境（同 run_passk_compare.sh）============================
source /home3/medcog/jycai6/.bashrc
module use /opt/tool/modulefiles/ || true
module load cuda/12.9 || true
conda activate "${CONDA_ENV:-sglang_infer}"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# 编译缓存放本地 /tmp（dp=8 并发编译 triton kernel，NFS 上会 race）
cache_root="/tmp/jmli27_genrmcheck_cache_$(hostname -s)"
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
export MASTER_PORT=34571
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
# GenRM 模型：必须和线上 server 同一个（否则比的不是同一个东西）
MODEL=${MODEL:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}

# 输入：verl 落盘的 validation_data 目录（每个 step 一个 parquet），自动收集全部 *.parquet。
#   也可换成 rollout_data 目录，或用 INPUT_FILES 直接给文件列表。
VAL_DATA_DIR=${VAL_DATA_DIR:-"/train21/medcog/permanent/jycai6/jmli27/output/verl_grpo_qwen3_5_27b_multitask/multitask_sftinit_v1_multi/validation_data"}
shopt -s nullglob
INPUT_FILES=( ${INPUT_FILES:-${VAL_DATA_DIR}/0.jsonl} )
shopt -u nullglob
if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "[ERR] 没找到输入文件：${VAL_DATA_DIR}/*.parquet（用 VAL_DATA_DIR 或 INPUT_FILES 指定）"; exit 1
fi

DATA_SOURCE=${DATA_SOURCE:-"med-exam"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_medexam_genrm_remote.py"}

# 采样：默认与训练脚本 multi 分支一致（temp 0.8 / top_p 1.0 / top_k -1 / max_tokens 8192 / 开思考）
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
ENABLE_THINKING=${ENABLE_THINKING:-true}
MAX_SAMPLES=${MAX_SAMPLES:--1}          # 先设 50 冒烟确认字段抽对，再 -1 全量
BATCH_SIZE=${BATCH_SIZE:-256}

# SGLang 引擎：27B 单卡放得下 → tp=1/dp=8 吞吐最高
TP=${TP:-1}
DP=${DP:-8}
MEM_FRACTION=${MEM_FRACTION:-0.9}
MAMBA_BACKEND=${MAMBA_BACKEND:-triton}  # Qwen3.5 GDN 必须 triton

# ===== 防复读/防崩 可选手段（默认中性=不启用；逐个开做对比实验）=====
# 复读的本质：模型把"已经说过的 token"又挑成最高概率。下面几个参数从不同角度打压它。
#
# MIN_P：min-p 采样。只保留"概率 ≥ min_p × 最大token概率"的候选，其余截掉再采样。
#   作用：砍掉长尾低概率 token，让采样更聚焦但仍有随机性。0=关；常用 0.05~0.1。
#   对复读的帮助：间接——配合温度让分布更健康，单独用效果一般。
MIN_P=${MIN_P:-0.0}
# REPETITION_PENALTY：token 级重复惩罚。对"之前出现过的 token"的 logit 除以一个系数，
#   压低它再次被选中的概率。1.0=关；>1 生效（1.05 温和 / 1.1 较强 / >1.2 易伤正常表达）。
#   特点：只看"出没出现过"，不看次数；对单 token 重复有效，对"整句循环"偏弱。
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
# PRESENCE_PENALTY：存在惩罚（OpenAI 风格）。某 token "只要出现过一次"就给固定扣分。
#   0=关；>0 生效（1.0 / 1.5）。Qwen3 官方推荐用它抗重复（0~2）。
#   对"整句刷屏"比 repetition_penalty 更有效，且对合法重复的 JSON 结构符（引号/括号）更友好。
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
# FREQUENCY_PENALTY：频次惩罚（OpenAI 风格）。扣分**随出现次数线性累加**——出现越多扣越狠。
#   0=关；>0 生效（0.5 / 1.0）。对"同一句话刷几十遍"的死循环最克制；但太大会伤 JSON 里高频的标点。
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-0.0}

OUT=${OUT:-"/train21/medcog/permanent/jycai6/jmli27/genrm_trunc_check/${DATA_SOURCE}_${TEMPERATURE}_${MAX_NEW_TOKENS}"}
mkdir -p "${OUT}"

echo "[Check] 模型=${MODEL}"
echo "[Check] 输入文件数=${#INPUT_FILES[@]}  data_source=${DATA_SOURCE}  out=${OUT}"

python "${SCRIPT_DIR}/check_genrm_truncation_sglang.py" \
    --model "${MODEL}" \
    --input "${INPUT_FILES[@]}" \
    --data-source "${DATA_SOURCE}" \
    --reward-fn-path "${REWARD_FN_PATH}" \
    --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" --enable-thinking "${ENABLE_THINKING}" \
    --min-p "${MIN_P}" --repetition-penalty "${REPETITION_PENALTY}" \
    --presence-penalty "${PRESENCE_PENALTY}" --frequency-penalty "${FREQUENCY_PENALTY}" \
    --max-samples "${MAX_SAMPLES}" --batch-size "${BATCH_SIZE}" \
    --tp "${TP}" --dp "${DP}" --mem-fraction "${MEM_FRACTION}" --mamba-backend "${MAMBA_BACKEND}" \
    --out "${OUT}"

echo "完成。结果：${OUT}/summary.json （每条裁判输出：${OUT}/judge_outputs.jsonl）"
