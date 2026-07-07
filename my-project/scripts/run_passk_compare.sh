#!/usr/bin/env bash
# 一键：顺序对 base / grpo / maxrl / dapo 跑无偏 Pass@K，最后画一张对比图。
#
# 为什么必须「顺序」而不是并行：每个模型都要独占整机 8 卡起 SGLang 引擎，
# 一次只能装一个模型；所以本脚本逐个 load→采样→打分→shutdown，全部跑完再统一画图。
# 断点续跑：某个模型已有 passk_summary.json 就自动跳过，可中断后重跑补齐。
#
# 用法：
#   bash run_passk_compare.sh                 # 用下面 MODELS 里写死的列表
#   N=32 TEMPERATURE=0.7 bash run_passk_compare.sh   # 临时改公共参数
#
# ⚠️ 公平对比的前提：所有模型共用同一组 N / TEMPERATURE / TOP_P / MAX_NEW_TOKENS /
#    同一验证集 / 同一 ENABLE_THINKING。改就一起改，别单独动某个模型。

set -xeuo pipefail

# ============================ 环境 ============================
source /home3/medcog/jycai6/.bashrc
module use /opt/tool/modulefiles/ || true
module load cuda/12.9 || true
conda activate sglang_infer

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 编译缓存放本地 /tmp（不要落到 $HOME/.triton 的 NFS）：
#   dp=8 时 8 个 DP worker 会并发编译同一 triton kernel；NFS 上并发读写同一缓存
#   会 race，报 FileNotFoundError: .../write_req_to_token_pool_triton.json。
#   本地 /tmp 无此问题（同 start_genrm_server_sglang.sh）。
cache_root="/tmp/jmli27_passk_cache_$(hostname -s)"
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
export MASTER_PORT=34567
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

# ============================ 公共评测参数（所有模型一致）============================
VAL_PATH=${VAL_PATH:-"/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val_fast_verl.parquet"}
N=${N:-64}                                   # 每题采样数（要 > 最大 k）
K_LIST=${K_LIST:-"1,2,4,8,16,32,64"}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
ENABLE_THINKING=${ENABLE_THINKING:-true}
MAX_PROMPTS=${MAX_PROMPTS:--1}               # 调试时设小（如 50）只跑前 N 题
TP=${TP:-1}                                  # 27B 单卡放得下：tp=1/dp=8 最快；放不下改 tp=8/dp=1
DP=${DP:-8}
MEM_FRACTION=${MEM_FRACTION:-0.9}
# GDN/SSM 内核后端：务必 triton。flashinfer 对 Qwen3.5 GDN 有 mbarrier 死锁
# （详见 notes/vllm-gdn-hang-and-triton-backend.md），等价于 server 端 --mamba-backend triton。
MAMBA_BACKEND=${MAMBA_BACKEND:-triton}
# verl 导出的 ckpt tokenizer 损坏（报 TokenizersBackend does not exist）。RL 不改 tokenizer，
# 所有模型统一借用 base 的 tokenizer（等价且可加载）。base 自己用它也是无害 no-op。
TOKENIZER_PATH=${TOKENIZER_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_blzk_rule.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score_blzk_rule}

OUT_ROOT=${OUT_ROOT:-"/train21/medcog/permanent/jycai6/jmli27/passk_eval/output/blzk_compare"}
mkdir -p "${OUT_ROOT}"

# ============================ 待测模型列表 ============================
# 每行 "标识|HF模型目录"。base 放第一个（对比基线）。
# 训练 ckpt 用 global_step_N/actor/huggingface（verl 自动导出的 HF 权重）。
MODELS=(
    "base|/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"
    "grpo_step137|/train21/medcog/permanent/jycai6/jmli27/output/verl_grpo_qwen3_5_27b_blzk/blzk_v3_rule_grpo/global_step_137/actor/huggingface"
    "maxrl_step137|/train21/medcog/permanent/jycai6/jmli27/output/verl_grpo_qwen3_5_27b_blzk/blzk_v3_rule_maxrl/global_step_137/actor/huggingface"
    "dapo_step137|/train21/medcog/permanent/jycai6/jmli27/output/verl_grpo_qwen3_5_27b_blzk/blzk_v3_rule_dapo/global_step_137/actor/huggingface"
)

# ============================ 逐个评测 ============================
for entry in "${MODELS[@]}"; do
    MID="${entry%%|*}"
    MPATH="${entry##*|}"
    OUT_DIR="${OUT_ROOT}/${MID}"

    if [ -f "${OUT_DIR}/passk_summary.json" ]; then
        echo "[skip] ${MID} 已有结果：${OUT_DIR}/passk_summary.json"
        continue
    fi
    if [ ! -e "${MPATH}" ]; then
        echo "[warn] ${MID} 模型路径不存在，跳过：${MPATH}"
        continue
    fi

    echo "==================== 评测 ${MID} ===================="
    python "${SCRIPT_DIR}/passk_sglang_blzk.py" \
        --model "${MPATH}" \
        --model_id "${MID}" \
        --val_path "${VAL_PATH}" \
        --n "${N}" --k_list "${K_LIST}" \
        --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" --enable_thinking "${ENABLE_THINKING}" \
        --max_prompts "${MAX_PROMPTS}" \
        --tp "${TP}" --dp "${DP}" --mem_fraction_static "${MEM_FRACTION}" \
        --mamba_backend "${MAMBA_BACKEND}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --reward_fn_path "${REWARD_FN_PATH}" --reward_fn_name "${REWARD_FN_NAME}" \
        --out_dir "${OUT_DIR}"
done

# ============================ 画对比图 ============================
# FONT_PATH：可选，指向一个 CJK 字体 .ttf/.otf，解决中文图例显示成方块。
FONT_PATH=${FONT_PATH:-""}
echo "==================== 画对比图 ===================="
python "${SCRIPT_DIR}/plot_passk.py" \
    --summaries "${OUT_ROOT}"/*/passk_summary.json \
    --out "${OUT_ROOT}/passk_compare.png" \
    ${FONT_PATH:+--font_path "${FONT_PATH}"}

echo "完成。结果目录：${OUT_ROOT}"
echo "  - 每模型：${OUT_ROOT}/<model_id>/passk_summary.json (+ generations.jsonl / per_prompt.jsonl)"
echo "  - 对比图：${OUT_ROOT}/passk_compare*.png ；对比表：${OUT_ROOT}/passk_compare*_*.csv"
