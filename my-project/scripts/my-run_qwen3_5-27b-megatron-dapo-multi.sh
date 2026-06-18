#!/usr/bin/env bash
# Qwen3.5-27B (Dense) GRPO RL with Megatron
#
# Qwen3.5 architecture notes（和 35B-A3B 共用）:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention which currently does
#   NOT support packed sequences (THD format) in Megatron-LM. Therefore:
#     - model.use_remove_padding=False           (forces bshd compute format)
#     - actor.megatron.use_remove_padding=False  (forces bshd compute format)
#     - actor.use_dynamic_bsz=False              (required for bshd mode)
#
# Dense vs 35B-A3B 差异:
#   - Qwen3.5-27B 是 dense 模型，**没有 expert**，所以不需要 EP / ETP
#   - dense 单 token 激活显存比 MoE-A3B 大（用所有参数 vs 只用激活参数），TP 通常调大一些
#   - 默认 TP=4（35B-A3B 是 TP=2）
#   - 配置数组里去掉所有 MoE 相关 override（aux_loss / z_loss / vanilla_mbridge）

source /home3/medcog/jycai6/.bashrc
conda activate verl_rl_v2

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -xeuo pipefail

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# 加大 vLLM V1 EngineCore 等 TP worker 的超时（对应 multiproc_executor → shm_broadcast）
# 兜底首次冷编译，避免 ptxas 编译时间超过默认 60s 被判死亡
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600



# 编译缓存：按节点持久化到本地 /tmp，跨多次运行共享（参考 vllm#36631 的最终结论）
# - 不带 UNIQUE_ID，可共用同一份缓存
# - 按 hostname 隔离：多机时每节点写自己 /tmp，避免 NFS 上 JIT 并发竞争

cache_root="/tmp/jmli27_verl_cache_$(hostname -s)"
export TRITON_CACHE_DIR="${cache_root}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor_cache"
export VLLM_CONFIG_ROOT="${cache_root}/vllm_config"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${cache_root}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${cache_root}/xdg_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CONFIG_ROOT}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}"

# 不再 rm -rf ~/.cache：缓存 env 已经指向本地 /tmp，HOME 下不会写新内容；
# 而且持久化缓存的全部意义就是跨运行复用，清掉会让下次启动重新 ptxas 5–10 分钟。

export NNODES=$WORLD_SIZE
export NODE_RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1

# # single node
# export RANK=0
# export NODE_RANK=$RANK
# export WORLD_SIZE=1
# export NNODES=$WORLD_SIZE
# export MASTER_PORT=34567
# export MASTER_ADDR=localhost

# export GLOO_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_IFNAME=eth0

export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
# 关闭 NVLink SHARP（in-network reduction），改走经典 ring/tree。
# 多机 H200 + Megatron→vLLM 权重广播 + bf16 大 allgather 已知会触发 NVLS code path hang
# （参考 NVIDIA/nccl#2167, NVIDIA/nccl#2077, NVIDIA-NeMo/RL#1961）。代价 ~5% 通信吞吐。
export NCCL_NVLS_ENABLE=0



export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# === NCCL flight recorder + debug ===
export TORCH_FR_BUFFER_SIZE=20000  
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET,ENV
export NCCL_DEBUG_FILE=/tmp/jmli27_nccl_${HOSTNAME}_%h_%p.log

############################# Ray 集群初始化（多机）#############################
# 单机（NNODES=1）：跳过这整段，main_ppo.py 里的 ray.init() 会自己起一个本地集群。
# 多机：调度器会在每个节点上同时运行本脚本。RANK=0 作为 head，RANK>0 作为 worker。
RAY_PORT=${RAY_PORT:-$((${MASTER_PORT:-6379} + 1))}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
EXPECTED_GPUS=$(( NNODES * 8 ))

# 清理本节点上次运行残留的 ray daemon，避免新 cluster 连到旧的 GCS。
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray /tmp/ray_tmp_* 2>/dev/null || true

if [ "${NNODES}" -gt 1 ]; then
    # 把 MASTER_ADDR 解析成具体的 IPv4，让 head 绑定的 IP 和 worker 连接的 IP 完全一致。
    HEAD_IP=$(getent hosts "${MASTER_ADDR}" | awk '{print $1}' | head -n1)
    [ -z "${HEAD_IP}" ] && HEAD_IP="${MASTER_ADDR}"
    echo "[Ray] RANK=${RANK} NNODES=${NNODES} MASTER_ADDR=${MASTER_ADDR} HEAD_IP=${HEAD_IP} PORT=${RAY_PORT}"

    if [ "${RANK}" == "0" ]; then
        echo "[Ray] 在 ${HEAD_IP}:${RAY_PORT} 上启动 head 节点 ..."
        ray start --head \
            --node-ip-address="${HEAD_IP}" \
            --port="${RAY_PORT}" \
            --dashboard-host=0.0.0.0 \
            --dashboard-port="${RAY_DASHBOARD_PORT}" \
            --num-gpus=8 \
            --disable-usage-stats

        # 等到「节点数」和「GPU 总数」都对齐才放行训练。
        echo "[Ray] 等待 ${NNODES} 个节点 + ${EXPECTED_GPUS} 张 GPU 就绪 ..."
        WAITED=0
        # 用 __RAYSTATS__ 作为哨兵标记，过滤掉 ray.init 自己混进 stdout 的 INFO 日志
        while true; do
            STATS=$(python3 - <<'PYEOF' 2>/dev/null | grep '^__RAYSTATS__ ' | tail -n1
import ray
ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")
alive = sum(1 for n in ray.nodes() if n["Alive"])
gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"__RAYSTATS__ {alive} {gpus}", flush=True)
PYEOF
)
            read -r _TAG ALIVE GPUS <<<"${STATS}"
            ALIVE=${ALIVE:-0}; GPUS=${GPUS:-0}
            echo "[Ray] nodes=${ALIVE}/${NNODES} gpus=${GPUS}/${EXPECTED_GPUS} (已等待 ${WAITED}s)"
            if [ "${ALIVE}" = "${NNODES}" ] && [ "${GPUS}" = "${EXPECTED_GPUS}" ]; then
                break
            fi
            if [ "${WAITED}" -ge 600 ]; then
                echo "[Ray] 等待集群就绪超时，打印当前 ray status 供排查："
                ray status || true
                exit 1
            fi
            sleep 5
            WAITED=$(( WAITED + 5 ))
        done
        echo "[Ray] 集群就绪，开始训练。"
        ray status || true
        trap 'echo "[Ray] 停止集群..."; ray stop --force' EXIT
    else
        # Worker：用重试循环 join
        echo "[Ray] Worker ${RANK}：尝试加入 ${HEAD_IP}:${RAY_PORT} ..."
        for i in $(seq 1 60); do
            if ray start --address="${HEAD_IP}:${RAY_PORT}" \
                         --num-gpus=8 \
                         --disable-usage-stats; then
                echo "[Ray] Worker ${RANK} 在第 ${i} 次尝试后成功加入。"
                break
            fi
            echo "[Ray] Worker ${RANK} 第 ${i} 次加入失败，10s 后重试 ..."
            sleep 10
            if [ "${i}" = "60" ]; then
                echo "[Ray] Worker ${RANK} 加入集群失败，退出。"
                exit 1
            fi
        done
        echo "[Ray] Worker ${RANK} staying alive ..."
        while ray status >/dev/null 2>&1; do
            sleep 30
        done
        echo "[Ray] 集群已停止，Worker ${RANK} 退出。"
        exit 0
    fi
fi

############################# Quick Config #############################
# Qwen3.5-27B 是 dense 模型，无 EP/ETP
# 27B dense 单 token 激活显存比 35B-A3B 大（用全部参数 vs 只用激活参数），
# 所以 TP 默认设大一点（TP=4），PP/CP 保持 1。
TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
GEN_TP=${GEN_TP:-8}    # vLLM rollout TP，独立于训练侧 TP

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

# ===== Reward mode 配置 =====
# REWARD_MODE: default | rule | disrm | genrm | genrm_remote | multi
#   default: 不设 custom_reward_function，让 verl 按 parquet 里 data_source 字段
#            自动路由到内置评分器（gsm8k / MATH / MATH-500 / codecontests / ...）。
#            适用于跑 verl examples/data_preprocess/ 里官方预处理过的数据集。
#   rule:    显式指定 custom_reward_function，覆盖自动路由（用于自定义规则评分，
#            比如本项目的医疗题）
#   disrm:   discriminative reward model（一个打分/分类模型当裁判）
#   genrm:   generative reward model（verl 托管：内部起 GenRM server + router）
#   genrm_remote: 外部 server 版 genrm（自己用 start_genrm_server.sh 起服务，
#                 verl 只调用 post 到 GENRM_BASE_URL 的奖励函数；最适合单独调试 GenRM）
#   multi:   ⭐ 多任务联合训练（本脚本默认）。一个分发奖励函数按每行 data_source 路由：
#            med-exam → 外部 GenRM（同 genrm_remote）；blzk / kie / zyzl-blzk → 本地规则。
#            train/val 用合并后的多任务 parquet（见 process_data/merge_parquet.py），
#            每行自带 data_source，verl 逐样本回调时把 data_source 传给分发函数。
REWARD_MODE=${REWARD_MODE:-multi}
REWARD_MANAGER_NAME=${REWARD_MANAGER_NAME:-dapo}

# 自定义奖励函数路径（rule/genrm/genrm_remote/multi 都用 REWARD_FN_PATH 这一个变量引用）
# multi 模式默认指向分发函数；单任务模式按需覆盖 REWARD_FN_PATH / REWARD_FN_NAME。
REWARD_FN_PATH=${REWARD_FN_PATH:-"/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_multitask.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score_multitask}


# 奖励模型路径：genrm/disrm 模式会启用 reward_model
GRM_MODEL_PATH=${GRM_MODEL_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}
DISRM_MODEL_PATH=${DISRM_MODEL_PATH:-"${GRM_MODEL_PATH}"}
export GRM_MODEL_NAME="${GRM_MODEL_NAME:-${GRM_MODEL_PATH}}"

# ===== 节点 / 资源池分配（policy 训练 vs reward model）=====
# 物理总节点 NNODES(=WORLD_SIZE) 由调度器给定。
#   - RM_NNODES=0   → reward model 与 policy colocate（共用同一批卡）；TRAIN_NNODES 自动 = NNODES
#   - RM_NNODES>0   → reward model 用独立资源池(standalone)；TRAIN_NNODES 自动 = NNODES - RM_NNODES
# 例（48 卡=6 节点）：RM_NNODES=2 → 自动 TRAIN_NNODES=4（policy 32 卡 + GenRM 16 卡）。
# GenRM 与 rollout 异步重叠只有standalone支持
if [[ "${REWARD_MODE}" == "genrm" || "${REWARD_MODE}" == "disrm" ]]; then
    # 仅托管 RM(genrm/disrm) 需要这几个——节点数 + RM 的 vLLM 资源放一起
    RM_NNODES=${RM_NNODES:-1}                            # ← 托管模式通常只改这一个
    GEN_RM_TP=${GEN_RM_TP:-4}                            # RM 的 vLLM TP（27B 建议 ≥4）
    GRM_GPU_MEM=${GRM_GPU_MEM:-0.85}                     # RM 独享卡(standalone)可拉高；colocate 要调低
else
    RM_NNODES=${RM_NNODES:-0}                            # rule/default/genrm_remote：不占集群内 RM 卡
fi
TRAIN_NNODES=${TRAIN_NNODES:-$(( NNODES - RM_NNODES ))}  # 默认=剩下的节点

# RM_NNODES>0 自动启用独立资源池，并做一次过/欠订阅检查
if [ "${RM_NNODES}" -gt 0 ]; then
    RM_ENABLE_POOL=True
    if [ $(( TRAIN_NNODES + RM_NNODES )) -ne "${NNODES}" ]; then
        echo "[Config] ⚠️ TRAIN_NNODES(${TRAIN_NNODES}) + RM_NNODES(${RM_NNODES}) != NNODES(${NNODES})，standalone 资源池可能调度不出来"
    fi
else
    RM_ENABLE_POOL=False
fi



# === 实验标识与 resume 策略 ===
# exp_name: 实验标识，**跨多次运行必须稳定**，让 resume_mode=auto 能找到上次的 ckpt。
#   - 默认按 reward 模式区分（rule / disrm / genrm 三套实验目录互不串扰）
#   - 要"开新实验从头跑"，改exp_name，新 exp_name 没 ckpt 就自动从头开始
#   - 同一 exp_name 的多次启动会 resume 同一个 CKPTS_DIR 下最新 ckpt
# RUN_ID: 每次启动一个新的时间戳，**只用于隔离 log 文件**，不影响 ckpt 路径
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

rollout_name="vllm"
project_name='verl_grpo_qwen3_5_27b_multitask'
# ⚠️ 从 SFT ckpt 起训：换了新 exp_name，确保 auto resume 不会误续到基座那次的 ckpt。
#    （resume_mode=auto 是按 CKPTS_DIR=output/${project}/${exp} 找最新 ckpt 的）
exp_name="multitask_sftinit_v1_${REWARD_MODE}"
adv_estimator=grpo

# ===== 初始 policy 模型路径 =====
# RL 的 actor 初始权重 / rollout / reference 都来自这个路径。
# 从 SFT ckpt 起训：指向 swift 全参 SFT 导出的 HF 格式 ckpt（含 config.json + *.safetensors + tokenizer*）。
# ⚠️ 路径请按实际核对（下面是按截图转写，注意 OCR 可能有偏差）。
# 基座模型（如需对比从头训）：/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B
SFT_CKPT_PATH="/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen_27b_sft_med_1043_20260601/model_output/qwen_27b_sft_med_1043_20260529_epoch/v0-20260603-084935/checkpoint-1118-resave"
HF_MODEL_PATH=${HF_MODEL_PATH:-"${SFT_CKPT_PATH}"}

# ===== 本地 parquet 数据路径（多任务：直接传文件列表，不合并）=====
# verl 的 data.train_files / val_files 支持传文件列表，自动拼接后由 data.shuffle=True 全局打乱。
# 每个 parquet 每行自带 data_source，reward_fn_multitask 据此分发打分。
# ⭐ 想去掉某类任务：把对应行注释掉即可，无需重新合并数据。
DATASET_ROOT=${DATASET_ROOT:-"/train21/medcog/permanent/jycai6/jmli27/dataset"}
TRAIN_FILES=(
    "${DATASET_ROOT}/medexam/medexam_train_verl.parquet"
    "${DATASET_ROOT}/blzk/blzk_train_verl.parquet"
    "${DATASET_ROOT}/kie/kie_train_verl.parquet"
    "${DATASET_ROOT}/zyzl_blzk/zyzl_blzk_train_verl.parquet"
)
VAL_FILES=(
    "${DATASET_ROOT}/medexam/medexam_val_verl.parquet"
    "${DATASET_ROOT}/blzk/blzk_val_verl.parquet"
    "${DATASET_ROOT}/kie/kie_val_verl.parquet"
    "${DATASET_ROOT}/zyzl_blzk/zyzl_blzk_val_verl.parquet"
)

# 把 bash 数组拼成 Hydra list 字面量： ['a','b',...]（无空格、单引号包路径）
join_hydra_list() {
    local sep="" out="[" f
    for f in "$@"; do
        out+="${sep}'${f}'"
        sep=","
    done
    printf '%s]' "${out}"
}

# 允许用环境变量 train_path/test_path 直接覆盖（单文件或自定义 Hydra list 都行）；
# 否则由上面的 TRAIN_FILES / VAL_FILES 数组生成 list。
train_path=${train_path:-$(join_hydra_list "${TRAIN_FILES[@]}")}
test_path=${test_path:-$(join_hydra_list "${VAL_FILES[@]}")}


BASE_OUT_DIR="/train21/medcog/permanent/jycai6/jmli27/"
CKPTS_DIR="${BASE_OUT_DIR}/output/${project_name}/${exp_name}"
# log 按 RUN_ID 分文件落到 CKPTS_DIR/logs/，每次崩溃 / 续跑都留独立日志便于对照
LOG_FILE="${CKPTS_DIR}/logs/run_${RUN_ID}.log"
mkdir -p "${CKPTS_DIR}" "${CKPTS_DIR}/logs"

############################# Parameter Arrays #############################

DATA=(
    # 加引号：train_path 可能是 ['a','b',...] 这种含 [ ] 的 Hydra list，避免被 bash 当 glob 展开
    "data.train_files=${train_path}"
    "data.val_files=${test_path}"
    # ⚠️ batch 整除性提示：
    # minimal_bsz = (Train_NNODES*8 / TP/PP/CP) * micro_per_gpu
    # NNODES=2 TP=4 → DP=4, minimal_bsz=4    train_batch × rollout.n % minimal_bsz == 0
    # NNODES=4 TP=4 → DP=8, minimal_bsz=8    
    # NNODES=6 TP=4 → DP=12, minimal_bsz=12  
    data.train_batch_size=480
    data.max_prompt_length=8192
    data.max_response_length=16384
    data.truncation='error'
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=32
    data.return_raw_chat=True
    data.shuffle=True
    data.seed=42
    data.train_max_samples=-1
    data.val_max_samples=-1
    # +data.apply_chat_template_kwargs='{enable_thinking:False}'
)

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    # Qwen3.5 GDN 不支持 THD，必须 False
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR=(
    # actor_rollout_ref.nccl_timeout=1800
    actor_rollout_ref.actor.optim.lr=1e-6
    # verl 里 train_batch_size 和 ppo_mini_batch_size 的单位都是 prompt 数
    # 必须 train_batch % mini_batch == 0
    # mini_batch × rollout.n % minimal_bsz == 0
    actor_rollout_ref.actor.ppo_mini_batch_size=120
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    # use_dynamic_bsz=False 时这行无效（GDN 必须 False）
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=False
    # DAPO recipe: no KL loss / no KL reward, use Clip-Higher and token-level loss.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean

    actor_rollout_ref.actor.megatron.use_mbridge=True
    # 27B 是 dense，不需要 vanilla_mbridge（vanilla_mbridge 主要给 MoE 用）
    # actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    # dense 模型不需要 EP/ETP，从配置里移除
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    # dense 模型移除 MoE 专属 loss 系数:
    #   moe_aux_loss_coeff 和 moe_z_loss_coeff 是 MoE router 专用的辅助 loss，dense 没有 router
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.dtype=bfloat16
    # ⭐ GDN prefill backend: flashinfer (默认) → triton
    #   flashinfer 的 chunk_gated_delta_rule kernel 在 CUDA Graph replay 下有 mbarrier
    #   死锁 bug，导致 TP rank 间 desync → NCCL allgather watchdog 触发挂死。
    #   参考: flashinfer-ai/flashinfer#3329, verl-project/verl#5659 评论 9, vllm#41862
    #   代价: 单 GDN kernel 慢 1.7-1.9×，整体 rollout 慢 5-10%
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
    #   权重同步bucket 传输
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    # actor_rollout_ref.rollout.max_model_len=24576
    # === 训练时的采样参数 ===
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    +actor_rollout_ref.rollout.repetition_penalty=1.0

    # === 单独控制验证 (val) 阶段的采样参数 ===
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    # False uses greedy sampling
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    # dense 模型不需要 EP/ETP
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
    # Dynamic Sampling: filter groups whose reward metric is all the same.
    +algorithm.filter_groups.enable=True
    +algorithm.filter_groups.metric=score
    +algorithm.filter_groups.max_num_gen_batches=10
)

# Reward 参数（支持 6 种模式：default / rule / disrm / genrm / genrm_remote / multi）
REWARD=(
    reward.num_workers=8
    reward.reward_manager.name=${REWARD_MANAGER_NAME}
    # Overlong reward shaping: max response is 16k; the last 8k tokens are the linear penalty buffer.
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=8192
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=16384
)

if [[ "${REWARD_MODE}" == "default" ]]; then
    # 走 verl 内置 default_compute_score：按 parquet 里每行的 data_source 字段
    # 路由到对应内置评分器（见 verl/utils/reward_score/__init__.py）。
    # ⚠️ 故意不设 reward.custom_reward_function —— 设了会覆盖自动路由。
    # 用法：parquet 必须由 verl 官方预处理脚本生成（带 data_source / ground_truth 字段）。
    REWARD+=(
        reward.reward_model.enable=False
    )
elif [[ "${REWARD_MODE}" == "rule" ]]; then
    REWARD+=(
        reward.reward_model.enable=False
        reward.custom_reward_function.path=${REWARD_FN_PATH}
        reward.custom_reward_function.name=${REWARD_FN_NAME}
    )
elif [[ "${REWARD_MODE}" == "disrm" ]]; then
    REWARD+=(
        reward.reward_model.enable=True
        reward.reward_model.enable_resource_pool=${RM_ENABLE_POOL}
        reward.reward_model.nnodes=${RM_NNODES}
        reward.reward_model.n_gpus_per_node=8
        reward.reward_model.model_path=${DISRM_MODEL_PATH}
        reward.reward_model.rollout.name=${rollout_name}
        reward.reward_model.rollout.dtype=bfloat16
        reward.reward_model.rollout.tensor_model_parallel_size=${GEN_RM_TP}
        reward.reward_model.rollout.gpu_memory_utilization=${GRM_GPU_MEM}
        reward.reward_model.rollout.prompt_length=2048
        reward.reward_model.rollout.response_length=1024
        reward.reward_model.rollout.skip_tokenizer_init=False
        # Qwen3.5 GDN: 同 actor，强制 vLLM prefill 走 triton，规避 flashinfer mbarrier 崩溃
        +reward.reward_model.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
    )
elif [[ "${REWARD_MODE}" == "genrm_remote" ]]; then
    # ===== 外部 server 模式 =====
    # GenRM 用 scripts/start_genrm_server.sh 起成独立 sglang 服务，
    # verl 不管它的卡/生命周期，只调用一个会 post 到 GENRM_BASE_URL 的奖励函数。
    # 跑前确保：① start_genrm_server.sh 已起好；② `bash start_genrm_server.sh --test` curl 通；
    #           ③ 下面两个 env 指向它（GENRM_MODEL_NAME 要 = server 的 --served-model-name）。
    export GENRM_BASE_URL=${GENRM_BASE_URL:-"http://100.85.97.73:8000"}
    export GENRM_MODEL_NAME=${GENRM_MODEL_NAME:-"genrm_remote"}
    REWARD+=(
        # 不让 verl 在集群内起 RM（GenRM 在集群外），policy 用全部节点
        reward.reward_model.enable=False
        reward.custom_reward_function.path=${REWARD_FN_PATH}
        reward.custom_reward_function.name=${REWARD_FN_NAME}
        # ⭐ 地址/模型名走 reward_kwargs 下发（多机可达，不依赖环境变量能否传到 ray worker）
        +reward.custom_reward_function.reward_kwargs.grm_base_url=${GENRM_BASE_URL}
        +reward.custom_reward_function.reward_kwargs.grm_model_name=${GENRM_MODEL_NAME}
        # 复读主要受温度影响（实测 temp 0.6→0.7→0.8 复读递减，top_k 无明显差异），不加任何惩罚。
        +reward.custom_reward_function.reward_kwargs.grm_temperature=0.8
        +reward.custom_reward_function.reward_kwargs.grm_top_p=1.0
        +reward.custom_reward_function.reward_kwargs.grm_top_k=-1
        # 开思考的复杂审核任务输出较长，max_tokens 给大些（8192）避免 JSON 被截断。
        +reward.custom_reward_function.reward_kwargs.grm_max_tokens=8192
        # 打分与 rollout 重叠（reward loop），判分耗时大多被生成时间盖住；timeout 仍设上限防卡死。
        +reward.custom_reward_function.reward_kwargs.grm_request_timeout=600
        +reward.custom_reward_function.reward_kwargs.grm_max_retries=1
        # 裁判思考开关（默认开，见脚本上方 GRM_ENABLE_THINKING）；JSON 解析失败由奖励函数内
        # json_repair 修复，修不了则 genrm_failed=1 软丢弃。
        +reward.custom_reward_function.reward_kwargs.grm_enable_thinking=True
    )
elif [[ "${REWARD_MODE}" == "genrm" ]]; then
    REWARD+=(
        reward.reward_model.enable=True
        reward.reward_model.enable_resource_pool=${RM_ENABLE_POOL}
        reward.reward_model.nnodes=${RM_NNODES}
        reward.reward_model.n_gpus_per_node=8
        reward.reward_model.model_path=${GRM_MODEL_PATH}
        reward.reward_model.rollout.name=${rollout_name}
        reward.reward_model.rollout.dtype=bfloat16
        reward.reward_model.rollout.tensor_model_parallel_size=${GEN_RM_TP}
        reward.reward_model.rollout.gpu_memory_utilization=${GRM_GPU_MEM}
        reward.reward_model.rollout.prompt_length=24576
        reward.reward_model.rollout.response_length=1024
        # reward.reward_model.rollout.max_model_len=26624
        reward.reward_model.rollout.skip_tokenizer_init=False
        # Qwen3.5 GDN: 同 actor，强制 vLLM prefill 走 triton，规避 flashinfer mbarrier 崩溃
        +reward.reward_model.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
        reward.custom_reward_function.path=${REWARD_FN_PATH}
        reward.custom_reward_function.name=${REWARD_FN_NAME}
        +reward.custom_reward_function.reward_kwargs.grm_temperature=0.6
        +reward.custom_reward_function.reward_kwargs.grm_top_p=0.95
        +reward.custom_reward_function.reward_kwargs.grm_max_tokens=8192
    )
elif [[ "${REWARD_MODE}" == "multi" ]]; then
    # ===== 多任务联合训练（分发奖励函数）=====
    # 一个 custom_reward_function（reward_fn_multitask.compute_score_multitask）按每行
    # data_source 路由：med-exam → 外部 GenRM（同 genrm_remote，走下面的 grm_* 配置）；
    # blzk / kie / zyzl-blzk → 本地规则评分（忽略 grm_* 配置）。
    # 因此这里同时下发 genrm_remote 的全部 reward_kwargs：med-exam 用得到，规则任务用不到。
    # 跑前提：med-exam 的外部 GenRM 已用 start_genrm_server.sh 起好并 curl 验证过。
    export GENRM_BASE_URL=${GENRM_BASE_URL:-"http://100.85.97.73:8000"}
    export GENRM_MODEL_NAME=${GENRM_MODEL_NAME:-"genrm_remote"}
    REWARD+=(
        # 不在集群内起 RM（med-exam 的 GenRM 在集群外），policy 用全部节点
        reward.reward_model.enable=False
        reward.custom_reward_function.path=${REWARD_FN_PATH}
        reward.custom_reward_function.name=${REWARD_FN_NAME}
        # med-exam 分支用：GenRM 地址 / 模型名 / 采样参数（经 reward_kwargs 下发，多机可达）
        +reward.custom_reward_function.reward_kwargs.grm_base_url=${GENRM_BASE_URL}
        +reward.custom_reward_function.reward_kwargs.grm_model_name=${GENRM_MODEL_NAME}
        # 复读主要受温度影响（实测 temp 0.6→0.7→0.8 复读递减，top_k 无明显差异），不加任何惩罚。
        +reward.custom_reward_function.reward_kwargs.grm_temperature=0.8
        +reward.custom_reward_function.reward_kwargs.grm_top_p=1.0
        +reward.custom_reward_function.reward_kwargs.grm_top_k=-1
        +reward.custom_reward_function.reward_kwargs.grm_max_tokens=8192
        +reward.custom_reward_function.reward_kwargs.grm_request_timeout=600
        +reward.custom_reward_function.reward_kwargs.grm_max_retries=1
        # 裁判思考开关（默认开，见脚本上方 GRM_ENABLE_THINKING）；JSON 解析失败由奖励函数内
        # json_repair 修复，修不了则 genrm_failed=1 软丢弃。
        +reward.custom_reward_function.reward_kwargs.grm_enable_thinking=True
    )
else
    echo "Invalid REWARD_MODE=${REWARD_MODE}, expected one of: default|rule|disrm|genrm|genrm_remote|multi"
    exit 1
fi

# ===== GenRM 调试落盘（排查裁判超长/JSON 解析失败）=====
# 设了 GRM_DEBUG_DIR 才落盘：解析失败的样本会把裁判完整 raw 回复 + actor 答案 + 标准答案 +
# token 数追加到 ${GRM_DEBUG_DIR}/genrm_debug_<host>_<pid>.jsonl，方便排查为什么超 8192。
# 经 reward_kwargs 下发（多机可达，不依赖环境变量能否传到 ray reward worker）。排查完取消即可。
GRM_DEBUG_DIR=${GRM_DEBUG_DIR:-"${CKPTS_DIR}/genrm_debug"}
if [ -n "${GRM_DEBUG_DIR}" ] && [[ "${REWARD_MODE}" == "multi" || "${REWARD_MODE}" == "genrm_remote" || "${REWARD_MODE}" == "genrm" ]]; then
    mkdir -p "${GRM_DEBUG_DIR}"
    REWARD+=( +reward.custom_reward_function.reward_kwargs.grm_debug_dir=${GRM_DEBUG_DIR} )
fi

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.n_gpus_per_node=8
    trainer.nnodes=${TRAIN_NNODES}
    # Resume mode: "auto", "disable", or "resume_path"
    # "auto": resume from last checkpoint if available  ← 已开启
    # "disable": start from scratch（要彻底从头训：改 expname 或临时把这行换成 disable）
    # "resume_path": resume from a user-defined path
    trainer.resume_mode=auto
    # trainer.max_actor_ckpt_to_keep=2    # 只留最近 2 个 actor ckpt
    trainer.save_freq=50
    trainer.val_before_train=True
    trainer.test_freq=10
    trainer.total_epochs=1

    trainer.log_val_generations=10
    +trainer.validation_data_dir=${CKPTS_DIR}/validation_data
    +trainer.rollout_data_dir=${CKPTS_DIR}/rollout_data
)

# main_ppo.py 里的 ray.init() 连接上面已经起好的多机集群
EXTRA_RAY_ARGS=()
if [ "${NNODES}" -gt 1 ]; then
    EXTRA_RAY_ARGS+=(+ray_kwargs.ray_init.address=auto)
fi

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    hydra.run.dir=${CKPTS_DIR}/hydra_logs \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA_RAY_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
