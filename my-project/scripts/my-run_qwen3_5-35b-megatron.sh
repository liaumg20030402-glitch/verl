#!/usr/bin/env bash
source /home3/medcog/jycai6/.bashrc
conda activate verl_rl

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -xeuo pipefail

# ===== Reward mode 配置 =====
# REWARD_MODE: rule | disrm | genrm
REWARD_MODE=${REWARD_MODE:-rule}
# reward manager 
REWARD_MANAGER_NAME=${REWARD_MANAGER_NAME:-dapo}


# 自定义奖励函数路径（rule/genrm 模式会用到）
REWARD_FN_PATH=${REWARD_FN_PATH:-"/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_blzk_rule.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score_blzk_rule}

# 奖励模型路径：genrm/disrm 模式会启用 reward_model
GRM_MODEL_PATH=${GRM_MODEL_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen2.5-7B-Instruct"}
DISRM_MODEL_PATH=${DISRM_MODEL_PATH:-"${GRM_MODEL_PATH}"}
export GRM_MODEL_NAME="${GRM_MODEL_NAME:-${GRM_MODEL_PATH}}"

# reward model rollout 资源
GEN_RM_TP=${GEN_RM_TP:-1}
GRM_GPU_MEM=${GRM_GPU_MEM:-0.35}

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# 编译缓存：全部放到节点本地 /tmp，避免多机共享 NFS 上 16 个 worker 并发 JIT
# 编译时的 flock/半写文件竞争（flashinfer gdn_prefill_sm90 在 NFS 上极易 ninja 失败）。
UNIQUE_ID=${UNIQUE_ID:-$(date +%Y%m%d_%H%M%S)}
tmp_run_dir="/tmp/jmli27_verl_run_${UNIQUE_ID}"
export TRITON_CACHE_DIR="${tmp_run_dir}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${tmp_run_dir}/inductor_cache"
export VLLM_CONFIG_ROOT="${tmp_run_dir}/vllm_config"
export FLASHINFER_WORKSPACE_BASE="${tmp_run_dir}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${tmp_run_dir}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${tmp_run_dir}/xdg_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CONFIG_ROOT}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}"

# 让 head清理本节点 NFS HOME 下可能残留的旧 cache
if [ "${RANK:-0}" == "0" ]; then
    rm -rf ~/.cache/vllm ~/.cache/torch/inductor ~/.triton/cache ~/.cache/flashinfer 2>/dev/null || true
fi

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

export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

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
        # 训练退出（不论成功失败）时，都 ray stop，避免 daemon 残留。
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
TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

rollout_name="vllm"
project_name='verl_grpo_qwen3_5_35b_blzk'
exp_name="verl_qwen3_5_35b_megatron_blzk_rule_${UNIQUE_ID}_multi"
adv_estimator=grpo

# ===== 本地模型路径 =====
HF_MODEL_PATH=${HF_MODEL_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-35B-A3B"}

# ===== 本地 parquet 数据路径 =====
train_path=${train_path:-"/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train_fast_verl.parquet"}
test_path=${test_path:-"/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val_fast_verl.parquet"}

BASE_OUT_DIR="/train21/medcog/permanent/jycai6/jmli27/"
LOG_FILE="${BASE_OUT_DIR}/log/${project_name}/grpo_verl_megatron_qwen35_a3b_${UNIQUE_ID}_multi.log"
CKPTS_DIR="${BASE_OUT_DIR}/output/${project_name}/${UNIQUE_ID}_multi"
mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${CKPTS_DIR}")"

############################# Parameter Arrays #############################

DATA=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=64
    # blzk有几百条大于4096
    data.max_prompt_length=2048
    data.max_response_length=1024
    data.truncation='error'
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=32
    data.return_raw_chat=True
    data.shuffle=True
    # 固定打乱顺序，可复现
    data.seed=42
    # 限制训练集验证集大小，加快测试速度（可根据实际情况调整，-1为全部）
    data.train_max_samples=-1
    data.val_max_samples=-1
    # +data.apply_chat_template_kwargs='{enable_thinking:False}'
)

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0

    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=3
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.max_model_len=8192
    # === 添加训练时的采样参数 ===
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
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)

# Reward 参数（支持 3 种模式：rule / disrm / genrm）
REWARD=(
    reward.num_workers=8
    reward.reward_manager.name=${REWARD_MANAGER_NAME}
)

if [[ "${REWARD_MODE}" == "rule" ]]; then
    # 规则函数：不启用 reward model，只走自定义函数
    REWARD+=(
        reward.reward_model.enable=False
        reward.custom_reward_function.path=${REWARD_FN_PATH}
        reward.custom_reward_function.name=${REWARD_FN_NAME}
    )
elif [[ "${REWARD_MODE}" == "disrm" ]]; then
    # 判别式 RM：启用 reward model，不设置 custom_reward_function（走内置 compute_score_disrm）
    REWARD+=(
        reward.reward_model.enable=True
        reward.reward_model.enable_resource_pool=False
        reward.reward_model.model_path=${DISRM_MODEL_PATH}
        reward.reward_model.rollout.name=${rollout_name}
        reward.reward_model.rollout.dtype=bfloat16
        reward.reward_model.rollout.tensor_model_parallel_size=${GEN_RM_TP}
        reward.reward_model.rollout.gpu_memory_utilization=${GRM_GPU_MEM}
        reward.reward_model.rollout.prompt_length=2048
        reward.reward_model.rollout.response_length=1024
        reward.reward_model.rollout.skip_tokenizer_init=False
    )
elif [[ "${REWARD_MODE}" == "genrm" ]]; then
    # 生成式 RM：启用 reward model + 自定义函数（函数内走 /v1/chat/completions）
    REWARD+=(
        reward.reward_model.enable=True
        reward.reward_model.enable_resource_pool=False
        reward.reward_model.model_path=${GRM_MODEL_PATH}
        reward.reward_model.rollout.name=${rollout_name}
        reward.reward_model.rollout.dtype=bfloat16
        reward.reward_model.rollout.tensor_model_parallel_size=${GEN_RM_TP}
        reward.reward_model.rollout.gpu_memory_utilization=${GRM_GPU_MEM}
        reward.reward_model.rollout.prompt_length=2048
        reward.reward_model.rollout.response_length=1024
        reward.reward_model.rollout.skip_tokenizer_init=False
        reward.custom_reward_function.path=${REWARD_FN_PATH}
        reward.custom_reward_function.name=${REWARD_FN_NAME}
        +reward.custom_reward_function.reward_kwargs.grm_temperature=0.0
        +reward.custom_reward_function.reward_kwargs.grm_top_p=1.0
        +reward.custom_reward_function.reward_kwargs.grm_max_tokens=512
    )
else
    echo "Invalid REWARD_MODE=${REWARD_MODE}, expected one of: rule|disrm|genrm"
    exit 1
fi

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.n_gpus_per_node=8
    trainer.nnodes=${NNODES}
    trainer.save_freq=300
    trainer.val_before_train=False
    trainer.test_freq=200
    trainer.total_epochs=1

    # Number of generations to log during validation
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