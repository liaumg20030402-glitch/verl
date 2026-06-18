#!/usr/bin/env bash
# 离线 Pass@K 评测脚本（blzk 规则任务）。
#
# 用途：加载一个已保存的 checkpoint，对验证集**每题采样 K 次**，让 verl 算出
#   best@K/mean(=Pass@K)、mean@K(≈Pass@1) 等指标。用来对照 GRPO vs MaxRL 的 ckpt。
#   —— 训练时 val_kwargs.n=1 不动（和基线可比、省算力），Pass@K 只在这里离线测。
#
# 原理：复用训练同一套 main_ppo + 同样的并行/模型配置（保证 ckpt 能正确加载），但：
#   - trainer.val_only=True            → 只做一次验证就退出，不训练
#   - trainer.resume_from_path=<ckpt>  → 加载指定 global_step_N
#   - val_kwargs.n=K + do_sample=True + temperature>0  → 采样 K 次，才能算 Pass@K
#
# 用法：
#   bash eval_passk_blzk.sh <CKPT_PATH> [K] [TEMPERATURE]
#   例：bash eval_passk_blzk.sh /train21/.../blzk_v3_rule/global_step_100 16 0.6
#   也可用环境变量：CKPT_PATH=... VAL_N=16 VAL_TEMPERATURE=0.6 bash eval_passk_blzk.sh
#
# ⚠️ 公平对比：对 GRPO ckpt 和 MaxRL ckpt 用**完全相同**的 K/temperature/top_p/同一验证集/
#    同一并行配置；并选可比的 step。结果看日志里的 `val-core/.../score/best@<K>/mean`。

set -xeuo pipefail

# ============================ 参数 ============================
CKPT_PATH="${CKPT_PATH:}"
VAL_N="${VAL_N:-32}"                       # K：每题采样次数（2 的幂最顺，如 16/32）
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"  # 必须 >0，否则贪心 K 条全一样、Pass@K==Pass@1
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_TOP_K="${VAL_TOP_K:--1}"

if [ -z "${CKPT_PATH}" ]; then
    echo "用法: bash eval_passk_blzk.sh <global_step_N 路径> [K] [temperature]"; exit 1
fi
if [[ "${CKPT_PATH}" != *"global_step_"* ]]; then
    echo "❌ CKPT_PATH 必须指向含 global_step_<N> 的目录，例如 .../blzk_v3_rule/global_step_100"; exit 1
fi

# ============================ 环境（与训练脚本一致）============================
source /home3/medcog/jycai6/.bashrc
conda activate verl_rl_v2

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

cache_root="/tmp/jmli27_verl_cache_$(hostname -s)"
export TRITON_CACHE_DIR="${cache_root}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor_cache"
export VLLM_CONFIG_ROOT="${cache_root}/vllm_config"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${cache_root}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${cache_root}/xdg_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CONFIG_ROOT}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}"

export NNODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-34567}
export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NVLS_ENABLE=0

############################# Ray 集群初始化（多机；单机自动跳过）#############################
RAY_PORT=${RAY_PORT:-$((${MASTER_PORT:-6379} + 1))}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
EXPECTED_GPUS=$(( NNODES * 8 ))
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray /tmp/ray_tmp_* 2>/dev/null || true

if [ "${NNODES}" -gt 1 ]; then
    HEAD_IP=$(getent hosts "${MASTER_ADDR}" | awk '{print $1}' | head -n1)
    [ -z "${HEAD_IP}" ] && HEAD_IP="${MASTER_ADDR}"
    if [ "${NODE_RANK}" == "0" ]; then
        ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
            --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus=8 --disable-usage-stats
        WAITED=0
        while true; do
            GPUS=$(python3 - <<'PYEOF' 2>/dev/null | grep '^__RAYGPU__ ' | tail -n1 | awk '{print $2}'
import ray
ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")
print(f"__RAYGPU__ {int(ray.cluster_resources().get('GPU', 0))}", flush=True)
PYEOF
)
            GPUS=${GPUS:-0}
            echo "[Ray] gpus=${GPUS}/${EXPECTED_GPUS} (${WAITED}s)"
            [ "${GPUS}" = "${EXPECTED_GPUS}" ] && break
            [ "${WAITED}" -ge 600 ] && { echo "[Ray] 等待超时"; ray status || true; exit 1; }
            sleep 5; WAITED=$(( WAITED + 5 ))
        done
        trap 'echo "[Ray] 停止集群..."; ray stop --force' EXIT
    else
        for i in $(seq 1 60); do
            ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=8 --disable-usage-stats && break
            echo "[Ray] worker ${NODE_RANK} 第 ${i} 次加入失败，10s 重试"; sleep 10
        done
        echo "[Ray] worker ${NODE_RANK} 保持存活直至集群停止 ..."
        while ray status >/dev/null 2>&1; do sleep 30; done
        exit 0
    fi
fi

############################# 配置（并行/模型必须与训练 ckpt 一致）#############################
# ⚠️ TP/PP/CP 必须与产出该 ckpt 的训练一致，否则 megatron ckpt 加载会对不上。
TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
GEN_TP=${GEN_TP:-8}                 # vLLM 生成用的 TP
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

HF_MODEL_PATH=${HF_MODEL_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}
# blzk 验证集（每题采样 K 次就在这上面）
test_path=${test_path:-"/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val_fast_verl.parquet"}
# val_only 不训练，但 data.train_files 必填——指向验证集即可，避免加载大训练集
train_path=${train_path:-"${test_path}"}

# blzk 规则奖励（0/1）。best@K 就建立在它返回的 score 上 → best@K = Pass@K。
REWARD_FN_PATH=${REWARD_FN_PATH:-"/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_blzk_rule.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score_blzk_rule}

# 日志输出目录（按 ckpt 名 + K 命名，便于区分 GRPO/MaxRL/各 step）
STEP_TAG="$(basename "${CKPT_PATH}")"            # global_step_N
EXP_TAG="$(basename "$(dirname "${CKPT_PATH}")")" # 实验名（如 blzk_v3_rule / maxrl...）
OUT_DIR=${OUT_DIR:-"/train21/medcog/permanent/jycai6/jmli27/passk_eval/${EXP_TAG}_${STEP_TAG}_K${VAL_N}_T${VAL_TEMPERATURE}"}
mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"
LOG_FILE="${OUT_DIR}/logs/passk_$(date +%Y%m%d_%H%M%S).log"

echo "[Pass@K] ckpt=${CKPT_PATH}  K=${VAL_N}  T=${VAL_TEMPERATURE}  out=${OUT_DIR}"

EXTRA_RAY_ARGS=()
[ "${NNODES}" -gt 1 ] && EXTRA_RAY_ARGS+=(+ray_kwargs.ray_init.address=auto)

############################# 启动 val_only 评测 #############################
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    hydra.run.dir="${OUT_DIR}/hydra_logs" \
    data.train_files="${train_path}" \
    data.val_files="${test_path}" \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.train_batch_size=480 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.return_raw_chat=True \
    data.shuffle=False \
    data.val_max_samples=-1 \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=120 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD} \
    algorithm.adv_estimator=grpo \
    reward.num_workers=8 \
    reward.reward_manager.name=dapo \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name="${REWARD_FN_NAME}" \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=passk_eval \
    trainer.experiment_name="${EXP_TAG}_${STEP_TAG}_K${VAL_N}" \
    trainer.default_local_dir="${OUT_DIR}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${CKPT_PATH}" \
    +trainer.validation_data_dir="${OUT_DIR}/validation_data" \
    "${EXTRA_RAY_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "[Pass@K] 完成。看日志里 val-core/<data_source>/score/best@${VAL_N}/mean (=Pass@${VAL_N})、mean@${VAL_N}(≈Pass@1)。"
echo "[Pass@K] 日志: ${LOG_FILE}"
