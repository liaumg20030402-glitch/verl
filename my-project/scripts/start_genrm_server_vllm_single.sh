#!/usr/bin/env bash
# 起 GenRM(裁判) vLLM 服务——**初版：单实例 TP=8**（用来验证"那次成功保活"不是偶然）。
#   - 单实例：TP=8，不加 --data-parallel-size（避开 vllm#22361 hybrid TP+DP 的坑）
#   - CUDA graph 开（不加 --enforce-eager），吞吐更高
#   - GDN prefill backend = triton（vllm#41862，flashinfer 默认 backend 有 mbarrier bug）
#   - 保活：每 15s 发一条 128-token 请求
# 流程：后台起 server → 等就绪 → 自测 → 前台保活。Ctrl+C 退出。
#       `bash start_genrm_server_first.sh --test` 对已运行的 server 再测一次。

set -xeuo pipefail

source /home3/medcog/jycai6/.bashrc
conda activate verl_rl_v2

export VLLM_USE_V1=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# 修复 vllm serve 启动时 diskcache→sqlite3→ICU 触发的 libstdc++ CXXABI 缺失（强制用 conda 的新 libstdc++）
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# 编译缓存指到本地 /tmp（别用 NFS 上的 ~/.triton/cache，多 worker JIT 竞争会损坏）
cache_root="/tmp/jmli27_genrm_cache_$(hostname -s)"
export TRITON_CACHE_DIR="${cache_root}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor_cache"
export VLLM_CONFIG_ROOT="${cache_root}/vllm_config"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${cache_root}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${cache_root}/xdg_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CONFIG_ROOT}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}"

# ============================ 可调参数 ============================
GENRM_MODEL_PATH=${GENRM_MODEL_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}
SERVED_NAME=${SERVED_NAME:-"genrm_remote"}            # 必须与训练侧 GENRM_MODEL_NAME 一致
GENRM_TP=${GENRM_TP:-8}                                # 单实例 TP=8（用满 8 卡，不加 DP，8 卡同步 all-reduce，gpu利用率大）
GENRM_GPUS=${GENRM_GPUS:-"0,1,2,3,4,5,6,7"}
GENRM_PORT=${GENRM_PORT:-8000}
GENRM_HOST=${GENRM_HOST:-0.0.0.0}                      # 0.0.0.0 才能被别的节点访问
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
GPU_MEM=${GPU_MEM:-0.90}
WAIT_READY=${WAIT_READY:-1800}
KEEPALIVE_INTERVAL=${KEEPALIVE_INTERVAL:-15}           # 每 15s 保活
KEEPALIVE_TOKENS=${KEEPALIVE_TOKENS:-128}              # 每条 128 token

# ============================ 自测模式 ============================
if [[ "${1:-}" == "--test" ]]; then
    echo "[Test] /v1/models ..."
    curl -sS "http://127.0.0.1:${GENRM_PORT}/v1/models" | head -c 500; echo
    echo "[Test] 一条 chat ..."
    curl -sS "http://127.0.0.1:${GENRM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"请只回复两个字：正常\"}],\"max_tokens\":16,\"temperature\":0}"
    echo
    exit 0
fi

# ============================ 启动 server（单实例 TP=8）============================
echo "[GenRM] 后台启动：model=${GENRM_MODEL_PATH} served=${SERVED_NAME} TP=${GENRM_TP} gpus=${GENRM_GPUS} port=${GENRM_PORT}"
CUDA_VISIBLE_DEVICES="${GENRM_GPUS}" vllm serve "${GENRM_MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${GENRM_TP}" \
    --host "${GENRM_HOST}" \
    --port "${GENRM_PORT}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --trust-remote-code \
    --gdn-prefill-backend triton &
SERVER_PID=$!
trap 'echo "[GenRM] 停止 server (pid=${SERVER_PID}) ..."; kill ${SERVER_PID} 2>/dev/null || true' EXIT INT TERM

# ============================ 等就绪 ============================
HEALTH_URL="http://127.0.0.1:${GENRM_PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${GENRM_PORT}/v1/chat/completions"
echo "[GenRM] 等待就绪（最多 ${WAIT_READY}s）..."
set +x
waited=0
until curl -sf "${HEALTH_URL}" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[GenRM] ❌ server 进程已退出（启动失败），见上方日志。"; exit 1
    fi
    if [ "${waited}" -ge "${WAIT_READY}" ]; then echo "[GenRM] ❌ 就绪超时（${WAIT_READY}s）"; exit 1; fi
    sleep 5; waited=$(( waited + 5 ))
    echo "[GenRM] ...加载中 (${waited}s)"
done
set -x
echo "[GenRM] ✅ 服务就绪（用时 ${waited}s）"

# 打印训练侧该用的 GENRM_BASE_URL（自动挑 100.x 对外 IP）
hostip="$(hostname -I | tr ' ' '\n' | grep -E '^100\.' | head -n1 || true)"
[ -z "${hostip}" ] && hostip="$(hostname -I | awk '{print $1}')"
echo "[GenRM] ★ 训练侧设置： GENRM_BASE_URL=http://${hostip}:${GENRM_PORT}"

set +x
echo "[Test] 自测一条 chat ..."
curl -sS "${CHAT_URL}" -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"请只回复两个字：正常\"}],\"max_tokens\":16,\"temperature\":0}" || true
echo
echo "[GenRM] 自测完成。"

# ============================ 保活循环（每 15s 发 128 token）============================
echo "[GenRM] 进入保活循环：每 ${KEEPALIVE_INTERVAL}s 发一条 ${KEEPALIVE_TOKENS}-token 请求。Ctrl+C 退出。"
while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[GenRM] ❌ server 进程已退出，停止保活。"; exit 1
    fi
    if curl -sS "${CHAT_URL}" -H "Content-Type: application/json" \
        -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"keepalive\"}],\"max_tokens\":${KEEPALIVE_TOKENS},\"temperature\":0}" \
        >/dev/null 2>&1; then
        echo "[GenRM] keepalive ok  $(date +%T)"
    else
        echo "[GenRM] ⚠️ keepalive 请求失败  $(date +%T)"
    fi
    sleep "${KEEPALIVE_INTERVAL}"
done
