#!/usr/bin/env bash
# 起 GenRM(裁判) SGLang 服务——用来和 vLLM 多实例版对比。
#
# 和 vLLM 版的关键区别：SGLang 的 --dp-size 自带 data-parallel 路由器，
#   一条命令、**一个端口**、内部把请求负载均衡到 N 个副本 → 在线扩吞吐很优雅。
#   所以这里只起【一个】sglang server，用 TP×DP 吃满 8 卡，对外只有一个地址，
#   奖励函数的 GENRM_BASE_URL 直接填这一个即可（不用逗号列表）。
#
#   总卡数 = TP_SIZE × DP_SIZE，必须等于 GENRM_GPUS 的卡数。
#   例：8 卡 → TP_SIZE=2 DP_SIZE=4（4 副本各 TP=2），或 TP_SIZE=1 DP_SIZE=8（8 副本纯 DP）。
#
# ⚠️ 头等大事：先看它**能不能把 Qwen3.5(GDN 线性注意力)加载起来**。
#   GDN 是新架构，SGLang 不一定支持/可能要特定 attention backend。加载报错就说明这版 SGLang
#   不支持 GDN，回 vLLM。日志在 /tmp/jmli27_sglang.log。
#
# 用法：bash start_genrm_server_sglang.sh          起服务、挂住
#       bash start_genrm_server_sglang.sh --test   对已运行的端口 curl
#
# 保活/压测同样用外部脚本 genrm_load_test.py（单地址直接压）。

set -xeuo pipefail

# ⚠️ vLLM 和 SGLang 强烈建议**分开两个 conda 环境**——共存常因 flashinfer/torch/transformers/
#   sgl-kernel 版本互相打架而崩。你不需要让它们在同一个 env 里：判分 server 是独立进程、走 HTTP，
#   奖励函数也只发 HTTP、不 import 任何引擎。所以：训练继续用 vLLM 的 verl_rl_v2，
#   这个 SGLang server 用**单独**的环境（先 `conda create -n sglang_genrm ...; pip install "sglang[all]"`），
#   用 CONDA_ENV 覆盖即可，完全不动你能跑的训练环境。
CONDA_ENV=${CONDA_ENV:-sglang_infer}
source /home3/medcog/jycai6/.bashrc
conda activate "${CONDA_ENV}"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# 同 vLLM 版：强制用 conda 的新 libstdc++，规避 CXXABI 缺失
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# 编译缓存指到本地 /tmp（单进程，一份即可，不存在多实例竞争）
cache_root="/tmp/jmli27_sglang_cache_$(hostname -s)"
export TRITON_CACHE_DIR="${cache_root}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor_cache"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${cache_root}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${cache_root}/xdg_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}"

# ============================ 可调参数 ============================
GENRM_MODEL_PATH=${GENRM_MODEL_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}
SERVED_NAME=${SERVED_NAME:-"genrm_remote"}            # 必须与训练侧 GENRM_MODEL_NAME 一致
TP_SIZE=${TP_SIZE:-1}                                  # 每个 DP 副本的张量并行度
DP_SIZE=${DP_SIZE:-8}                                  # 数据并行副本数（内部路由器负载均衡）
GENRM_GPUS=${GENRM_GPUS:-"0,1,2,3,4,5,6,7"}           # 圈给 GenRM 的卡；卡数必须 = TP_SIZE×DP_SIZE
GENRM_PORT=${GENRM_PORT:-8000}                         # 单端口（SGLang 内部路由到各 DP 副本）
GENRM_HOST=${GENRM_HOST:-0.0.0.0}                      # 0.0.0.0 才能被别的节点访问
CONTEXT_LEN=${CONTEXT_LEN:-40960}                      # = vLLM 的 max-model-len
MEM_FRAC=${MEM_FRAC:-0.90}                             # = vLLM 的 gpu-memory-utilization
MAX_RUNNING=${MAX_RUNNING:-0}                          # >0 时设 --max-running-requests（每副本最大并发）；0=不设，sglang自动按显存推理
# --attention-backend 只控制【全注意力层】（Qwen3.5 每 4 层一个），**不是 GDN bug 所在**。
#   可选: flashinfer/fa3/fa4/triton/trtllm_mha。H200 上 flashinfer 起步；别用 fa3（cudagraph bug #22800）。
ATTN_BACKEND=${ATTN_BACKEND:-"flashinfer"}
# ⭐ GDN 在本版 SGLang 里归到 Mamba/SSM 那套，控制 flag 是 --mamba-backend（不是 --linear-attn-*）。
#   H200 默认 flashinfer → 会踩 mbarrier 死锁 + no_buffer 调度下精度退化(#20791)；triton FLA 两者都不受影响。
#   这就是 vLLM 的 --gdn-prefill-backend triton 的等价物，务必 triton。
MAMBA_BACKEND=${MAMBA_BACKEND:-"triton"}               # triton/flashinfer；务必 triton
ENFORCE_EAGER=${ENFORCE_EAGER:-0}                      # 1 → --disable-cuda-graph（关 CUDA graph 兜底死锁）
WAIT_READY=${WAIT_READY:-1800}                         # 等就绪上限(秒)；27B 冷启动慢
MONITOR_INTERVAL=${MONITOR_INTERVAL:-30}              # 前台监控心跳间隔(秒)

export CUDA_VISIBLE_DEVICES="${GENRM_GPUS}"
IFS=',' read -ra _gpu_arr <<< "${GENRM_GPUS}"
TOTAL_GPUS=${#_gpu_arr[@]}
NEED_GPUS=$(( TP_SIZE * DP_SIZE ))
if [ "${TOTAL_GPUS}" -ne "${NEED_GPUS}" ]; then
    echo "[SGLang] ❌ 卡数(${TOTAL_GPUS}) != TP_SIZE×DP_SIZE(${TP_SIZE}×${DP_SIZE}=${NEED_GPUS})"; exit 1
fi
echo "[SGLang] 单 server：TP=${TP_SIZE} DP=${DP_SIZE}（${DP_SIZE} 副本，内部路由），port=${GENRM_PORT}"

HEALTH_URL="http://127.0.0.1:${GENRM_PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${GENRM_PORT}/v1/chat/completions"

# ============================ 自测模式 ============================
if [[ "${1:-}" == "--test" ]]; then
    echo "[Test] /v1/models ..."
    curl -sS "${HEALTH_URL}" | head -c 400; echo
    echo "[Test] 一条 chat ..."
    curl -sS "${CHAT_URL}" -H "Content-Type: application/json" \
        -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"请只回复两个字：正常\"}],\"max_tokens\":16,\"temperature\":0}"
    echo
    exit 0
fi

# ============================ 启动单个 SGLang server ============================
logf="/logs/sglang.log"
# 组装可选参数（数组形式，避免空值塞进命令行）
extra_args=()
[ -n "${ATTN_BACKEND}" ] && extra_args+=(--attention-backend "${ATTN_BACKEND}")
# GDN/SSM 内核后端切 triton（关键：规避 flashinfer mbarrier 死锁 + no_buffer 精度退化 #20791）
[ -n "${MAMBA_BACKEND}" ] && extra_args+=(--mamba-backend "${MAMBA_BACKEND}")
[ "${MAX_RUNNING}" -gt 0 ] && extra_args+=(--max-running-requests "${MAX_RUNNING}")
[ "${ENFORCE_EAGER}" = "1" ] && extra_args+=(--disable-cuda-graph)

echo "[SGLang] 启动：model=${GENRM_MODEL_PATH} served=${SERVED_NAME} TP=${TP_SIZE} DP=${DP_SIZE} log=${logf}"
python -m sglang.launch_server \
    --model-path "${GENRM_MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tp-size "${TP_SIZE}" \
    --dp-size "${DP_SIZE}" \
    --host "${GENRM_HOST}" \
    --port "${GENRM_PORT}" \
    --dtype bfloat16 \
    --context-length "${CONTEXT_LEN}" \
    --mem-fraction-static "${MEM_FRAC}" \
    --trust-remote-code \
    "${extra_args[@]}" \
    > "${logf}" 2>&1 &
SERVER_PID=$!
trap 'echo "[SGLang] 停止 server (pid=${SERVER_PID}) ..."; kill ${SERVER_PID} 2>/dev/null || true' EXIT INT TERM

# ============================ 等就绪 ============================
echo "[SGLang] 等待就绪（最多 ${WAIT_READY}s；日志 ${logf}）..."
set +x
waited=0
until curl -sf "${HEALTH_URL}" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[SGLang] ❌ server 进程已退出（很可能 SGLang 不支持 GDN，或参数不对），看 ${logf}。"; exit 1
    fi
    if [ "${waited}" -ge "${WAIT_READY}" ]; then echo "[SGLang] ❌ 就绪超时（${WAIT_READY}s）"; exit 1; fi
    sleep 5; waited=$(( waited + 5 ))
    echo "[SGLang] ...加载中 (${waited}s)"
done
set -x
echo "[SGLang] ✅ 服务就绪（用时 ${waited}s）"

# 打印训练侧该用的 GENRM_BASE_URL（单地址；DP 路由在 server 内部）
hostip="$(hostname -I | tr ' ' '\n' | grep -E '^100\.' | head -n1 || true)"
[ -z "${hostip}" ] && hostip="$(hostname -I | awk '{print $1}')"
echo "[SGLang] ★ 训练侧设置（单地址即可，DP 路由在 server 内部）："
echo "[SGLang]   GENRM_BASE_URL=http://${hostip}:${GENRM_PORT}"

# 自测一条 chat
set +x
echo "[Test] 自测一条 chat ..."
curl -sS "${CHAT_URL}" -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"请只回复两个字：正常\"}],\"max_tokens\":16,\"temperature\":0}" || true
echo
echo "[SGLang] 自测完成：返回 JSON 即 OK。"

# ============================ 前台挂住（不发请求，只监控）============================
# 保活/压测用外部脚本 genrm_load_test.py（单地址直接压）。这里只监控进程是否还活着。
echo "[SGLang] 进入前台监控（每 ${MONITOR_INTERVAL}s 检查）。保活请另跑 genrm_load_test.py。Ctrl+C 退出。"
while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[SGLang] ❌ server 进程退出，看 ${logf}。"; exit 1
    fi
    echo "[SGLang] alive  $(date +%T)"
    sleep "${MONITOR_INTERVAL}"
done
