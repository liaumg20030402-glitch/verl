#!/usr/bin/env bash
# 起 GenRM(裁判) vLLM 服务——**多进程极简版**（外部 server 模式）。
#
# 设计：起 N 个**独立**的 vllm serve（各 TP=GENRM_TP、纯 TP、不加 --data-parallel-size），
#   各占连续的卡、各监听不同端口；奖励函数把 GENRM_BASE_URL 填成这 N 个地址的逗号列表，
#   按请求随机挑一个 → 等价 DP=N 的吞吐、但避开 vllm 在线 DP 的坑(#22361/#41862)。
#
# 本脚本**只负责起服务并把它们挂住**，不再内置保活循环。
#   保活/压测一律用外部脚本 genrm_load_test.py（随机 prompt 才能把 GPU 打到高利用率，
#   固定 prompt 会命中 prefix cache、利用率虚低被集群杀——血泪教训）。
#   防空闲被杀：另起一个窗口跑 `python3 genrm_load_test.py --concurrency 8 --duration 999999`
#   （低并发兜底即可；训练真实判分负载本身也会喂活它）。
#
# 必要的稳定性设置（保留）：
#   - 每个实例**独立编译缓存目录**（按端口隔离）——多个独立 vllm 进程不能抢同一份 triton/inductor 缓存。
#   - GDN 必须 --gdn-prefill-backend triton（#41862）。CUDA graph 默认开（吞吐高）。
#   - 27B bf16(~54GB) 单卡 H200 放得下：要更稳更快可设 GENRM_TP=1 → 8 实例、且无 worker 间 shm_broadcast。
#
# 用法：bash start_genrm_server.sh          起服务、挂住
#       bash start_genrm_server.sh --test   对已运行的各端口逐个 curl

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

# 编译缓存基目录（真正的缓存目录在下面每个实例按端口隔离，避免多进程 JIT 抢同一份写坏）
CACHE_BASE="/tmp/jmli27_genrm_cache_$(hostname -s)"

# ============================ 可调参数 ============================
GENRM_MODEL_PATH=${GENRM_MODEL_PATH:-"/train21/medcog/permanent/leijiang19/pretrain_models/Qwen3.5-27B"}
SERVED_NAME=${SERVED_NAME:-"genrm_remote"}            # 必须与训练侧 GENRM_MODEL_NAME 一致
GENRM_TP=${GENRM_TP:-2}                                # 每个实例的 TP（纯 TP，无 DP）；设 1 → 8 实例更稳更快
GENRM_GPUS=${GENRM_GPUS:-"0,1,2,3,4,5,6,7"}           # 圈给 GenRM 的卡
BASE_PORT=${BASE_PORT:-8000}                           # 第 1 个实例端口，后续 +1
GENRM_HOST=${GENRM_HOST:-0.0.0.0}                      # 0.0.0.0 才能被别的节点访问
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}                  # judge prompt = question+model_answer+模板+GT，留足余量
GPU_MEM=${GPU_MEM:-0.90}                                # 每个实例独享自己的卡，可拉高
WAIT_READY=${WAIT_READY:-1800}                          # 等就绪上限(秒)；27B 冷启动慢，给足
MONITOR_INTERVAL=${MONITOR_INTERVAL:-30}               # 前台监控心跳间隔(秒)

# 实例数 N = 圈的卡数 / GENRM_TP；卡数必须能被 TP 整除
IFS=',' read -ra _gpu_arr <<< "${GENRM_GPUS}"
TOTAL_GPUS=${#_gpu_arr[@]}
if [ $(( TOTAL_GPUS % GENRM_TP )) -ne 0 ]; then
    echo "[GenRM] ❌ 卡数(${TOTAL_GPUS}) 不能被 GENRM_TP(${GENRM_TP}) 整除"; exit 1
fi
N_INSTANCES=$(( TOTAL_GPUS / GENRM_TP ))
echo "[GenRM] 将起 ${N_INSTANCES} 个实例（每个 TP=${GENRM_TP}），端口 ${BASE_PORT}..$(( BASE_PORT + N_INSTANCES - 1 ))"

# ============================ 自测模式 ============================
if [[ "${1:-}" == "--test" ]]; then
    for i in $(seq 0 $(( N_INSTANCES - 1 ))); do
        port=$(( BASE_PORT + i ))
        echo "[Test] 实例 ${i} :${port} /v1/models ..."
        curl -sS "http://127.0.0.1:${port}/v1/models" | head -c 300; echo
    done
    exit 0
fi

# ============================ 启动 N 个独立 server ============================
declare -a SERVER_PIDS=()
declare -a PORTS=()
for i in $(seq 0 $(( N_INSTANCES - 1 ))); do
    port=$(( BASE_PORT + i ))
    gpus="$(IFS=,; echo "${_gpu_arr[*]:$(( i * GENRM_TP )):${GENRM_TP}}")"   # 该实例的连续 TP 张卡
    logf="/tmp/jmli27_genrm_${port}.log"
    echo "[GenRM] 实例 ${i}: gpus=${gpus} port=${port} log=${logf}"
    # 子 shell 给每个实例一份独立缓存目录（按端口隔离）；exec 让子 shell 直接变成 vllm 进程，
    # 故 $! 就是 vllm 的 PID，trap/kill-0 照常生效。
    (
        inst_cache="${CACHE_BASE}_${port}"
        export TRITON_CACHE_DIR="${inst_cache}/triton_cache"
        export TORCHINDUCTOR_CACHE_DIR="${inst_cache}/inductor_cache"
        export VLLM_CONFIG_ROOT="${inst_cache}/vllm_config"
        export FLASHINFER_WORKSPACE_BASE="${inst_cache}/flashinfer_cache"
        export FLASHINFER_JIT_DIR="${inst_cache}/flashinfer_cache/jit"
        export XDG_CACHE_HOME="${inst_cache}/xdg_cache"
        mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CONFIG_ROOT}" \
                 "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}"
        export CUDA_VISIBLE_DEVICES="${gpus}"
        exec vllm serve "${GENRM_MODEL_PATH}" \
            --served-model-name "${SERVED_NAME}" \
            --tensor-parallel-size "${GENRM_TP}" \
            --host "${GENRM_HOST}" \
            --port "${port}" \
            --dtype bfloat16 \
            --max-model-len "${MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${GPU_MEM}" \
            --trust-remote-code \
            --gdn-prefill-backend triton
    ) > "${logf}" 2>&1 &
    SERVER_PIDS+=($!)
    PORTS+=("${port}")
done
# Ctrl+C / 退出时把所有实例一起关掉
trap 'echo "[GenRM] 停止所有 server ..."; kill ${SERVER_PIDS[*]} 2>/dev/null || true' EXIT INT TERM

# ============================ 等所有实例就绪 ============================
echo "[GenRM] 等待 ${N_INSTANCES} 个实例就绪（最多 ${WAIT_READY}s；各实例日志 /tmp/jmli27_genrm_<port>.log）..."
set +x
waited=0
while true; do
    for pid in "${SERVER_PIDS[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[GenRM] ❌ 有实例进程退出（启动失败），看 /tmp/jmli27_genrm_*.log。"; exit 1
        fi
    done
    ready=0
    for port in "${PORTS[@]}"; do
        curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && ready=$(( ready + 1 ))
    done
    echo "[GenRM] ready ${ready}/${N_INSTANCES} (${waited}s)"
    [ "${ready}" -eq "${N_INSTANCES}" ] && break
    if [ "${waited}" -ge "${WAIT_READY}" ]; then echo "[GenRM] ❌ 就绪超时（${WAIT_READY}s）"; exit 1; fi
    sleep 5; waited=$(( waited + 5 ))
done
set -x
echo "[GenRM] ✅ 全部 ${N_INSTANCES} 个实例就绪（用时 ${waited}s）"

# 拼出训练侧该用的 GENRM_BASE_URL 逗号列表（自动挑 100.x 对外 IP）
hostip="$(hostname -I | tr ' ' '\n' | grep -E '^100\.' | head -n1 || true)"
[ -z "${hostip}" ] && hostip="$(hostname -I | awk '{print $1}')"
URLS=""
for port in "${PORTS[@]}"; do URLS="${URLS:+${URLS},}http://${hostip}:${port}"; done
echo "[GenRM] ★ 训练侧设置（逗号列表，奖励函数随机轮询各实例）："
echo "[GenRM]   GENRM_BASE_URL=${URLS}"

# 自测：每个实例发一条 chat
set +x
for port in "${PORTS[@]}"; do
    echo "[Test] :${port} chat:"
    curl -sS "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"请只回复两个字：正常\"}],\"max_tokens\":16,\"temperature\":0}" || true
    echo
done
echo "[GenRM] 自测完成：各端口返回 JSON 即 OK。"

# ============================ 前台挂住（不发请求，只监控）============================
# 保活/压测请用外部脚本 genrm_load_test.py。这里只是把 server 挂在前台、并监控进程是否还活着；
# 任一实例进程退出就报错退出（trap 会把其余实例一起关掉）。
echo "[GenRM] 进入前台监控（每 ${MONITOR_INTERVAL}s 检查一次）。保活请另跑 genrm_load_test.py。Ctrl+C 退出。"
while true; do
    for pid in "${SERVER_PIDS[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[GenRM] ❌ 有实例进程退出，看 /tmp/jmli27_genrm_*.log。"; exit 1
        fi
    done
    echo "[GenRM] alive ${N_INSTANCES}/${N_INSTANCES}  $(date +%T)"
    sleep "${MONITOR_INTERVAL}"
done
