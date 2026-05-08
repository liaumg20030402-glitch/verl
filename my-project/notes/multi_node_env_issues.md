# 多机训练环境配置踩坑笔记（面试复习向）

场景：在 H200 集群上用 verl 跑 Qwen3.5-35B-A3B（MoE）GRPO，通过公司自研 `ky` 调度器以 PtJob 形式提交到 2×8=16 卡。单机 8 卡能跑，切多机各种报错。

调度器行为：ky 在每台机上**同时**运行同一份 shell 脚本，注入 `MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE`，但**不自动组 Ray 集群**，也不是 torchrun 风格。所以要在脚本里按 RANK 分支，手工 `ray start`。

---

## 问题清单（按调试顺序）

### 问题 1：Ray 只认一个节点 → `available GPUs 8.0 < desired 16`

**现象**：`create_resource_pool()` 里抛 `ValueError: Total available GPUs 8.0 is less than total desired GPUs 16`，明明调度了 2 台机。

**排查**：只要 16 != 8×2，就一定是 Ray 没把两台机组到一个 cluster。训练驱动能启动说明 head 建好了，worker 没 join。

**根因**：第一版脚本里 head 的 `until` 轮询判定逻辑有 bug（见问题 3），实际 worker 根本没正确加入。

**面试要点**：
- 定位方法：`ray.cluster_resources()['GPU']` vs `ray.nodes()` 两个指标一起看
- 任何基于 Ray 的多机框架都先验证"ray cluster 本身对不对"，再看训练侧
- 对 ky 这种"每机一份脚本"的调度器，RANK=0 当 head 是约定成俗的做法，但需要 scheduler 注入稳定的 MASTER_ADDR/RANK

---

### 问题 2：HEAD 绑定 IP 与 Worker 连接 IP 不一致（多网卡陷阱）

**现象**：日志里 head 报 `ray start --address='11.15.230.71:20042'`，而 worker 从 `MASTER_ADDR=172.19.60.168` 去连。两个 IP 段不同，一个是 IB/计算网，一个是管理网。

**根因**：H200 节点通常有多张网卡（管理网 eno1 + IB mlx5_*）。我一开始让 head 用 `hostname -I | awk '{print $1}'` 取第一个 IP → 踩到 IB IP；worker 用的是调度器给的 `MASTER_ADDR`（管理网 IP）。Ray 的 GCS 在 head 绑的 IP 上 listen，worker 去连另一个 IP。虽然两张网卡物理可路由，但语义上 Ray cluster 就是"裂开"的。

**修复**：**head 和 worker 都从 `MASTER_ADDR` 解析**，保证两边使用同一个 IP：
```bash
HEAD_IP=$(getent hosts "${MASTER_ADDR}" | awk '{print $1}' | head -n1)
[ -z "${HEAD_IP}" ] && HEAD_IP="${MASTER_ADDR}"
```

**面试要点**：
- 多机 RDMA 集群必然多网卡，别想当然用 `hostname -I`
- `--node-ip-address` 是 Ray head 对外广告的地址，必须选一个 worker 能主动连通的
- NCCL 可以走 IB（`NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME`），但 Ray control-plane 走 TCP，两者可以不同网卡

---

### 问题 3：Ray INFO 日志污染 shell 解析，until 循环永不退出

**现象**：head 等待集群就绪的 `until` 循环跑了 600s 超时，日志里显示 `nodes=2026-04-24/2 gpus=11:55:22,486/16` —— 数字完全不对。

**根因**：我写的 python 片段只 `print(f"{alive} {gpus}")`，但 `ray.init()` 自己会往 **stdout** 打一条 `INFO worker.py:1810 -- Connecting to existing Ray cluster at address: ...`。这条 INFO 日志和我的数字混在一起，shell 里 `read -r ALIVE GPUS <<<"${STATS}"` 把日志的时间戳当成了变量值，于是比较永远不等于 `"2"` / `"16"`。

**修复**：加哨兵前缀 + grep 过滤：
```bash
STATS=$(python3 - <<'PYEOF' 2>/dev/null | grep '^__RAYSTATS__ ' | tail -n1
import ray
ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")
print(f"__RAYSTATS__ {alive} {gpus}", flush=True)
PYEOF
)
read -r _TAG ALIVE GPUS <<<"${STATS}"
```

**面试要点**：
- Shell 调 Python 取值，永远别假设 stdout 是干净的
- SDK 往 stdout 打日志是常见陷阱（Ray、transformers、HF datasets 都干过）
- 健壮做法：哨兵字符串 / JSON / 或者 stdout 全重定向只留 exit code

---

### 问题 4：until 循环里只数节点数，GPU 资源 race

**现象**：集群 2 节点都 Alive 了，但训练启动后立刻报 `available GPUs < desired GPUs`。

**根因**：Ray 注册节点和发布节点资源（GPU/CPU）**不是原子的**。我一开始只检查 `ray.nodes()` 的 Alive 数，判定通过后立刻启动训练；此时 worker node 已注册但 GPU 资源还没 publish 到 GCS，`cluster_resources()` 还是 8。

**修复**：同时校验节点数和 GPU 总数：
```python
alive = sum(1 for n in ray.nodes() if n["Alive"])
gpus = int(ray.cluster_resources().get("GPU", 0))
# 必须 alive == NNODES AND gpus == NNODES*8
```

**面试要点**：
- 分布式系统的"就绪"定义要看你真正依赖的量，不要 proxy
- 轮询循环要有 liveness（最终会进）+ safety（进了就真的就绪）两个性质

---

### 问题 5：Hydra `struct mode` 拒绝未知 key

**现象**：集群起来了、进入训练，但 `python3 -m verl.trainer.main_ppo` 秒报：
```
Could not override 'ray_kwargs.ray_init.address'.
To append to your config use +ray_kwargs.ray_init.address=auto
Key 'address' is not in struct
```

**根因**：Hydra/OmegaConf 默认在 `struct mode`，只允许覆盖 schema 已定义的 key；`ray_kwargs.ray_init` 是个空 dict，`address` 根本不存在，必须用 `+` 语法**追加**。

**修复**：`ray_kwargs.ray_init.address=auto` → `+ray_kwargs.ray_init.address=auto`。

**面试要点**：
- Hydra 语法：`=` 覆盖、`+=` 追加 list、`+` 前缀新增、`++` 前缀 force 覆盖
- 配置框架的 struct mode 是防止拼写错误、不是拦着你扩展；看到报错先想清楚是「拼错了」还是「这就是新 key」

---

### 问题 6：flashinfer JIT 编译在 NFS 上并发竞争

**现象**：集群起来、训练启动、vLLM 初始化时报：
```
subprocess.CalledProcessError: Command '[ninja, -v, -C, 
  /home3/.../.cache/flashinfer/0.6.4/90a/cached_ops/gdn_prefill_sm90, ...]'
  returned non-zero exit status 1.
```

**背景**：Qwen3.5 用 Gated Delta Net（GDN）线性注意力，vLLM 需要为 H200（SM90a）JIT 编译一个 flashinfer kernel。

**根因**：两层 race
1. `~/.cache/flashinfer/` 在 NFS 挂载的 home 上
2. 脚本开头 `rm -rf ~/.cache/flashinfer` 两台机**同时执行**，互相删对方正在写的文件
3. 删完之后 16 个 vLLM worker（2×8）并发往同一个 NFS 目录写 `build.ninja`，NFS 的 flock 不可靠，半写文件导致 ninja 报错

**修复**：
```bash
# 1. 把所有 JIT cache 重定向到节点本地 /tmp
export FLASHINFER_WORKSPACE_BASE="${tmp_run_dir}/flashinfer_cache"
export FLASHINFER_JIT_DIR="${tmp_run_dir}/flashinfer_cache/jit"
export XDG_CACHE_HOME="${tmp_run_dir}/xdg_cache"
export TRITON_CACHE_DIR="${tmp_run_dir}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${tmp_run_dir}/inductor_cache"
export VLLM_CONFIG_ROOT="${tmp_run_dir}/vllm_config"

# 2. 只让 RANK=0 清理 NFS 残留（避免两个节点同时 rm）
if [ "${RANK:-0}" == "0" ]; then
    rm -rf ~/.cache/vllm ~/.cache/torch/inductor ~/.triton/cache ~/.cache/flashinfer
fi
```

**面试要点**：
- **任何 JIT/编译缓存放 NFS 都是雷**：flashinfer / triton / torch.compile / vLLM 都带 JIT
- 修法两类：① 每节点本地 cache（现在的做法）② 集群共享 cache + 预编译一次（适合固定模型反复跑）
- `XDG_CACHE_HOME` 是 Linux 标准兜底，重定向所有走 freedesktop 规范的 cache
- 多机脚本里任何 `rm -rf ~/.cache` 都要想清楚"是谁在 rm、别人正在写什么"

---

### 问题 7：worker 保活循环的日志噪音（不影响功能，但面试可能问）

**现象**：训练日志里 worker 节点每 30s 打一行 `+ ray status` / `+ sleep 30`。

**根因**：脚本开头 `set -xeuo pipefail` 里的 `-x` 会 trace 每一条 bash 命令；worker 的保活循环里这两条命令被反复打。

**为什么保活循环必须存在**：worker 脚本一旦退出，ky 会把它标记为 done 并清理容器 → `ray start --address` 起的 daemon 进程被杀 → worker 掉出集群 → head 训练直接失败。所以 worker 必须"赖着"直到 head 训练结束。

**修复方式**：局部关掉 `-x`（`{ set +x; } 2>/dev/null`），或拉长 sleep 到 60s。

**面试要点**：
- 多机脚本里 head 和 worker 的生命周期是不对称的：head = 做正事，worker = 活到正事做完
- 别让 bash tracing 污染训练日志，用户/运维会以为是训练异常

---

### 问题 8：官方文档 `docs/start/multinode.rst` 里的 `ray job submit` 是什么？和我脚本里的 `ray.init()` 有什么区别？

**`ray job submit` 是什么**

它是一条**面向"已经在运行"的 Ray 集群的作业提交 CLI**，通过 Ray dashboard 的 HTTP API（默认端口 `:8265`）把一段 Python 脚本当成 job 丢进集群跑。**它不负责起集群**。

```bash
# 前提：集群已经通过 ray start --head / ray start --address 在两台机上起好
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo trainer.nnodes=2 ...
```

提交后 dashboard 会拉起一个子进程当 driver，可以用 `ray job list / logs / status / stop` 监控。

**与你脚本的对比**

| 维度 | `ray job submit` | 你的脚本 |
|---|---|---|
| 谁来起集群 | 不负责，需先 `ray start` | 脚本内部 `ray start --head` / `ray start --address` |
| Driver 在哪跑 | dashboard 在 head 上拉起 driver 子进程 | head 节点直接 `python3 -m verl.trainer.main_ppo` |
| `ray.init()` 在哪 | dashboard 启动 driver 时由 `main_ppo.py` 自己调用 | `main_ppo.py` 调用，连接 `address=auto` 找已有集群 |
| 集群生命周期 | 长期存活，可跑 N 个 job | 一次性：训练退出 `trap` 触发 `ray stop` 拆掉集群 |
| 日志 | `ray job logs <id>` 集中收 | 自己 `tee` 到文件 |
| 适合谁 | 常驻 Ray 集群、多人/多任务共用、交互提交 | 调度器（ky/slurm）一次性任务 |

**关键点：两者不是二选一**。`ray job submit` 仍然需要先用 `ray start` 把集群组好。它只是把"提交训练 driver"这一步从「直接 `python3 -m ...`」换成了「让 dashboard 帮你拉起来」。

**为什么 ky 场景下不用 `ray job submit`**

ky 本身就是 job 调度器，已经管了"任务排队、资源分配、节点拉起、超时清理"等责任。再套一层 `ray job submit` 等于双重 job 管理 —— 既要 ky 等到 head 节点起来 dashboard 才 listen，又要在 head 上额外保留一个常驻 dashboard 进程，没收益还增加复杂度。所以你脚本里**直接 `python3 -m verl.trainer.main_ppo` + `ray.init(address=auto)`**是最简洁的写法，把 ky 当外层调度，Ray 当内层并行框架，职责清晰。

**面试要点**：
- `ray start` = 起集群（control-plane）；`ray.init()` = driver 接入集群；`ray job submit` = 通过 dashboard 提交 driver。三者职责不同
- 选哪种取决于**集群是常驻还是一次性**：常驻 → submit + dashboard；一次性 → 脚本里 start + 直接 python
- 区分"集群生命周期"和"训练 job 生命周期"，是面试官想听的概念

---

## 串起来的一条总线

所有问题本质上围绕**三类失配**：
1. **网络拓扑的失配**（问题 2）：多网卡、hostname 解析、Ray 控制面 vs NCCL 数据面
2. **时序/一致性失配**（问题 3、4、6）：并发启动里的 race —— 节点注册 vs 资源发布、shell IO buffer、NFS flock
3. **约定/schema 失配**（问题 5、7）：Hydra struct mode、bash 选项、scheduler 约定（ky 给什么环境变量、生命周期如何）

面试时想好怎么用 30 秒讲完其中一个，然后"我后来在脚本里做了 X、Y 两层兜底"收尾。

---

## 面试话术模板

**Q：你在多机环境下遇到过什么问题？**

A（30 秒版）：
> 在 H200 上用 verl 跑 Qwen3.5-35B MoE，通过我们的 `ky` 调度器拉 2 机 16 卡。最头疼是 **Ray cluster 组不起来**：ky 每台机各跑一份脚本，我按 RANK=0 当 head、其他当 worker，但 head 用 `hostname -I` 拿到的是 IB IP，worker 用 scheduler 给的 MASTER_ADDR 是管理网 IP，两边看到的"集群地址"对不上。后来统一让两边都用 `getent hosts $MASTER_ADDR` 解析，保证一致。

A（扩展版可以再讲 flashinfer NFS 并发编译和 Hydra `+` 前缀这两个）。

**Q：怎么定位是 Ray 没组起来，不是训练本身问题？**

A：verl 启动时会做 `ray.cluster_resources()` vs `trainer.nnodes × trainer.n_gpus_per_node` 对齐检查，报 `available GPUs X < desired Y`。看到这个错就先 `ray status` 看集群对不对，别去动训练配置。
