# Qwen3.5 多机训练 hang 根因总结：FlashInfer GDN kernel + 解药

> **TL;DR**：Qwen3.5 系列在 vLLM TP=8、多机 RL 训练（GRPO 等）下随机 step 挂死的"老毛病"
> ——根因在 FlashInfer 的 `chunk_gated_delta_rule` kernel，CUDA Graph replay 下
> 触发 **warp-specialized mbarrier 死锁**。
> **解药**：vLLM 配置 `--gdn-prefill-backend triton`，把 GDN 前向从 FlashInfer 换成 Triton。

---

## 一、症状总结

### 现象
- Qwen3.5（27B Dense / 35B-A3B / Next 等带 GDN 的版本）
- 多机训练（一般 2 节点 16 卡起）
- vLLM rollout，TP=8，async 模式
- 训练跑 **几步到几十步随机挂死**
- GPU 利用率掉到 0%，无报错，**10 分钟后 NCCL watchdog 触发 timeout**
- 报错样子：
  ```
  WorkNCCL(SeqNum=N, OpType=_ALLGATHER_BASE,
           NumelIn=M, NumelOut=8M, Timeout(ms)=600000)
  ran for 600007 milliseconds before timing out.
  ```

### py-spy 抓到的关键证据（来自 [ms-swift#8506](https://github.com/modelscope/ms-swift/issues/8506)）

```text
Rank 1（卡死的元凶）:
    gdn_prefill (flashinfer/gdn_prefill.py:52)
    chunk_gated_delta_rule (flashinfer/gdn_prefill.py:196)
    fi_chunk_gated_delta_rule (vllm/.../qwen3_next.py:138)
    gdn_attention_core (vllm/.../qwen3_next.py:1451)

Rank 0, 2-7（等 Rank 1 的 7 个）:
    all_reduce (vllm/.../symm_mem.py:148)
    all_reduce (vllm/.../cuda_communicator.py:201)
    _all_reduce_out_place (vllm/.../parallel_state.py:514)
```

**Rank 1 还在算 GDN attention kernel；其他 7 个已经过了 GDN，在后面那个 all_reduce 上等死**。
TP 内 rank 之间本应 lockstep —— 但 GDN kernel **打破了这个假设**。

---

## 二、根因（来自 [flashinfer#3329](https://github.com/flashinfer-ai/flashinfer/issues/3329)）

cuda-gdb 调试发现：

> "Kernel deadlocks on a **single SM block**; other blocks of the same launch complete normally.
> cuda-gdb shows the stuck warp parked on an **mbarrier wait** —
> looks like a warp-specialized **producer/consumer mbarrier mis-arrival**."

翻译：
- GDN kernel 用了 warp-specialized 编程模式（warp 分工：一组 producer 取数据，一组 consumer 算）
- 它们用 CUDA `mbarrier`（memory barrier）同步
- **某个 SM 的某个 warp 永远等不到对面的 mbarrier 信号** → 永久挂死
- **只在 CUDA Graph capture + replay 路径下触发，直接调 kernel 不挂**
- hang 概率跟 batch 形状强相关：特定 N 值下 60-100% 触发

### 跟我的数据相关吗？是

flashinfer#3329 给的触发 shape：`cu_seqlens = [prefill_T] + [6]*(N-1)`
- 一个 prefill 序列 + N-1 个 decode 步
- N=73, 81, 97 时 hang 概率 60-100%

vLLM continuous batching 模式下，**每一步都是 1 prefill + 几百 decode**。
长 response（max=16384）= active request 数 N 经常落到 80-100+ 的高 hang 概率区间。
**长尾数据 = 高 hang 概率**。

---

## 三、解药：切到 Triton 后端

flashinfer#3329 原文：

> "Switching the prefill backend to **Triton/FLA** (vLLM's `--gdn-prefill-backend triton`)
> at the exact same shapes: **0/35 hangs**.
> Costs ~1.7-1.9× per-call kernel time vs flashinfer."

**35 次重试 0 次挂，1.7-1.9× kernel 慢（整体 rollout 慢 5-10%）**。

### 在 verl 里怎么开（vLLM 后端）

`ROLLOUT` 数组里加一行：

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
```

vLLM 0.18.0 源码确认（[engine/arg_utils.py:618](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/engine/arg_utils.py#L618)）：

```python
gdn_prefill_backend: Literal["flashinfer", "triton"] | None = None
```

### 在 verl 里怎么开（SGLang 后端）

**SGLang 也有同样的 bug**——默认在 H100/H200 上用 FlashInfer 跑 GDN，
所以 SGLang 用户同样需要切到 triton。

SGLang `server_args.py:275`：
```python
LINEAR_ATTN_KERNEL_BACKEND_CHOICES = ["triton", "cutedsl", "flashinfer"]
```

SGLang 默认行为（`server_args.py:3073`）："defaulting --linear-attn-decode-backend to flashinfer"
——SM90 (H100/H200) 上**默认走 FlashInfer**，跟 vLLM 一模一样。

CLI 三个相关 flag：
- `--linear-attn-backend` — 同时设 prefill + decode
- `--linear-attn-prefill-backend` — 单独覆盖 prefill（**bug 在这条路径**）
- `--linear-attn-decode-backend` — 单独覆盖 decode

verl 配置（`rollout.name=sglang` 时）：
```bash
# 只切 prefill（最小变量）
+actor_rollout_ref.rollout.engine_kwargs.sglang.linear_attn_prefill_backend=triton

# 或者 prefill + decode 都切（更激进的兜底）
+actor_rollout_ref.rollout.engine_kwargs.sglang.linear_attn_backend=triton
```

### vLLM vs SGLang 对照表（统一参考）

| | vLLM | SGLang |
|---|---|---|
| 默认 GDN 后端（H200）| FlashInfer | FlashInfer |
| 触发 mbarrier bug | ✅ | ✅ |
| 切 triton 的 CLI flag | `--gdn-prefill-backend triton` | `--linear-attn-prefill-backend triton` |
| verl 透传的 hydra key | `engine_kwargs.vllm.gdn_prefill_backend=triton` | `engine_kwargs.sglang.linear_attn_prefill_backend=triton` |
| 性能代价 | ~5-10% rollout 慢 | ~5-10% rollout 慢 |

### verl 透传机制（两个引擎共用）

[vllm_async_server.py:212](verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L212)：
```python
engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
... **engine_kwargs   # 传给 vLLM EngineArgs
```

SGLang 也是同样的 `engine_kwargs` 机制，verl 用 `_get_engine_kwargs_key()` 根据
`rollout.name` 自动选 key（"vllm" 或 "sglang"）。

### 启动后验证

```bash
# 1. 检查 hydra 是否透传成功
grep -A 3 "engine_kwargs" ${CKPTS_DIR}/logs/run_*.log | head

# 2. 检查实际选了哪个 backend
# vLLM:
grep -iE "gdn.*backend|prefill.*backend" ${CKPTS_DIR}/logs/run_*.log | head
# SGLang:
grep -iE "linear.attn.*backend|gdn.*backend" ${CKPTS_DIR}/logs/run_*.log | head
```

---

## 四、知识点扩展：FlashInfer / FLA / Triton 三者关系

这是用户经常搞混的事情。**它们不是完全并列**，下面这张图厘清。

```
┌─────────────────────────────────────────────────────────────┐
│  Triton（OpenAI 出的 GPU kernel 编译器 / DSL）              │
│  - 是一种"写 kernel 的语言/编译器"，不是 kernel 库本身       │
│  - 类似 CUDA 但 Python 风格、autotuning 友好                 │
│  - 是工具，不是产品                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ 用 Triton 写出来的库
                           ↓
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼────────┐                  ┌─────────▼──────────┐
│  FLA           │                  │  vLLM 内置 Triton   │
│  (flash-       │                  │  attention kernel   │
│   linear-      │                  │  (vLLM 自己实现)    │
│   attention)   │                  │                     │
│                │                  │                     │
│ - 专门做       │                  │ - vLLM 写的         │
│   线性注意力   │                  │   Triton kernel     │
│ - Mamba/GDN/   │                  │ - 标准 MHA 也有     │
│   RWKV/RetNet  │                  │   Triton 版本       │
│ - 第三方库     │                  │ - 内置              │
└────────────────┘                  └─────────────────────┘

————————— 跟 Triton 这条线并行的另一条 —————————

┌─────────────────────────────────────────────────────────────┐
│  FlashInfer（NVIDIA 支持的高性能 attention kernel 库）       │
│  - 直接用 CUDA C++ 写（部分用 CUTLASS 模板）                 │
│  - 不是用 Triton 写的                                        │
│  - 是个"成品 kernel 库"，类似 FlashAttention 的竞品           │
│  - 包含 MHA / MLA / GDN / paged attention 等                 │
│  - 在 vLLM 里作为一种 backend 选择                            │
└─────────────────────────────────────────────────────────────┘
```

### 一行式总结

| 名字 | 是什么 | 跟 GDN 的关系 |
|---|---|---|
| **Triton** | GPU kernel 编译器 + 语言（OpenAI） | 用来**写** GDN kernel 的工具 |
| **FLA**（flash-linear-attention） | 线性注意力 kernel **库**，用 Triton 实现 | 提供 **Triton 版** GDN kernel |
| **FlashInfer** | 高性能 attention kernel 库，用 CUDA/CUTLASS 写 | 提供 **CUDA 版** GDN kernel（**有 bug 那个**）|
| **flash-attn**（FlashAttention） | 标准 MHA / GQA 的高性能实现（Tri Dao 等），CUDA 写 | **不做 GDN**，只做标准注意力 |

### `flash-attn` 跟 `flash-linear-attention` 的区别

很多人（包括之前的我）把这俩搞混。它们是**互补关系，不是替代关系**——一个负责标准注意力层，一个负责线性注意力层。

| 维度 | **flash-attn** | **flash-linear-attention (FLA)** |
|---|---|---|
| 作者 / 来源 | Tri Dao 等（DAO-AI Lab） | sustcsonglin 等（开源社区） |
| 实现语言 | CUDA C++ / CUTLASS（v3 开始） | **Triton** |
| 复杂度 | O(N²) attention（用 tiling 优化访存）| **O(N) 线性 attention** |
| 处理的层 | 标准 MHA / GQA / sliding window / MLA | **Mamba / GDN / RWKV / RetNet / Gated Linear Attention** |
| pip 包名 | `flash-attn` | `flash-linear-attention` |
| Python 入口 | `from flash_attn import flash_attn_func` | `from fla.layers import GatedDeltaNet` 等 |
| 用在 Qwen3.5 哪些层 | 标准 attention 层（hybrid 模型里非 GDN 层）| GDN 层 |

**hybrid 模型为什么两个都需要**：

Qwen3.5 是 **hybrid 架构**——一部分层是标准 attention，一部分层是 GDN：

```
Qwen3.5-27B 共 N 层（伪代码）:
  layer 0:  GDN (linear attention)        ← 走 FLA 或 FlashInfer-GDN
  layer 1:  Standard attention (GQA)      ← 走 flash-attn 或 FlashInfer-MHA
  layer 2:  GDN                            ← 走 FLA
  layer 3:  Standard attention             ← 走 flash-attn
  ...      （交替）
```

所以你 [install_verl_rl-v2.sh](verl/my-project/scripts/install_verl_rl-v2.sh) 里同时装这两个：

```bash
pip install "flash-attn==2.8.3" --no-build-isolation -v          # 标准 attention 层用
pip install "flash-linear-attention==0.4.2" --no-build-isolation -v  # GDN 层用
```

**这两个都不可缺**——少了 flash-attn 标准 attention 层算不动，少了 FLA 训练侧 GDN 层算不动。

### 跟 FlashInfer 的层级关系总结

```
                         一个 Qwen3.5 forward pass
                                    │
                ┌───────────────────┴───────────────────┐
                ↓                                       ↓
        标准 attention 层                            GDN 层
                │                                       │
        ┌───────┴────────┐                  ┌──────────┴──────────┐
        ↓                ↓                  ↓                     ↓
   flash-attn      FlashInfer            FLA                FlashInfer
   (训练侧用)      (vLLM 推理侧用)      (Triton 实现)       (CUDA 实现 ← bug 在这)
                                       训练侧 + vLLM 可选    vLLM 默认
```

**vLLM 推理时**：
- 标准 attention 走 FlashInfer 的 MHA kernel（**没 bug**）
- GDN 走 FlashInfer 的 GDN kernel（**有 mbarrier 死锁 bug**）→ 切到 Triton 修复

**训练时（Megatron / HF）**：
- 标准 attention 走 flash-attn
- GDN 走 FLA
- **从头到尾都没用 FlashInfer**，所以训练侧不受这个 bug 影响

### "Triton/FLA" 在 flashinfer#3329 里指什么？

flashinfer#3329 作者写的 "Triton/FLA" 是**两个名字指同一件事**：
- vLLM 的 `--gdn-prefill-backend triton` 选项
- 内部实际调用的是 **FLA 库里用 Triton 写的 GDN kernel**

所以 **FLA 和 Triton 不是并列**——**FLA 用 Triton 写**。
**FLA 和 FlashInfer 才是并列**（两个独立的 kernel 库实现，一个用 Triton，一个用 CUDA）。

### vLLM 里 GDN 后端的实际选择

```python
# vllm/engine/arg_utils.py
gdn_prefill_backend: Literal["flashinfer", "triton"]
```

- `"flashinfer"` → 用 FlashInfer 库的 CUDA 实现（**默认，有 mbarrier 死锁**）
- `"triton"` → 用 Triton 实现的 GDN kernel（**FLA 库或 vLLM 内置的 Triton 版**）

vLLM main 分支还多了 `"cutedsl"`（Blackwell 用 CUTE DSL 写），0.18 没有。

### 跟 Megatron 训练侧的 attention backend 是不同概念

[my-run_qwen3_5-27b-megatron-v3.sh:293](verl/my-project/scripts/my-run_qwen3_5-27b-megatron-v3.sh#L293)：
```bash
++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
```

这是 **Megatron 训练侧** 的 attention 后端（用 TransformerEngine 里包的 FlashAttention）。
**跟 vLLM rollout 侧的 GDN 后端是完全独立的两套东西**。
GDN bug 只影响 vLLM rollout 侧，Megatron 训练侧不受影响（Megatron 自己实现的 GDN 路径不同）。

---

## 五、GDN（Gated Delta Net）简介

### 这是什么
**Gated Delta Net 是 Qwen3.5 系列引入的线性注意力变体**。

| 维度 | 标准注意力 (MHA) | GDN（Gated Delta Net）|
|---|---|---|
| 计算复杂度 | O(N²) 序列长度平方 | **O(N) 线性** |
| 长序列友好性 | 差 | 好 |
| KV cache | 完整 KV cache | **常数大小 state**（不随长度增长）|
| 思想来源 | Vaswani 2017 | DeltaNet（2024）+ 门控 |

### 为什么 Qwen3.5 要用它

1. **支持超长上下文**：O(N) 复杂度 + 常数 KV → 100K+ token 不爆显存
2. **推理便宜**：每个 token 只需要常数 state，不像 MHA 要扫一遍完整 KV
3. **混合架构**：Qwen3.5 通常**几层 GDN + 几层标准 attention 交替**（hybrid），兼顾长程依赖和精度

### 跟 Mamba 的关系

GDN 和 Mamba（SSM）是亲戚关系：
- 都是**线性 RNN-like** 结构
- 都用"压缩 state"代替"完整 KV cache"
- Mamba 用**选择性 SSM**做 gating，GDN 用**delta rule + gate**

### 为什么 GDN kernel 容易写出 bug

GDN 的数学结构比标准 attention 复杂：

```
GDN 核心递推:
S_t = S_{t-1} - β_t · S_{t-1} · k_t · k_t^T  +  v_t · k_t^T
      ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                forget                            insert
out_t = q_t^T · S_t
```

- **state S 是个矩阵**（不是向量）
- 需要**逐 token 更新**（不像 attention 可以并行处理整个序列）
- 实际 kernel 把序列切 chunk，**chunk 内并行 + chunk 间串行**

工程复杂度：
- 需要 warp 协作（producer/consumer 模式）
- 需要 mbarrier 同步
- 不同 chunk 长度（特别是末尾残余）需要不同处理路径
- → FlashInfer 的 CUDA 实现里某个 warp **极少数情况 mbarrier mis-arrival → 死锁**

Triton 版（FLA）不死锁的原因：
- Triton 编译器**不用复杂的 warp specialization**，用更简单的 block-level 并行
- 没有手写 mbarrier，bug 面更小
- 代价是 kernel 慢 1.7-1.9×

---

## 五点五、为什么 SFT 不会触发这个 bug，只有 RL 训练才会

**关键观察**：同一个 Qwen3.5 模型，用 ms-swift 做 SFT 训练**完全不会**遇到这个 hang。
原因是 SFT 和 RL 的执行路径完全不同。

### 三条路径对照

| 阶段 | SFT（如 swift） | RL 训练（如 verl 的 train_step） | RL rollout（如 verl 的 vLLM 推理） |
|---|---|---|---|
| 推理引擎 | **无**（不需要）| **无**（直接 Megatron / FSDP forward）| **有**（vLLM）|
| GDN 实现 | **FLA (Triton)** | **FLA (Triton)** | **FlashInfer (CUDA)** ← 默认 |
| CUDA Graph | **不开** | 一般不开 | **默认开**（vLLM 关键优化） |
| 是否触发 GDN bug | ❌ **否** | ❌ **否** | ✅ **会** |

bug 三个必要条件：
1. **FlashInfer 的 GDN kernel**（不是 FLA Triton 版）
2. **CUDA Graph capture + replay**（不是直接调用）
3. 特定 batch 形状（continuous batching 下的 prefill + many decode）

**SFT 三个条件一个都不满足，所以稳如老狗**。
**RL 训练侧（Megatron forward / backward）也都不满足，所以也稳**。
**只有 RL rollout 侧（vLLM 推理）三个条件全占齐，才挂**。

### 验证：你 SFT 环境里没有 flashinfer 包

你的 swift SFT env 用 `pip list` 看：
```
flash-attn                     2.8.3
flash-linear-attention         0.4.x
# 没有 flashinfer
```

这是 **swift 框架不需要 flashinfer**，因为：
- swift SFT 直接用 HuggingFace transformers 的 modeling 代码
- transformers 里 Qwen3.5 的 GDN 层调用 **FLA**（默认）或退回原生实现
- **从来不调 FlashInfer 的 `chunk_gated_delta_rule`**
- 自然不会撞 mbarrier 死锁

你的 verl_rl_v2 env 里有 flashinfer，因为：
- verl 装了 vllm（pip install vllm 把 flashinfer 拉进来了）
- vllm 在 rollout 时**默认用 FlashInfer 的 GDN kernel**
- → 命中 bug

### 验证 vLLM 是不是真用 FlashInfer

下次启动 v3 脚本（**未加 triton 配置时**），看 log：
```bash
grep -i "Using FlashInfer GDN prefill" ${CKPTS_DIR}/logs/run_*.log
```

如果出现，说明 vLLM 确实在用 FlashInfer，确认默认行为。

切到 triton 之后应该变成：
```bash
grep -i "Using Triton/FLA GDN prefill" ${CKPTS_DIR}/logs/run_*.log
```

### 这给我们的启示

1. **SFT → RL 迁移时，bug 才暴露**——很多团队（包括你）SFT 阶段没事，
   切到 RL 一调多机就崩。**因为 RL 引入了新的依赖链（vLLM → FlashInfer → CUDA Graph）**。

2. **不能用 SFT 经验估计 RL 的稳定性**——SFT 没踩的雷不代表 RL 没有。
   每一层栈（HF transformers / Megatron / vLLM / FlashInfer / NCCL）都有自己的 corner case。

3. **如果只是想做推理（不训练）**——可以用 ms-swift 的 vllm_mode 起 vLLM，
   **同样会撞这个 bug**。修复方式一样：`--gdn-prefill-backend triton`。

---

## 六、相关 issue 索引

| Issue | 角色 |
|---|---|
| [flashinfer-ai/flashinfer#3329](https://github.com/flashinfer-ai/flashinfer/issues/3329) | **根因 issue**：cuda-gdb 抓到 mbarrier 死锁 + 给出 triton workaround |
| [modelscope/ms-swift#8506](https://github.com/modelscope/ms-swift/issues/8506) | py-spy 抓到 TP 内 desync（rank 1 在 GDN，其它在 all_reduce）|
| [verl-project/verl#5659](https://github.com/verl-project/verl/issues/5659) | verl + Qwen3.5 + Megatron 同款 bug，评论 9 @DBMing 推荐 triton |
| [vllm-project/vllm#41862](https://github.com/vllm-project/vllm/issues/41862) | Qwen3.5 EP=8 hybrid GDN deadlock，TP + `--gdn-prefill-backend triton` 100% 工作 |
| [vllm-project/vllm#35104](https://github.com/vllm-project/vllm/issues/35104) | V1 engine workers die after idle period，jsboige 区分两个 Bug A/B |
| [verl-project/verl#4709](https://github.com/verl-project/verl/issues/4709) | 多机 GRPO 训练随机 step hang（同一 bug 类） |
| [verl-project/verl#3873](https://github.com/verl-project/verl/issues/3873) | DAPO Megatron+SGLang 第二步 hang（NCCL timeout） |
| [NVIDIA-NeMo/RL#1846](https://github.com/NVIDIA-NeMo/RL/issues/1846) | NeMo-RL 也有"训练 4-6h 后 stall"，跨框架通病 |

---

## 七、关联的"防御性配置"（不一定相关，但有人加过）

下面这些不是这个 bug 的修复，但社区里有人推荐叠加。**根因清楚后这些都不必加**。

| 配置 | 跟 GDN bug 关系 | 该加吗 |
|---|---|---|
| `gdn_prefill_backend=triton` | **直接修** | **必加** |
| `enforce_eager=True` | 间接（关 CUDA Graph） | 可选 backup |
| `NCCL_NVLS_ENABLE=0` | 无关，修另一类 NCCL hang | 已加，保留 |
| `NCCL_ALGO=Ring` | 无关 | 不加 |
| `actor_rollout_ref.nccl_timeout=1800` | 无关 | 不加 |
| `fuse_allreduce_rms=False` | 间接 | 可选 |
| `NCCL_CUMEM_ENABLE=0` | 无关，且在某些环境会触发 IB EFAULT | **不要加** |

---

## 八、教训

1. **跨框架 issue 多看几个**——本 bug 在 verl/swift/NeMo-RL/OpenRLHF 都有报告，但根因 issue 在
   flashinfer 而不是任何一个 RL 框架。如果只搜 verl/vllm 找不到。

2. **看 py-spy 栈而不是只看 NCCL watchdog 报错**——watchdog 只告诉你哪个 op 卡了，不告诉你为什么。
   py-spy 才能告诉你"rank 1 卡在 kernel 里，rank 2-7 卡在 all_reduce 等它"这种 desync 证据。

3. **GDN kernel 不像 attention 那样简单**——它的 warp-specialized 实现引入了 mbarrier 同步，
   而 mbarrier mis-arrival 在某些 corner case 下会永久死锁。

4. **CUDA Graph replay 路径 ≠ 直接调用路径**——很多 bug 只在 graph replay 下触发，
   单元测试用直接调用查不出来。

5. **"调 NCCL timeout 让它别报错"是反指标**——死锁的等待时间是无穷大，
   调 600s → 7200s 唯一区别是浪费多 1.5 小时再崩。

---

*最后更新：2026-05-26*
*作者修复路径：v3 脚本里加 `+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton`，
其它防御性配置保持现状即可。*