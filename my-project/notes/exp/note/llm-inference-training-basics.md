# LLM 推理与训练基础知识
https://zhuanlan.zhihu.com/p/1999089391205897472
## 一、注意力机制与 KV Cache

### QKV 是什么？

Transformer 每层的注意力计算中，每个 token 会生成三个向量：

- **Q（Query）**：当前 token "想找什么"
- **K（Key）**：每个历史 token "我是什么"
- **V（Value）**：每个历史 token "我的内容是什么"

计算公式：

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

当前 token 的 Q 与所有历史 token 的 K 做点积，得到注意力权重，再加权求和 V，得到输出。

### KV Cache 的作用

**问题**：自回归生成时，生成第 n 个 token 需要用到前 n-1 个 token 的 K、V。如果每步都重新计算，代价极高。

**解决**：把已算过的 K、V 缓存起来，每步只计算新 token 的 K、V 追加到 cache。

```
第1步: 算 K₁,V₁ → 存入 cache
第2步: 算 K₂,V₂ → 存入 cache，直接读 K₁,V₁
第n步: 只算 Kₙ,Vₙ，历史全部从 cache 读取 ✓
```

**代价**：KV cache 占大量显存，序列越长占用越多。这是推理显存的主要消耗，也是 `gpu_memory_utilization` 参数控制的核心。

---

## 二、推理的两个阶段

### 阶段一：Prefill（预填充）

处理用户输入的整个 prompt，一次性并行计算所有 prompt token 的注意力，建立 KV cache，并预测第一个生成 token。

```
输入: [tok₁][tok₂]...[tokₙ]  (整个 prompt)
         ↓ 并行 attention（所有 token 同时计算）
KV cache: K₁V₁, K₂V₂, ..., KₙVₙ  ← 全部存好
输出: 第一个生成 token
```

- 特点：**计算密集**，GPU 利用率高
- 时间复杂度：O(n²)，prompt 越长越慢

### 阶段二：Decode（解码）

有了 KV cache 后，逐个生成新 token，每步只计算当前新 token 的注意力。

```
第n+1步: 用新 token 的 Q，与 cache 中所有 K 做 attention → 生成下一个 token
第n+2步: 同上，cache 增长一个 ...
```

- 特点：**串行**，每步只算一个 token，GPU 计算单元大量闲置
- 瓶颈：显存带宽（每步要读完整 KV cache）

### 整体流程

```
用户 prompt (N tokens)
      │
      ▼
 ┌─────────┐
 │ Prefill │  → 并行处理所有 prompt token，建立 KV cache
 └────┬────┘
      │
      ▼
 ┌─────────┐
 │ Decode  │  → 逐 token 串行生成，直到 <EOS>
 └─────────┘
```

---

## 三、vLLM 的关键优化参数

### Chunked Prefill（分块预填充）

**问题**：长 prompt 的 Prefill 会完全阻塞正在 Decode 的其他请求。

**解决**：把 Prefill 切成 chunk，每次 forward 只处理 ≤ `max_num_batched_tokens` 个 token，Prefill chunk 和 Decode token 交错执行。

```
没有 chunked prefill：
[====长 Prefill A（阻塞）====][Decode-A][Decode-A]...
                              [====Prefill B====][Decode-B]...

有 chunked prefill（max_num_batched_tokens=4096）：
[Prefill-A chunk1 + Decode-B] → 1次 forward
[Prefill-A chunk2 + Decode-B] → 1次 forward
[Prefill-A chunk3 + Decode-B] → 1次 forward
[Decode-A + Decode-B]         → 1次 forward
```

verl 配置：`actor_rollout_ref.rollout.enable_chunked_prefill=True`（默认已开启）

### Prefix Caching（前缀缓存）

如果多个 prompt 共享相同前缀（如相同 system prompt），只计算一次这段前缀的 KV，后续所有请求共享这份 cache。

**在 RL 训练中的问题**：每次策略更新后，旧的 KV cache 是用旧模型权重算的，对新模型无效（stale cache）。

**verl 的处理方式**：

| 模式 | 如何处理 stale cache |
|------|---------------------|
| FSDP（COLOCATED 模式） | `wake_up()` 时自动调 `reset_prefix_cache()`，显式设 `enable_prefix_caching=False` 是保守写法 |
| Megatron（HYBRID 模式） | `update_weights` 时通过 `abort_all_requests(reset_prefix_cache=True)` 清理；async 流水线内 cache 有效，保持默认 `True` |

### free_cache_engine

rollout 结束后释放 vLLM 的 KV cache GPU 内存，让训练阶段可以使用这部分显存。

verl 默认值：`True`。共卡训练必须保持 True。

---

## 四、训练后端对比：FSDP vs Megatron vs DeepSpeed

### FSDP（Fully Sharded Data Parallel）

**原理**：把模型参数、梯度、优化器状态分片存储在各 GPU，forward 时临时 All-Gather 拼出完整参数。

```
平时: GPU0:[shard-0]  GPU1:[shard-1]  GPU2:[shard-2]  GPU3:[shard-3]
           ↓ All-Gather（forward 前）
forward: 每张卡都临时持有完整模型参数 → 计算 → 释放
```

- 并行维度：数据并行（+ 可选 Ulysses 序列并行）
- 峰值显存：完整模型大小（All-Gather 时）
- 实现：PyTorch 原生（FSDP1 / FSDP2）
- 适用规模：7B~30B，单机 / 少节点
- 配置简单：主要参数只有 `fsdp_size`
- verl 配置关键字：`actor_rollout_ref.actor.strategy=fsdp2`

### Megatron-LM

**原理**：把每层的权重矩阵切开分配到不同 GPU，每张卡只负责计算矩阵的一部分，结果 All-Reduce 合并。

```
Linear 层 weight [4096 × 16384]，TP=4：
GPU0: [4096×4096]  GPU1: [4096×4096]  GPU2: [4096×4096]  GPU3: [4096×4096]
各自计算 → All-Reduce → 合并得完整输出
```

支持多维并行：
- **TP（张量并行）**：切权重矩阵，需要 NVLink 高速互联
- **PP（流水线并行）**：不同 Transformer 层分配到不同卡
- **EP（专家并行）**：MoE 模型，不同专家在不同卡
- **CP（上下文并行）**：超长序列切分并行

- 峰值显存：1/TP 模型大小（forward 时无需 All-Gather）
- 适用规模：30B+，多机多卡
- 配置复杂：TP/PP/EP/CP 需要仔细规划整除关系
- verl 配置关键字：`actor_rollout_ref.actor.megatron.*`

### DeepSpeed（ZeRO）

**原理**：与 FSDP 类似，也是分片策略，但实现更早，功能更丰富。

ZeRO 三个阶段：
- **ZeRO-1**：只分片优化器状态
- **ZeRO-2**：分片优化器状态 + 梯度
- **ZeRO-3**：分片优化器状态 + 梯度 + 参数（等同于 FSDP）

额外特性：
- **ZeRO-Offload**：把优化器状态/梯度 offload 到 CPU/NVMe
- **ZeRO-Infinity**：更激进的 NVMe offload
- 集成 pipeline parallelism（但不如 Megatron 成熟）

> verl 目前主要支持 FSDP 和 Megatron，不原生支持 DeepSpeed 作为训练后端。

### 三者横向对比

| 维度 | FSDP | Megatron | DeepSpeed ZeRO-3 |
|------|------|----------|------------------|
| 并行类型 | 数据并行（分片） | TP/PP/EP/CP | 数据并行（分片） |
| forward 峰值内存 | 完整模型 | 1/TP 模型 | 完整模型 |
| 实现复杂度 | 低 | 高 | 中 |
| 适用规模 | ~30B 内 | 30B+ | ~30B 内 |
| 多机扩展性 | 一般 | 优秀 | 较好 |
| verl 支持 | ✓（原生） | ✓（原生） | ✗ |
| 权重同步到 vLLM | 直接（HF 格式） | 需 mbridge 转换 | — |

---

## 五、verl 中的 Rollout 模式

rollout（用当前策略生成回复）和 training（更新策略）在 GPU 资源上有三种关系：

### COLOCATED 模式（共卡串行）

FSDP 脚本使用。rollout 和 training 共用同一组 GPU，交替执行。

```
时间轴：
[──Rollout N──][──Train N──][──Rollout N+1──][──Train N+1──]

rollout 时：vLLM 占用显存，训练参数 offload 到 CPU
training 时：vLLM sleep（释放 KV cache），FSDP All-Gather 参数训练
```

wake_up 时自动调用 `reset_prefix_cache()`，清除过期 KV cache。

### HYBRID 模式（共卡异步流水线）

Megatron 脚本使用。rollout(N+1) 和 train(N) 重叠执行，吞吐更高。

```
时间轴：
[──Rollout N──────────]
              [──Train N──────────]
                          [──Rollout N+1──]
                                      [──Train N+1──]
```

可行的原因：Megatron TP 使得每张卡 forward 时只需 1/TP 参数，不需要 All-Gather 完整模型，与 vLLM 的显存不冲突。

### STANDALONE 模式（分卡）

rollout 和 training 使用不同的 GPU 组，完全独立，适合超大规模训练。

---

## 六、verl 配置参数速查

### Rollout 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `rollout.mode` | `async` | vLLM 引擎模式（sync=LLM，async=AsyncLLM） |
| `rollout.gpu_memory_utilization` | `0.5` | KV cache 占显存比例 |
| `rollout.free_cache_engine` | `True` | rollout 后释放 KV cache |
| `rollout.enable_prefix_caching` | `True` | 前缀 KV cache 复用 |
| `rollout.enable_chunked_prefill` | `True` | 分块 Prefill |
| `rollout.max_num_batched_tokens` | `8192` | 每次 forward 最大 token 数 |
| `rollout.enforce_eager` | `False` | 禁用 CUDA graph（False=启用 graph，更快） |
| `rollout.ignore_eos` | `False` | 是否忽略 EOS 继续生成 |
| `rollout.tensor_model_parallel_size` | `2` | vLLM 的 TP 大小 |
| `rollout.n` | `1` | 每个 prompt 采样几条回复（GRPO 通常设 5~8） |

### FSDP 训练关键参数

| 参数 | 含义 |
|------|------|
| `actor.strategy` | `fsdp2`：使用 PyTorch FSDP2 |
| `actor.fsdp_config.fsdp_size` | 参与分片的 GPU 数量 |
| `actor.fsdp_config.param_offload` | 参数 offload 到 CPU |
| `actor.fsdp_config.optimizer_offload` | 优化器状态 offload 到 CPU |
| `actor.fsdp_config.ulysses_sequence_parallel_size` | Ulysses 序列并行度 |

### Megatron 训练关键参数

| 参数 | 含义 |
|------|------|
| `actor.megatron.tensor_model_parallel_size` | TP 大小，切权重矩阵 |
| `actor.megatron.pipeline_model_parallel_size` | PP 大小，切 Transformer 层 |
| `actor.megatron.expert_model_parallel_size` | EP 大小，MoE 专用 |
| `actor.megatron.param_offload` | 参数 offload 到 CPU |
| `actor.megatron.optimizer_offload` | 优化器 offload 到 CPU |
| `actor.megatron.grad_offload` | 梯度 offload 到 CPU |
| `actor.megatron.use_mbridge` | 启用 mbridge 做 Megatron→vLLM 权重格式转换 |
