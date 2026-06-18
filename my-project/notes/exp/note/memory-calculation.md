# 大模型显存计算公式与训练 / 推理对比

整理 LLM 训练（Megatron）和推理（vLLM）下的显存构成、计算公式和对比。配 27B / 35B 的具体数字方便快速估算。

## 一、统一的"显存五大类"

任何阶段的 GPU 显存占用都可以拆成这 5 块，看哪些类别**有/无**就是训推差异：

| 类别 | 训练 | 推理 | 备注 |
|---|---|---|---|
| ① **模型权重 (params)** | ✓ | ✓ | bf16 = 2 bytes/param |
| ② **梯度 (grad)** | ✓ | ✗ | 大小同 params |
| ③ **优化器状态 (optimizer)** | ✓ | ✗ | Adam: 8 bytes/param 起 |
| ④ **激活 (activation)** | ✓ 大量保留 | ✓ 仅瞬时 | 训练保留给反向用 |
| ⑤ **KV cache** | ✗ | ✓ | 推理特有，按并发增长 |
|   **CUDA workspace** | ✓ | ✓ | 2-3 GB 固定开销 |

**核心差异**：训练靠 ②③④ 占大头，推理靠 ⑤ 占大头。

## 二、单位换算速查

```
1 B (param) × bf16 (2 bytes) = 2 GB / B-param
1 B (param) × fp32 (4 bytes) = 4 GB / B-param
1 B (param) × int8           = 1 GB / B-param
1 B (param) × int4           = 0.5 GB / B-param

记忆口诀: bf16 模型大小 (GB) ≈ 参数量 (B) × 2
```

示例：
- 7B bf16 ≈ 14 GB
- 14B bf16 ≈ 28 GB
- 27B bf16 ≈ 54 GB
- 35B bf16 ≈ 70 GB
- 70B bf16 ≈ 140 GB
- 671B bf16 ≈ 1.3 TB

## 三、训练显存逐项公式（Megatron）

### ① 模型权重

```
权重显存 = N_params × bytes_per_param / (TP × PP × CP)
```

EP 不在分母（MoE 的 expert 参数只摊到 EP 维度，attention/embed 不切）。

### ② 梯度

```
梯度显存 ≈ 权重显存
```
（bf16 训练时梯度也是 bf16）

### ③ 优化器状态（Adam 混合精度）

```
Adam fp32 master weights : N × 4 bytes / DP
+ momentum (m)            : N × 4 bytes / DP
+ variance (v)            : N × 4 bytes / DP
─────────────────────────────────────
合计                      : N × 12 bytes / DP
```

如果用 **fp32 全精度 Adam**（不带 master）：8 bytes / param
如果用 **bf16 Adam**：4 bytes / param

### ④ 激活（最难估，是大头）

```
activation per layer ≈ batch × seq × hidden × (具体多少倍取决于 attention 实现)
```

经验值（Megatron + flash attention，BSHD）：
```
activation 总 ≈ batch × seq × hidden × num_layers × 10-30 bytes
```

**recompute** 能砍掉大部分：`recompute_granularity=full` 把激活降到 ~1/3 ~ 1/5。

### ⑤ + CUDA workspace

固定 2-3 GB，包括 CUDA context、cuDNN、NCCL、PyTorch caching allocator 等。

### 训练总显存（单卡，无 offload）

```
单卡显存 = (权重 + 梯度 + 优化器) / (TP×PP×CP×DP 视情况)
        + 激活（按 micro_batch_size × seq_len 算）
        + workspace 2-3 GB
```

### Offload 的效果

```
param_offload=True    : 权重丢 CPU,GPU 只在 forward 时拉回
optimizer_offload=True: 优化器状态丢 CPU
grad_offload=True     : 梯度算完丢 CPU
```

全开 offload 后 GPU 实占降到 **激活 + workspace ≈ 10-50 GB / 卡**，**用 CPU 内存 / NVMe 换 GPU 显存**。

## 四、推理显存逐项公式（vLLM）

### ① 模型权重

```
权重显存 = N_params × bytes_per_param / TP
        × 1.1-1.2 (CUDA workspace / checkpoint engine overhead)
```

vLLM 不需要 PP/CP/EP 分母（用 TP-only 推理）。

### ② KV Cache（**推理大头**）

**单 token 单 sample KV cache**：
```
KV cache per token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
                    ↑ K 和 V 各一份
```

注意是 `num_kv_heads`（GQA 共享后的数量），不是 `num_attention_heads`。

**完整序列 KV cache**：
```
KV cache per sequence = KV per token × max_seq_len
```

**全 batch KV cache**：
```
total KV cache = KV per sequence × batch_size
```

TP=N 切分后：每卡 `total / N`。

### ③ vLLM 的预留机制

vLLM 启动时按 `gpu_memory_utilization` **预先抢一块固定大小的显存**，分给 KV cache 池子：

```
vLLM 总预留 = gpu_memory_utilization × 单卡总显存
KV cache 预算 = vLLM 总预留 - 模型权重 - CUDA workspace
```

**关键**：调大 `gpu_memory_utilization` = 更大 KV cache 池 = 更多并发 = 更快 throughput。

### ④ 活动激活（瞬时）

推理时不保留 activation 做反向，只在 forward 时短暂占用：
- **Prefill 阶段**：3-8 GB / 卡（看 prompt 长度）
- **Decode 阶段**：1-2 GB / 卡（每步只算 1 个新 token）

完成后立即释放。

### 推理总显存（单卡）

```
单卡显存 = 模型权重 / TP + workspace + KV cache 预算
        ≤ gpu_memory_utilization × 单卡总显存
```

## 五、训练 vs 推理对比表（核心）

| 维度 | 训练 (Megatron) | 推理 (vLLM) |
|---|---|---|
| 主导显存占用 | 优化器 + 激活 | KV cache |
| 模型权重存在 | ✓ | ✓ |
| 梯度存在 | ✓（同权重大小）| ✗ |
| 优化器状态 | ✓（最大头，Adam 12 bytes/param）| ✗ |
| 激活保留 | ✓（巨大，按 batch×seq 算）| ✗（只瞬时）|
| KV cache | ✗ | ✓（动态分配，占预留 80%+）|
| 并行维度 | TP × PP × CP × DP × EP | 只用 TP |
| 显存优化主战场 | offload / recompute / ZeRO | gpu_memory_utilization / KV 量化 / Prefix cache |
| 调优目标 | 让大模型能塞进卡 | 让并发尽量大 |

## 六、应用：Qwen3.5-27B 显存帐本

### 训练阶段（你脚本 TP=4 + ALL_OFFLOAD=True + H200 141GB）

```
权重 (bf16)        : 27 × 2 / TP=4 = 13.5 GB → offload 到 CPU
梯度 (bf16)         : 13.5 GB → offload
优化器 (Adam fp32)   : 27 × 12 / DP 分摊 → 全部 offload
                     32 卡: DP=8, → 单卡 40 GB → offload
激活 (per micro=1) : seq=24576 × hidden=5120 × layers=64 × ~20 bytes ≈ 160 GB
                     ÷ TP=4 = 40 GB / 卡(无 recompute)
                     × recompute_full(~1/4) ≈ 10 GB / 卡
临时 buffer + workspace : 5-10 GB
────────────────────────────────────────
GPU 实占                : 15-25 GB / 卡（offload 把大头丢 CPU,recompute 砍激活）
```

### 推理阶段（你脚本 GEN_TP=8 + gpu_memory_utilization=0.6）

```
预留总量              : 141 × 0.6 = 84.6 GB / 卡
─────────────────────────────────────────
权重 (bf16, TP=8)     : 27 × 2 / 8 × 1.2 ≈ 8 GB
CUDA + workspace      : 2-3 GB
KV cache 可用预算     : 84.6 - 8 - 3 ≈ 73 GB / 卡 ← 大头!
活动激活 (瞬时)        : <5 GB

KV cache 容量（按 24K 序列、bf16、GQA 8 kv-heads 估算）:
  per token : 2 × 64 layers × 8 heads × 128 dim × 2 = 256 KB
  per seq   : 256 KB × 24576 ≈ 6 GB
  TP=8 后    : 6/8 = 0.75 GB / 卡 / 24K 全长序列
  73 GB 能容: 73/0.75 ≈ 97 条满长度序列
```

## 七、应用：Qwen3.5-35B-A3B MoE 显存帐本

### 训练阶段（TP=2 + EP=8 + ALL_OFFLOAD=True）

```
MoE 权重 (bf16)      : 35 × 2 = 70 GB 总
                      Expert 部分按 EP=8 切: dense 部分 / TP=2, expert / EP=8
                      → 单卡约 12-15 GB → offload
梯度 + 优化器          : 全部 offload 到 CPU
激活                  : MoE 多了一层 router 输出,但只激活 3B
                      → ~12-20 GB / 卡 (recompute 后)
────────────────────────────────────────
GPU 实占               : 20-35 GB / 卡
```

### 推理阶段（GEN_TP=8）

```
权重 (35B 全量,bf16,TP=8) : 35 × 2 / 8 × 1.2 ≈ 10.5 GB
   注意: vLLM 推理时 MoE 模型按总参数算,不是激活参数
KV cache 可用预算          : 73 GB / 卡（gpu_mem_util=0.5,实际你脚本是 0.5）
活动激活                   : ~5 GB
```

## 八、面试常见数学题

### Q: Llama-3-70B 单卡能不能跑推理？

```
权重 bf16 = 70 × 2 = 140 GB
单卡 H200 = 141 GB
→ 不够,必须 TP≥2
TP=2 时: 70/2 = 35 GB + workspace + KV cache → 一张 H200 跑半 model OK
```

### Q: 7B bf16 训练单卡需要多少显存（不 offload, Adam）？

```
权重:     7  × 2 = 14 GB
梯度:     7  × 2 = 14 GB
优化器:    7  × 12 = 84 GB（Adam fp32）
激活:     ~20 GB (batch=1, seq=2K, recompute)
workspace: 3 GB
─────────────────────
合计:      ~135 GB
```
所以 7B 训练单卡 80GB A100 跑不动，必须 offload 或 ZeRO。**这就是为什么 7B 也要多卡 / FSDP**。

### Q: KV cache 决定并发能力，怎么估算？

```
Llama-3-8B (32 层, 32 KV-head, head_dim=128):
  per token: 2 × 32 × 32 × 128 × 2 = 524 KB / token
  per 4K seq: 524 × 4096 / 1024 = ~2.1 GB / 序列

单卡 80GB,权重 16GB,workspace 3GB
KV cache 预算 ≈ 80 × 0.9 - 16 - 3 = 53 GB
可容并发: 53 / 2.1 ≈ 25 条 4K 序列
```

## 九、调优套路

### 训练侧

| 问题 | 解法 |
|---|---|
| 权重塞不下 | TP↑ / PP↑（多卡分摊）|
| 优化器塞不下 | optimizer_offload=True |
| 梯度塞不下 | grad_offload=True |
| 激活塞不下 | recompute_granularity=full（用算力换显存）|
| 整体紧张 | ZeRO-3 / FSDP / 所有 offload 全开 |

### 推理侧

| 问题 | 解法 |
|---|---|
| 模型本身塞不下 | TP↑ |
| KV cache 不够 → 排队 | gpu_memory_utilization↑（0.5 → 0.8）|
| 长序列吃 KV | KV 量化（int8 / fp8）|
| 同 prompt 多 response | enable_prefix_caching=True |
| 高并发 | chunked_prefill + 大 max_num_batched_tokens |

## 十、面试金句

> 显存按 5 大类拆解：**模型权重 / 梯度 / 优化器 / 激活 / KV cache**。训练独占的是梯度 + 优化器 + 保留激活，推理独占的是 KV cache。**训练里 Adam 优化器是隐藏大头（12 bytes/param，比权重大 6 倍）**，推理里 KV cache 是大头（vLLM 用 gpu_memory_utilization 提前抢一块给它）。
>
> 推理侧反直觉的点：**模型权重只占小头**，KV cache 决定并发能力。所以 vLLM 调优 ROI 最高的不是换模型 / 换 TP，而是调 `gpu_memory_utilization`——0.5 调到 0.8 通常让 throughput 涨 30%+。
>
> MoE 模型的特别处：**推理显存按总参数算**（vLLM 不知道哪些 expert 会被激活，要把所有 expert 都加载到 GPU），**训练用 EP 切 expert 才能省**。这就是为什么 MoE "推理友好"是有条件的——只有训练侧用 expert parallel 才能真正省显存。
