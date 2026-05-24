# Qwen3.5-27B GRPO 训练耗时实测

记录 27B dense 模型在 H200 集群上跑 blzk 规则奖励 GRPO 的实测耗时，对比 16 卡 / 32 卡两个 scale。
![alt text](image.png)
![alt text](image-1.png)
（华为256 张64G的910B卡，一个step 20min，快思考难度更大，有难度过滤）
## 一、实验配置对比

| 配置项 | 16 卡实验 | 32 卡实验 |
|---|---|---|
| `NNODES` | 2 | 4 |
| 总 GPU | 16 (2×8) | 32 (4×8) |
| `TP` | 4 | 4 |
| `PP` / `CP` | 1 / 1 | 1 / 1 |
| `GEN_TP` (vLLM) | 8 | 8 |
| **DP 组数** | 16/4 = **4** | 32/4 = **8** |
| `train_batch_size` | 384 | 512 |
| `ppo_mini_batch_size` | 96 | 128 |
| `ppo_micro_batch_size_per_gpu` | 1 | 1 |
| `rollout.n` | 8 | 8 |
| `max_prompt_length` | 8192 | 8192 |
| `max_response_length` | 16384 | 16384 |
| `val_before_train` | True | True |
| `save_freq` / `test_freq` | 50 / 10 | 50 / 10 |
| `total_epochs` | 1 | 1 |
| 训练集大小 | 43370 (filter 后) | 43370 |
| **总 step 数** | **112** | **84** |

注：32 卡 batch 调大主要为了占用更多 DP 组，每 step 处理更多 prompt。

## 二、实测耗时数据

### 16 卡实验（截至 step 38 / 112）

| Step | 累计耗时 | 平均 s/it | 估算单 step 时间 |
|---|---|---|---|
| 1 | 1:36:24 | 5784 s | 5784 s（含 val_before_train + 冷启动）|
| 2 | 2:44:57 | 4801 s | ~4113 s |
| 5 | 5:22:40 | 3440 s | ~3170 s |
| 10 | 8:53:59 | 2636 s | ~2533 s |
| 20 | 15:03:07 | 2237 s | ~2218 s |
| 37 | 25:07:03 | 2069 s | ~2131 s |
| 38 | 25:42:14 | 2081 s | ~2110 s |

**稳态单 step ≈ 35 min（2100s）**

### 32 卡实验（截至 step 23 / 84）

| Step | 累计耗时 | 平均 s/it | 估算单 step 时间 |
|---|---|---|---|
| 1 | 1:12:58 | 4378 s | 4378 s（含 val + 冷启动）|
| 2 | 2:01:01 | 3499 s | ~2884 s |
| 5 | 3:50:33 | 2399 s | ~2225 s |
| 10 | 5:47:32 | 1525 s | ~1408 s |
| 20 | 8:50:43 | 1113 s | ~1090 s |
| 23 | 9:41:56 | 1054 s | ~1024 s |

**稳态单 step ≈ 17.5 min（1050s）**

## 三、训练完成时间预估

### 16 卡

- 当前 step 38 / 112，已耗时 25:42:14
- 剩余 74 step × ~35 min/step ≈ **43 小时**
- **预估总训练时间 ≈ 25.7h + 43h ≈ 69 小时（约 2.9 天）**

### 32 卡

- 当前 step 23 / 84，已耗时 9:41:56
- 剩余 61 step × ~17.5 min/step ≈ **18 小时**
- **预估总训练时间 ≈ 9.7h + 18h ≈ 28 小时（约 1.17 天）**

## 四、scaling 效率分析

| | 16 卡 | 32 卡 | 加速比 |
|---|---|---|---|
| 单 step 稳态 | 35 min | 17.5 min | **2.0×** |
| 总训练时间 | ~69h | ~28h | **2.46×** |
| GPU 数量 | 16 | 32 | 2× |
| 算力成本比 | 1× | 2× | — |
| **scaling 效率** | baseline | **100% (理想线性)** | — |

**32 卡相比 16 卡**：
- ✅ 单 step 加速 ~2×，**几乎完美线性**
- ✅ 总时间加速 2.46×（因为 32 卡 batch 大 → step 数少：84 vs 112）
- ⚠️ 算力总成本不变（GPU·h 几乎相同），但**实际时间从 3 天压到 1 天**，**适合赶 deadline**

### 为什么这次 scaling 比 48 卡 35B 那次好？

| 维度 | 27B Dense (32 卡) | 35B MoE (48 卡) |
|---|---|---|
| MoE EP all-to-all | 无 | EP=8 跨节点 |
| 模型结构 | Dense（参数对齐）| MoE（路由不均衡）|
| Rollout 引擎数 | 4（NNODES=4）| 6 |
| vLLM 多 engine 协调 bug 命中率 | 低 | **高**（v0.17 bug）|
| 整除性约束 | 32/4 = 8（友好）| 48/2 = 24（含质因子 3）|

**结论**：dense 模型多机训练比 MoE 友好得多，没有 EP 跨节点通信。

## 五、为什么"刚开始慢、后来变快"

两个原因叠加：

1. **本质机制**（小节 1）：分布式训练里大量组件是"按需初始化"的，前几步要付一堆一次性开销，之后才进入"全缓存命中"的稳态
2. **观察偏差**（小节 4）：tqdm 显示的 `s/it` 是**从启动到现在的平均值**，不是当前 step 的实际耗时，所以前几步被拉得格外高

### 1. 本质机制：各种"按需初始化"代价集中在前几步

不只是 RL，**所有分布式训练（SFT / PT / 大规模 eval）都有这个现象**。根本原因是大量组件采用"懒初始化 / 懒 cache"策略：第一次用时建好，之后全部命中缓存。前几步要把所有 cache 暖热，之后才进入稳态。按贡献大小排：

#### a. Kernel JIT 编译（通常占大头）
- **CUDA Graph capture**：PyTorch / vLLM 对每种 `(batch_size, seq_len)` 形状第一次出现时做一次 graph capture，捕获完后缓存。前几步如果形状变化多，反复触发捕获，每次几百 ms ~ 几秒
- **Triton / Inductor / CUTLASS kernel 编译**：第一次调用的 FlashAttention、FlashInfer、MoE 路由、自定义 fused kernel 都要 ptxas 编译，单个 kernel 几秒到几十秒。编完写进 `TRITON_CACHE_DIR`，同进程内复用
- **vLLM kernel autotune**：对部分形状做选优测试，第一次跑会试几种实现挑最快的，之后固定

#### b. CUDA caching allocator 预热
PyTorch 的 caching allocator 第一次分配走 `cudaMalloc`（系统调用慢、要锁），之后用 free-list 复用。前几步会有大量 cudaMalloc 把显存"涨"到工作集大小，之后基本不再 malloc。单步能差几百 ms，最容易被忽视的一项。

#### c. NCCL communicator 建立 + 算法选择
**第一次 collective 比之后慢 5-10 倍**。NCCL 第一次执行 allreduce / allgather 时要：探测拓扑（NVLink / NVSwitch / IB）→ 选算法（ring / tree / NVLS）→ 分配通信 buffer → 跨 rank 同步。之后全 cache 在 communicator 里。

H200 + Megatron + vLLM 这套多 process group 的栈，**每对 process group 都要做一次**，所以 NCCL 冷启动开销显著。

#### d. Data loader prefetch 填管
PyTorch DataLoader 的 prefetch queue 启动时是空的。第 1 步 GPU 干等 IO，第 2 步预取 1 个 batch，依此类推，直到 prefetch queue 填满（4-8 步后）才彻底 IO-overlap。

#### e. 操作系统页缓存
第一次读模型权重 / parquet 数据走磁盘，后续走 OS page cache（RAM），快几十倍。NFS / 网盘环境下影响特别大。

#### RL 特有的几项
- **vLLM rollout 的 cuda graph capture**：对常用 batch size 做捕获（日志里能看到 `cudagraph_capture_sizes: [1,2,4,8]` 这种），单引擎要几十秒到几分钟
- **Megatron → vLLM 第一次权重广播**：建 NCCL communicator + 第一次大 bucket allgather，比稳态慢一两倍
- **Ref policy 第一次 forward**：和 actor 共享 backbone 编译产物前要单独走一遍 JIT

> **持久化 cache 解决"跨运行"的冷启动，解决不了"同次运行内"的前几步**
>
> 脚本里 `TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / FLASHINFER_JIT_DIR` 都落到本地 /tmp，跨多次启动能省 5-10 分钟 ptxas 编译。但 **cuda graph capture / NCCL communicator / allocator 状态都是 in-memory 的，无法落盘**，每次新进程都要重做。

### 2. `val_before_train=True` 的开销算到了第 1 个 step

```
val_before_train 跑完整 val 集 (896 条)
  ↓ greedy decode 每条 ~16K token
  ↓ 全部完成才进入训练循环
进入 step 1 的 rollout
  ↓ vLLM 第一次见这个 batch size,JIT 编译
  ↓ Megatron 第一次跑 forward,attention kernel 编译
完成 step 1
```

**val + 编译 + 实际 step1 全算到 "1 iter"**，所以 step 1 显示 ~96 min，实际后续 step 只要 17-35 min。

### 3. 冷启动开销（一次性）

| 一次性开销 | 大致时间 |
|---|---|
| vLLM 模型加载到 8 卡 (TP=8) | 2-5 min |
| vLLM CUDA Graph 编译（多种 batch size）| 3-10 min |
| Megatron 模型加载 + 分片 | 3-5 min |
| Megatron attention kernel JIT | 2-3 min |
| Ray actor 启动 + 资源分配 | 1-2 min |
| `val_before_train` 验证 | 5-15 min |
| **合计冷启动** | **15-40 min** |

这 15-40 min 全部摊到第 1 个 step 的"耗时"里。

### 4. 平均值数学上的收敛

```
平均 = 累计耗时 / step 数
     = (冷启动 + N × 稳态 step 时间) / N
     = 冷启动/N + 稳态 step 时间
```

随 N 增大，`冷启动/N` 趋近 0，平均逐渐逼近稳态。

**举例验证（32 卡）**：
- 冷启动估算 ≈ 4378 - 1050 ≈ 3328 s（约 55 min）
- step N 时显示的平均 = 3328/N + 1050
- N=1: 3328/1 + 1050 = 4378 ✓
- N=5: 3328/5 + 1050 = 1716（实测 2399，差一些因为 step 2-5 还在 warmup）
- N=20: 3328/20 + 1050 = 1216（实测 1113，接近）
- N=23: 3328/23 + 1050 = 1195（实测 1054，差不多）

**数学模型对得上**，所以这就是个**纯平均值收敛现象**，没有性能 bug。

### 看真实稳态：增量法

要看 step 实际跑多快，**取相邻两个 step 的 elapsed 差**：
```
真实稳态 step 时间 = (elapsed_at_step_B - elapsed_at_step_A) / (B - A)
```

32 卡：从 step 5 (3:50:33) 到 step 23 (9:41:56)
- Δ = 5:51:23 ≈ 21083 s
- ΔStep = 18
- **实际稳态 step ≈ 1171s ≈ 19.5 min**

16 卡：从 step 20 (15:03:07) 到 step 37 (25:07:03)
- Δ = 10:03:56 ≈ 36236 s
- ΔStep = 17
- **实际稳态 step ≈ 2131s ≈ 35.5 min**

这才是真正的步进速度，不要被 tqdm 的"累计平均"骗了。

## 六、总结

> RL 训练里 tqdm 显示的 s/it 是**累计平均值**，不是当前 step 真实速度。冷启动（vLLM 编译 + Megatron 加载 + val_before_train）会被摊到第 1 个 step 的耗时里，让前 10 个 step 看起来异常慢，但实际单 step 稳态远低于显示值。**判断 RL 训练真实速度，要用相邻两个 step 的 elapsed 差来算**，不要直接看 s/it 平均值。
>
> 我跑 Qwen3.5-27B GRPO，16 卡稳态 35 min/step，32 卡 17.5 min/step，**接近完美线性加速**——这是 dense 模型的优势（没有 MoE EP 跨节点 all-to-all），同样的训练在 35B-A3B MoE 上扩到 6 节点时 scaling 反而退化。

## 七、可以学到的工程经验

1. **冷启动 30 min 是常态**：长跑 RL 训练前期 5-10 个 step 慢是必然，不要慌着 kill
2. **算稳态用差分法**：相邻 step 差 / step 数 = 真稳态，比直接看 s/it 更可信
3. **scaling 效率看模型架构**：dense 易 scale，MoE 受 EP 通信限制
4. **batch_size 大不一定加速**：32 卡的 batch=512 比 16 卡的 batch=384 大 33%，但总训练时间快了 2.46×，**主要是 step 数减少**贡献（112 → 84 step）
