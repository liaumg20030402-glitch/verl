# AllReduce / AllToAll 与 MoE 训练（面试复习向）

围绕分布式训练里的两个核心通信原语，以及 MoE 模型相对 Dense 模型在架构 / 训练 / 推理上的差异展开。

---

# 一、AllReduce 是什么

## 1.1 定义

**AllReduce**：每个 rank 持有一个张量 `T_i`，集合操作完成后，**所有 rank 都拥有同一个聚合结果**：
```
result = reduce_op(T_0, T_1, ..., T_{N-1})    # reduce_op 通常是 sum / mean / max
```
每个 rank 拿到的最终值 **完全一致**。

**直观例子**：8 张卡各算了一份梯度，AllReduce 之后，每张卡都拿到 8 张卡梯度的求和。

## 1.2 算法实现：Ring AllReduce

经典实现是 NCCL 的 **ring all-reduce**：

```
Step 1: reduce-scatter
    8 张卡形成一个环 (0→1→2→...→7→0)
    把张量切成 8 份，每张卡轮流转发并累加自己负责的那一份
    最终: 每张卡持有 1/8 的完整 sum

Step 2: all-gather
    再绕一圈，每张卡把自己持有的 1/8 sum 广播给所有人
    最终: 每张卡持有完整的 sum
```

**通信量**：每张卡发送 `2 × T × (N-1) / N`（T 是张量大小，N 是 rank 数）—— **几乎和 rank 数无关**，这就是 ring all-reduce 的精妙之处。

## 1.3 实际用途

- **数据并行 (DP) 同步梯度**：每个 step backward 后，DP 组内 all-reduce 梯度
- **TP 列切分时的输出归约**：`Y = X·W` 中 W 按列切，每个 rank 算 partial Y，最后 all-reduce 拼成完整 Y
- **统计指标聚合**：loss、accuracy 这种标量也是 all-reduce

## 1.4 性能考量

- **带宽-bound**：受限于 NIC 带宽（NVLink、InfiniBand）
- **延迟-bound**（小张量时）：每个环节有 startup 开销
- **跨节点通信**比节点内通信慢 1 个数量级（IB ~50GB/s vs NVLink ~300GB/s）

## 1.5 面试金句

> Ring AllReduce 的核心创新是**通信量与 rank 数解耦**，每张卡的发送量恒定在 `2T(N-1)/N ≈ 2T`，让 DP 能 scale 到几百张卡。但这一切的前提是**带宽**够，跨节点 IB 链路一旦拥塞或某卡慢，整个 ring 就被最慢的拖住 —— 这也是为什么大模型训练要重金堆 NVLink + IB。

---

# 二、AllToAll 是什么

## 2.1 定义

**AllToAll**：每个 rank 持有一个 N 等分的张量 `[T_i,0, T_i,1, ..., T_i,N-1]`，集合操作完成后，**rank i 拿到所有人发给 i 的那块**：
```
rank_i 之前: [T_i,0, T_i,1, ..., T_i,N-1]   # 它给所有人准备的
rank_i 之后: [T_0,i, T_1,i, ..., T_N-1,i]   # 所有人给它的
```
**转置语义**：把发送矩阵的行变成列。

## 2.2 直观例子

8 张卡跑 MoE，每张卡有 1024 个 token、8 个 expert（每张卡持有 1 个 expert）。每个 token 被路由到某个 expert。

AllToAll 前：每张卡持有 1024 个 token + 它们各自要去的 expert id  
AllToAll 后：每张卡持有"应该被自己这个 expert 处理"的所有 token（来自全部 8 张卡）

## 2.3 实际用途

- **MoE 的 Expert Parallel (EP)**：dispatch（把 token 发到对应 expert）+ combine（把 expert 输出送回原 token 位置）
- **TP 中的某些算子**：activation 的重排
- **稀疏 attention**：把不同 head/序列段的数据重新分发

## 2.4 性能考量 vs AllReduce

| | AllReduce | AllToAll |
|---|---|---|
| 通信量（每卡） | `2T(N-1)/N ≈ 2T` | `T(N-1)/N ≈ T` |
| 通信复杂度 | O(T) | O(T) 但**实际更慢** |
| 通信模式 | 环状，链式接力 | **全连接** N×N 对点对点 |
| 网络拓扑要求 | 友好（链式即可） | **严苛**（任意两 rank 直连最快） |
| 跨节点表现 | 较好 | **明显恶化**（多对多冲突） |

**AllToAll 在跨节点时性能掉得很厉害**：N 个 rank 之间 N² 条流同时挤同一根 IB 链路，带宽利用率远低于 ring 拓扑。

## 2.5 面试金句

> AllReduce 和 AllToAll 通信量量级相同，但实际开销 AllToAll 大得多 —— AllReduce 是接力式（每时刻只有 N 条流），AllToAll 是全对全（N² 条流挤同一链路）。所以 **MoE 的 EP 跨节点扩展性远不如 DP 的 AllReduce**，DeepSeek-V3 / Qwen3.5 这类大 MoE 模型在多机训练时通信经常是瓶颈。

---

# 三、Dense 模型 vs MoE 模型

## 3.1 架构差异

### Dense FFN（传统 Transformer）

每个 transformer block 里 FFN：
```python
y = W2(GELU(W1(x)))    # 所有 token 都过同一组 W1, W2
```
- 参数：`W1` + `W2`，比如 hidden=4096, intermediate=11008，参数 ~90M / FFN
- 每个 token 用**所有参数**做计算

### MoE FFN（混合专家）

每个 transformer block 里 FFN 替换成：
```python
expert_indices = TopK(Router(x), k=2)      # 每个 token 选 top-k 个 expert
expert_weights = Softmax(Router(x)[topk])  # 路由权重
y = sum(weight_i * Expert_i(x) for i in expert_indices)
```
- 参数：`E` 个 expert（比如 128 个），每个 expert 是一个小 FFN
- 每个 token 只用 `k=2` 个 expert（**激活参数 ≪ 总参数**）

### Qwen3.5-35B-A3B 具体数字

> "35B-A3B" 的命名：**35B 总参 / A3B 激活参数 ≈ 3B**

- 总参数：35B（保留全部 expert 的权重）
- 激活参数：~3B（每 token 只走 2 个 expert）
- 推理 FLOPs 约等于 3B dense 模型
- 显存（KV cache 除外）约等于 35B dense 模型

**这就是 MoE 的核心卖点：用 35B 的"知识容量"做 3B 的"推理成本"**。

## 3.2 训练上的关键差异

| 维度 | Dense | MoE |
|---|---|---|
| 并行策略 | DP + TP + PP + CP | DP + TP + PP + CP + **EP** |
| 通信原语 | 主要 AllReduce | AllReduce **+ AllToAll** |
| 显存 | 参数 = 激活参数 = 总参 | 参数 = 总参（全 expert 都要存）|
| 计算 | 所有 token 做相同计算 | 不同 token 走不同 expert，**负载不均** |
| 训练稳定性 | 直接 | **需要 aux loss 防 expert 塌缩** |
| 训练速度（per token） | 慢（参数全用上）| 快（只用激活参数）|

### 详细展开

#### 1. Expert Parallel (EP)：MoE 独有的并行维度

把 128 个 expert 分散到多张卡上，比如 EP=8 → 每张卡持有 16 个 expert。
- **dispatch**: token 通过 AllToAll 送到目标 expert 所在的卡
- **expert 计算**: 各卡独立处理自己负责的 expert
- **combine**: AllToAll 把结果送回原 token 所在卡

**EP 一个 forward 至少 2 次 AllToAll**，这是 MoE 通信开销的主要来源。

#### 2. 负载均衡（aux loss）

如果 router 一边倒，导致某 expert 持续被 99% token 选中，另 expert 几乎空闲：
- 训练效率差：高负载 expert 成为瓶颈
- 模型质量差：参数容量没充分利用

**MoE aux loss** 强制路由分布均匀：
```python
aux_loss = E × sum(fraction_i × prob_i)    # 鼓励负载和概率都均匀分布
total_loss = task_loss + 0.01 × aux_loss
```

你脚本里有这两行：
```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
```
就是开 aux loss 的（z_loss 是另一种正则，防 router logits 数值爆炸）。

#### 3. 路由不可微 → straight-through

TopK 选择是离散操作，不可微。常用 trick：前向用 hard TopK，反向用 softmax 的梯度（straight-through estimator）。

#### 4. 数值精度更敏感

router 是个小网络（hidden → num_experts 的线性层），softmax + topk 对数值精度敏感。
- 训练时 router 通常用 fp32 算（即使其他用 bf16）
- vLLM 推理时为了速度可能 bf16 router，**和训练精度不一致 → 推理时路由可能选到不同 expert → logprob 偏差**
- 这就是 [reward解答.md](./reward解答.md) 里讲过的 "rollout vs training logprob drift" 的物理根源之一

## 3.3 推理 / Rollout 上的差异

| | Dense | MoE |
|---|---|---|
| KV cache 大小 | 按总参算 | **按总参算（不是激活参数）**，因为 attention 层是 dense |
| 显存占用 | 参数 + KV cache | 参数（35B）+ KV cache（按 35B 算）|
| 单步 forward 时间 | 跟参数量线性相关 | 跟**激活参数**线性相关（更快）|
| vLLM 实现 | 标准 | 需要专门的 MoE kernel（fused MoE）|
| EP 用不用 | n/a | **vLLM rollout 通常不用 EP，只用 TP**（吞吐优先）|

⚠️ **常见误解**：以为 Qwen3.5-35B-A3B 的 KV cache "只按 3B 算"。**错**。KV cache 是 attention 层的副产物，attention 是 dense 的，KV 按 35B 模型的 hidden / head 数算。这就是为什么 vLLM 跑 MoE 还是吃显存。

## 3.4 性能权衡总结

### MoE 的优点
- ✅ **训练 / 推理 FLOPs 低**（只用激活参数）→ 同样硬件能训更大"容量"模型
- ✅ **专家化**：不同 expert 自然分工，可能学到更细致的知识区分
- ✅ **稀疏性带来 scaling 优势**：参数翻倍但 FLOPs 不翻倍

### MoE 的代价
- ❌ **显存占用按总参数算**（35B MoE ≈ 35B Dense 的显存）
- ❌ **通信开销大**（AllToAll 比 AllReduce 难 scale）
- ❌ **训练不稳定**（router 塌缩、load imbalance）
- ❌ **推理基础设施复杂**（需要专门 MoE kernel、EP 调度）
- ❌ **微调 / RL 训练时 routing drift**：训练改了权重，rollout 时 routing 可能跟不上

### 什么时候用 MoE

- 有大显存（H100/H200）但想训更大"知识容量"的模型
- 推理需要极致 throughput 而显存够（API 服务场景）
- 任务多样化 / 多语言 / 多领域 → expert 自然分工有意义

## 3.5 你训练 Qwen3.5-35B-A3B 时的具体影响

```bash
# 你脚本的并行配置
TP=2          # 张量并行
PP=1          # 流水并行
CP=1          # 序列并行
EP=8          # ← MoE 独有
ETP=1         # expert 内部的张量并行
```

EP=8 意味着：
- 128 个 expert 分到 8 张 GPU 上，每张持有 16 个 expert
- 每个 forward 在 MoE 层做 2 次 AllToAll（dispatch + combine）
- **跨节点 EP**（48 卡 / 6 节点时）：AllToAll 跨节点流量爆炸 → 训练慢

为什么 EP=8 是个甜蜜点：
- EP=8 + 8 卡 / 节点 → **AllToAll 全在节点内 NVLink 上跑**（300GB/s）
- EP=16 → AllToAll 必须跨节点 → 走 IB（50GB/s）→ 慢 6×
- EP=4 → expert 切得粗，单 expert 显存压力大

这就是为什么 verl 官方示例几乎都用 **EP = 一个节点的 GPU 数**。

---

# 四、面试高频题模板

## Q1: AllReduce 和 AllToAll 的区别？

> AllReduce 是"每人都拿到全局聚合结果"，AllToAll 是"每人把自己持有的 N 份数据按 rank 分发出去"。通信量量级相同（O(T)），但 **AllReduce 是接力环、AllToAll 是全连接**，跨节点时 AllToAll 性能掉得多。所以 DP 用 AllReduce 能 scale 到几百卡，MoE 的 EP 用 AllToAll 通常控制在节点内不跨节点。

## Q2: 为什么 MoE 训练比 Dense 难？

四个原因：
1. **通信**：多了 AllToAll，且跨节点性能差
2. **负载均衡**：router 容易塌缩，需要 aux loss
3. **数值精度**：router 对 fp16 / bf16 敏感
4. **训练/推理 drift**：vLLM 和 Megatron 的 routing 数值可能不一致

## Q3: MoE 模型 35B-A3B 的显存按多少算？

> 参数显存按 35B 算（所有 expert 都要存）；推理 FLOPs 和单步时间按 3B 算（激活参数）；**KV cache 按 35B 的 hidden / head 数算**（attention 是 dense 的，不参与 MoE 稀疏化）。简化口诀：**MoE 省算力不省显存**。

## Q4: EP 为什么不能像 DP 一样无脑 scale？

> AllToAll 通信复杂度看起来是 O(T)，但实际 N² 条流挤同一链路，跨节点时严重退化。EP=8 + 8 卡/节点 是经典配置，让 AllToAll 全在 NVLink 上跑；EP 跨节点的话需要 IB ≥ 800Gbps 才不拖速。

## Q5: 训练时 MoE 用 EP，rollout 时为什么不用？

> Rollout 一般用 vLLM，vLLM 默认只支持 TP，不用 EP。原因是推理时 batch 通常小，token 数少，AllToAll 开销 / 计算量比训练时还差。所以 vLLM 把 expert 沿 TP 维度切分，每个 TP rank 持有所有 expert 的一部分参数。**这导致训练和推理的 routing 数值路径不一致**，是 PPO/GRPO 时 logprob drift 的来源之一。

---

# 五、必背"金句库"

- **"AllReduce 通信量与 N 无关，AllToAll 通信量量级一样但 N² 条流"** —— 解释为什么 DP scale 好、EP scale 不好
- **"MoE 省算力不省显存：35B 参数都要存，但只算 3B"** —— 一句话讲清 MoE 优劣
- **"EP 通常控制在一个节点内：NVLink 300GB/s vs IB 50GB/s 差 6 倍"** —— 工程经验
- **"MoE 训练要 aux loss 防 expert 塌缩，z_loss 防 router 数值爆炸"** —— 论文细节
- **"vLLM rollout 用 TP-only routing 和 Megatron 训练 EP routing 数值不一致 → drift"** —— 深度细节
- **"verl HybridEngine 重算 old_log_probs 用训练引擎，绕过 vLLM/Megatron drift"** —— 联动 [reward解答.md](./reward解答.md)
