# 熵坍塌解决方案（verl 框架内可用工具大全）

不只讲 KL loss，**从 entropy 公式出发**枚举所有可干预的杠杆，让你看到 verl 里 8 类工具如何从不同角度防止 entropy collapse。

---

## 一、从公式出发：熵坍塌的本质是什么

### 熵的定义

```
H(π) = -Σ_v π(v|s) · log π(v|s)
```

对 LLM 而言，在一个位置上 π 是一个 V 维（词表大小）概率分布。

### 熵坍塌的数学本质

`H(π)` 接近 0 当且仅当 **π 的概率质量集中到极少数 token 上**（极端：one-hot，H=0）。

```
分布 1: [0.5, 0.3, 0.15, 0.05, ...]  H ≈ 1.2  (健康)
分布 2: [0.95, 0.04, 0.01, ...]       H ≈ 0.25 (偏低)
分布 3: [0.999, 0.0009, ...]          H ≈ 0.01 (坍塌)
```

**RL 训练让 entropy 下降的根本机制**：
- PPO loss 鼓励"高 advantage 的 token 概率上升"
- 反复迭代后，被反复鼓励的 token 概率→1，其他→0
- 直到 rollout 采样时所有 N 条 response 都一样 → group_std=0 → 训练信号消失

所以**防 entropy collapse = 让 π 分布的概率质量"被偷走"的速度变慢，或拉回均匀**。

### 6 个干预维度

从公式看，能影响 `H(π)` 的杠杆有 6 类：

| 维度 | 干预方式 | 在 verl 里的工具 |
|---|---|---|
| **① 显式拉 entropy 大** | 直接在 loss 加 -H 项 | `entropy_coeff` |
| **② 限制 π 偏离 π_ref** | KL 约束 | `use_kl_loss`, `use_kl_in_reward` |
| **③ 减慢 π 变化速度** | 降低梯度更新强度 | `lr`, `ppo_epochs`, `mini_batch_size`, `grad_clip` |
| **④ 让低概率 token 有上升空间** | 改 clip 形状 | DAPO clip-higher（`clip_ratio_high > low`）|
| **⑤ 给 π 多样化的反馈信号** | 多采样 / 多视角 reward | `rollout.n`, reward shaping |
| **⑥ 自动过滤坍塌组** | runtime 检测 std=0 | DAPO Dynamic Sampling（`reward_manager=dapo`）|

下面逐一展开。

---

## 二、verl 框架里 8 类解决方案

### 方案 1：直接奖励 entropy（公式维度 ①）

最直接，把 entropy 作为正则项加到 loss：

```
L_total = L_policy + kl_loss_coef × KL - entropy_coeff × H
                                          ↑
                                    负号 = 鼓励 H 大
```

配置：
```bash
actor_rollout_ref.actor.entropy_coeff=0.001
```

**作用机制**：optimizer 在追求"reward 高 token 概率上升"的同时，也被"分布尽量平均"拉扯。两股力平衡到一个 entropy 平台。

**值选择**：
- 0.001~0.005（RLHF 常见）
- 0.01（DAPO 论文）
- 默认 0（你目前的设置，不防 collapse）

**优点**：直接、可控、效果最强
**缺点**：可能让模型"假装多样"，本质学习能力下降

---

### 方案 2：KL 约束（公式维度 ②）

让 π_θ 不能离 π_ref 太远，**间接限制 entropy 下降速度**（因为 ref 通常 entropy 较高）。

#### 2.1 在 loss 里加 KL（GRPO 推荐）

```bash
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl   # k3 estimator
```

公式：
```
L_total = L_policy + kl_loss_coef × KL(π_θ || π_ref) - entropy_coeff × H
                     ↑
                惩罚 KL 大
```

#### 2.2 在 reward 里扣 KL（经典 PPO）

```bash
algorithm.use_kl_in_reward=True
algorithm.kl_ctrl.kl_coef=0.01
```

公式：
```
shaped_reward = reward - β × KL(π_θ || π_ref)
advantage = (shaped_reward - group_mean) / group_std
```

KL 通过 advantage 间接影响梯度。

#### 两种方式对照

| | 方式 A：加 loss | 方式 B：扣 reward |
|---|---|---|
| 梯度路径 | KL 直接产生梯度 | 通过 advantage 缩放 |
| 信号强度 | 每条 token 都受约束 | advantage=0 时无约束 |
| GRPO 论文 | ✅ | ❌ |
| 经典 PPO | 可用 | ✅ |

**verl 两个都开会 warning**（双重收税）。

---

### 方案 3：减慢梯度更新（公式维度 ③）

不直接干预 entropy，**降低 π 变化速度**，给 entropy 缓冲时间。

#### 3.1 降低 learning rate

```bash
actor_rollout_ref.actor.optim.lr=5e-7   # 1e-6 → 5e-7
```

每次更新步长变小 → 分布变化慢 → entropy 下降斜率变小。

#### 3.2 降低 ppo_epochs（同批数据少训几遍）

```bash
actor_rollout_ref.actor.ppo_epochs=1    # 默认就是 1
```

同一批 rollout 数据训练遍数越多 → π_new 离 π_old 越远 → entropy 越容易塌。

#### 3.3 增大 mini_batch_size（梯度噪声小、更稳）

```bash
actor_rollout_ref.actor.ppo_mini_batch_size=256   # 128 → 256, 每 step 更新次数减少
```

每次 optimizer.step 用更多 sample 估计梯度，**噪声降低 → 更新更平滑 → 不会突然 collapse**。

代价：每 step 更新次数减少（rollout 数据利用率低）。

#### 3.4 加 gradient clipping

```bash
actor_rollout_ref.actor.grad_clip=1.0
```

防止 outlier sample 推动巨大梯度更新，把 π 一次性推向极端。

---

### 方案 4：改 clip 形状（公式维度 ④）—— DAPO Clip-Higher

PPO clip 公式：
```
L_policy = -E[ min(ratio × A, clip(ratio, 1-ε_low, 1+ε_high) × A) ]
```

**经典对称 clip**：ε_low = ε_high = 0.2

**DAPO Clip-Higher**：ε_high > ε_low，比如 ε_low=0.2, ε_high=0.28

#### 为什么非对称 clip 防 collapse

考虑两种 token：
- **高概率 token A**（π_old=0.7，被 reward 强化）：ratio 可能 → 1.4，**超过 1+ε=1.2 被 clip**，更新被压制 ✓
- **低概率 token B**（π_old=0.001，被 reward 强化）：ratio 可能 → 3.0，但 clip 上界也是 1+ε=1.2，被 clip 压制 ❌

**对称 clip 下，低概率 token 的"翻身机会"被压制得和高概率一样**，导致它们永远不可能从 0.001 涨到有意义的 0.05。结果：高概率 token 继续被强化，低概率 token 永远低 → entropy 缩。

**Clip-Higher 把上界放宽（ε_high=0.28 → clip 上界 1.28）**，给低概率 token 更多上升空间，**直接对抗 entropy collapse 的微观机制**。

配置：
```bash
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
```

DAPO 论文实测对 entropy 维持有效，**这是四件套里最针对 entropy collapse 的一条**。

---

### 方案 5：增加采样多样性（公式维度 ⑤）

#### 5.1 提高 rollout.n

```bash
actor_rollout_ref.rollout.n=16   # 8 → 16, 每 prompt 采更多 response
```

**作用机制**：
- N 越大，**group 内 reward 出现方差的概率越高**
- group_std=0 的 group 被 dapo 过滤的概率降低
- 模型有更多"有效训练 token"

代价：rollout 时间翻倍。

#### 5.2 不要做 sampling 截断（关键！）

```bash
# 必须保持中性值,任何收紧都会加速 entropy 下降
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1
actor_rollout_ref.rollout.repetition_penalty=1.0
```

**关键认知**：sampling filter（top_p<1 / top_k>0）会让 rollout 实际分布**比 raw policy 更尖**，等于训练前就先 collapse 了一截。**RL 训练绝对禁止 sampling filter**。

参考 [参数.md 采样流水线章节](./参数.md#rollout-采样参数完整流水线repetition_penalty--temperature--top_k--top_p)。

---

### 方案 6：改 loss 聚合方式（公式维度 ③ 细化）—— Token-Level Loss

```bash
actor_rollout_ref.actor.loss_agg_mode=token-mean   # 而不是 seq-mean
```

**对比**：
- `seq-mean`（默认）：先 token 平均、再 sequence 平均。**长 response 的 token 权重低**。
- `token-mean`（DAPO 推荐）：所有 token 等权重。**长 response 贡献大**。

**为什么影响 entropy**：长 response 通常对应模型"展开思考"的 case，token 多样性高。如果用 seq-mean，长 response 的多样性贡献被稀释 → 训练倾向"短而尖"的输出 → entropy 下降。

token-mean 让长 response 的多样性保留更多权重，**间接维持 entropy**。

---

### 方案 7：runtime 自动过滤（公式维度 ⑥）—— DAPO Dynamic Sampling

```bash
reward.reward_manager.name=dapo   # 你已经在用
```

**作用机制**：rollout 完后，如果某 prompt 的 N 条 response **reward 全 0 或全 1**（group_std=0），**整组丢弃，不参与训练**。

#### 这条为什么能防 collapse

entropy collapse 的**晚期症状**是 rollout 全一样 → group_std=0。dapo 会自动跳过，**避免坍塌的 group 继续推动 policy 朝坍塌方向更新**。

**但这是兜底，不是治本**。dapo 帮你"延缓崩盘"，真正的因还是上面 1-6 类工具。

---

### 方案 8：改训练数据 / reward（公式维度外，但相关）

#### 8.1 加 reward 多样性

如果 reward 只有 0/1（你的 blzk_rule 就是），group 内方差容易消失（success rate 接近 0 或 1 时）。

**解法**：
- 引入**格式分**（部分正确给 0.3-0.5）
- 用 **GenRM** 替代规则（连续分数，方差更稳）
- 加 **process reward**（多步打分）

#### 8.2 难度过滤数据

如果 dataset 太简单（90%+ 成功率）：
- 大部分 group 全 1，dapo 过滤掉
- 实际有效训练样本少
- entropy 下降快（成功的 response 模式被反复强化）

→ 移除太简单的样本，保留中等难度（30-70% 成功率）。

GRPO 比 PPO 对此宽容（看 [rl-algorithms.md](./rl-algorithms.md)），但极端时仍需处理。

---

## 三、工具按强度 / 风险分级

按"激进 → 保守"排序：

| 级别 | 工具 | 风险 | 推荐使用时机 |
|---|---|---|---|
| **🟢 必开（零成本）** | DAPO manager + Clip-Higher | 无 | 始终开 |
| **🟢 必开（零成本）** | rollout sampling 中性值 | 无 | 始终开 |
| **🟡 推荐**（GRPO 默认）| use_kl_loss=True (0.01) | 学习速度略降 | 训练长 / 想稳定 |
| **🟡 推荐**（GRPO 默认）| entropy_coeff=0.001 | 略影响 reward 上限 | 看到 entropy 偏低就开 |
| **🟡 推荐**| loss_agg_mode=token-mean | 无 | 长 response 任务 |
| **🟠 微调**| 降 lr (1e-6 → 5e-7) | 收敛慢 | entropy 快塌时 |
| **🟠 微调**| 增大 mini_batch_size | rollout 数据利用率降 | 训练不稳 |
| **🔴 极端**| entropy_coeff=0.01 | 学不动 | 数据极端时 |
| **🔴 极端**| 重新过滤数据 | 重跑数据 pipeline | 万不得已 |

---

## 四、推荐组合（按你 27B 的情况）

### 当前你的"激进 baseline"
```bash
# 全部关掉,只剩 PPO clip 兜底
actor.use_kl_loss=False
actor.entropy_coeff=0
algorithm.use_kl_in_reward=False
# clip 对称,无 clip-higher
# loss_agg_mode=seq-mean (默认)
```
观察到 entropy 0.37 → 0.13，快塌。

### 推荐"GRPO 标准配方"（一步到位）

```bash
# 在 ACTOR 数组里改 / 加这些:
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl       # 你已经有
actor_rollout_ref.actor.entropy_coeff=0.001           # 0 → 0.001
actor_rollout_ref.actor.clip_ratio_low=0.2            # 显式设
actor_rollout_ref.actor.clip_ratio_high=0.28          # DAPO clip-higher
actor_rollout_ref.actor.loss_agg_mode=token-mean      # DAPO token-level loss
```

总 loss：
```
L = L_policy(clip-higher)
  + 0.01 × KL(π_θ || π_ref)
  - 0.001 × H(π_θ)
```

三层防御，每层从不同角度抵抗 collapse，**几乎不会塌**。

### 渐进式（如果想分步验证）

第一次实验（最稳，看效果）：
- 只加 `use_kl_loss=True` 和 `entropy_coeff=0.001` 两条

第二次实验（开 clip-higher）：
- 加 `clip_ratio_high=0.28`

第三次实验（完整 DAPO）：
- 加 `loss_agg_mode=token-mean`

每次保留 metric，对比 reward 上升幅度 vs entropy 维持。

---

## 五、监控指标：怎么知道 entropy 在塌

### 直接信号

| 指标 | 健康 | 危险 |
|---|---|---|
| `actor/entropy` | 0.5~1.5 平台 | 单调下降到 < 0.2 |
| `actor/kl_loss`（如果开了 use_kl_loss）| 0.5~2.0 稳定 | 飙到 5+ |

### 间接信号（先于 entropy 显现）

| 指标 | 含义 | 触发阈值 |
|---|---|---|
| `actor/pg_clipfrac` | clip 触发率 | > 30% 持续 |
| `actor/ppo_kl` | π_new vs π_old | > 1e-3 持续上升 |
| `critic/advantages/std` | group 内方差 | < 0.3 持续 |
| `judge_reason: ok` 占比 | 任务正确率 | 接近 100%（success 饱和也会塌）|
| dapo filtered group 占比 | 全 0/1 group 比例 | > 30% 持续 |

**多个指标一起恶化** = 实锤 collapse，应该立即停止并加 entropy/KL 防御。

---

## 六、面试金句

### 公式视角版

> Entropy collapse 的本质是 `H(π) = -Σ p log p` 里 p 的概率质量被集中到极少数 token。从公式看，防 collapse 有 6 个干预维度：**直接拉 H 大**（entropy bonus）、**限制 π 偏移 ref**（KL）、**减慢 π 变化速度**（lr / ppo_epochs / batch）、**改 clip 形状给低概率 token 翻身机会**（DAPO clip-higher）、**多样化反馈信号**（rollout.n）、**runtime 自动过滤坍塌组**（DAPO Dynamic Sampling）。

### DAPO 视角版

> DAPO 论文的四件套实际是从不同维度防 entropy collapse：**Clip-Higher** 直接干预 clip 形状，**Dynamic Sampling** 在坍塌后兜底过滤，**Token-Level Loss** 防止长 response 的多样性被稀释，**Overlong Reward Shaping** 改 reward landscape 让长度区分度变细。**四件套都不是直接奖励 entropy**，但都是为了同一个目标——**让概率分布不要过快塌缩**。这是把 "entropy collapse" 从单一指标问题升级成"训练动力学"问题的完整工程方案。

---

## 七、与相关笔记的交叉引用

- 各 loss 项的公式：[entropy-and-loss-formulas.md](./entropy-and-loss-formulas.md)
- DAPO vs Naive RewardManager：[dapo_vs_naive_reward_manager.md](./dapo_vs_naive_reward_manager.md)
- RL 算法选型：[rl-algorithms.md](./rl-algorithms.md)
- 采样参数完整流水线：[参数.md](./参数.md) (在 "rollout 采样参数完整流水线" 章节)
- 27B 实验观察到的 collapse 现象：[exp记录/27bdense/实验结果.md](./exp记录/27bdense/实验结果.md)
