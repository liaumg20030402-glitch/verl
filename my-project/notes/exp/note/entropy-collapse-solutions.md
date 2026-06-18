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
| **⑥ 自动过滤坍塌组** | runtime 检测 std=0 | DAPO Dynamic Sampling（`algorithm.filter_groups.enable=True`）|

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
- verl 默认 `entropy_coeff: 0`（不防 collapse；DAPO 论文也基本依赖 Clip-Higher 而非显式 entropy bonus）

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

**verl 两个都开会打印 `NOTICE: You have both enabled in-reward kl and kl loss.`**（[utils/config.py:170](../../verl/utils/config.py#L170)），是提示不是 error，但等于双重收税，一般二选一。verl 默认两个都 False，开 GRPO 时通常只开 `use_kl_loss=True`。

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

**原理**：backward 后计算所有参数梯度的全局 L2 范数，若超过阈值就等比例缩小，方向不变、大小被压到阈值以内。

```
‖g‖ = √(Σ gᵢ²)            ← 所有参数梯度拼成一个向量，算范数

如果 ‖g‖ > grad_clip：
    g ← g × (grad_clip / ‖g‖)    ← 方向不变，只缩小幅度
```

**为什么能防 entropy collapse**：  
RL 训练中偶尔会遇到 outlier sample（reward 极端高/低），产生巨大梯度，一次 optimizer.step 就把 policy 推向极端，entropy 急剧下降。grad_clip 把单步更新幅度上界封死：

```
正常 step: ‖g‖ ≈ 0.3  →  正常更新，π 小幅变化
outlier:   ‖g‖ = 18.0 →  clip → ‖g‖ = 1.0 →  π 受控变化 ✓
```

**verl 默认值已经是 1.0**（在 [`workers/config/actor.py:302`](../../verl/workers/config/actor.py#L302) 的 `FSDPActorConfig.grad_clip` 字段，Megatron 也是同等默认），不设也生效，显式写出是为了可见性。值越小越保守，一般不需要改动。

---

### 方案 4：改 clip 形状（公式维度 ④）—— DAPO Clip-Higher

PPO clip 公式：
```
L_policy = -E[ min(ratio × A, clip(ratio, 1-ε_low, 1+ε_high) × A) ]
```

**经典对称 clip**：ε_low = ε_high = 0.2（**verl 默认 `clip_ratio_low=0.2, clip_ratio_high=0.2`**，即对称）

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
- group_std=0 的 group 被 `filter_groups` 丢弃的概率降低
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



---

### 方案 6：改 loss 聚合方式（公式维度 ③ 细化）—— Token-Level Loss

```bash
# verl 默认就是 token-mean,以下显式写出只是为了可见性
actor_rollout_ref.actor.loss_agg_mode=token-mean
```

**verl 的可选项**（见 [actor.py:212-217](../../verl/workers/config/actor.py#L212-L217)）：
- `token-mean`（**verl 默认**，DAPO 推荐）：把整个 mini-batch 里所有 valid token 的 loss 直接 `masked_mean`，每个 token 等权重。**长 response 贡献大**。
- `seq-mean-token-sum`：先把每条 sequence 内的 token loss 求和（不除以长度），再对 sequence 取均值。**长 response 权重最大**（梯度按长度线性放大）。
- `seq-mean-token-mean`：先 token 平均、再 sequence 平均（经典 PPO 实现的常见做法）。**长 response 的 token 权重低**。
- `seq-mean-token-sum-norm`：在 `seq-mean-token-sum` 基础上再除一个 scale_factor（默认 response_length）做归一化。


**为什么影响 entropy**：长 response 通常对应模型"展开思考"的 case，token 多样性高。如果用 `seq-mean-token-mean`，长 response 的多样性贡献被稀释 → 训练倾向"短而尖"的输出 → entropy 下降。

verl 默认的 `token-mean` 让长 response 的多样性保留更多权重，**间接维持 entropy**——这一条你不需要主动开。

---

### 方案 7：runtime 自动过滤（公式维度 ⑥）—— DAPO Dynamic Sampling

**重要勘误**：Dynamic Sampling **不是** `reward_manager=dapo` 触发的。在 verl 里这两件事是分开的：

| 功能 | 配置 | 实现位置 |
|---|---|---|
| **Overlong Reward Shaping**（长度惩罚） | `reward_model.reward_manager=dapo` + `reward_model.overlong_buffer.*` | [reward_manager/dapo.py:121-130](../../verl/workers/reward_manager/dapo.py#L121-L130) |
| **Dynamic Sampling**（过滤全 0/全 1 组） | `algorithm.filter_groups.enable=True` + `algorithm.filter_groups.metric=acc` | [config/algorithm.py:43-56](../../verl/trainer/config/algorithm.py#L43-L56) + trainer 主循环 resample |

#### 7.1 Dynamic Sampling 配置（真正过滤坍塌组的那条）

```bash
algorithm.filter_groups.enable=True
algorithm.filter_groups.metric=acc           # 也可以是 score / seq_reward 等
algorithm.filter_groups.max_num_gen_batches=10   # 重采样上限,0 = 不限
data.gen_batch_size=1536                     # 生成 batch
data.train_batch_size=512                    # 训练 batch (留 buffer 让 filter)
```

**作用机制**：rollout 完后，如果某 prompt 的 N 条 response 上 `metric` 全部相同（例如 acc 全 1 或全 0 → group_std=0），整组丢弃；如果丢弃后凑不满 `train_batch_size`，trainer 主循环会继续 rollout 直到凑够（或达到 `max_num_gen_batches` 上限）。

#### 7.2 DAPO reward manager 实际做什么 —— Overlong Reward Shaping

**配置语法（注意 `+` 前缀，因为是 hydra 里"新增字段"而非"覆盖"）**：

```bash
reward.reward_manager.name=dapo                                        # 切到 DAPORewardManager
+reward.reward_kwargs.overlong_buffer_cfg.enable=True
+reward.reward_kwargs.overlong_buffer_cfg.len=4096                     # 缓冲区长度
+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0           # 最大扣分
+reward.reward_kwargs.overlong_buffer_cfg.log=False                    # 是否在 reward_extra_info 里记
+reward.reward_kwargs.max_resp_len=16384                               # 必须 = data.max_response_length
```

**verl 里的默认值**：`overlong_buffer_cfg` **没有 yaml 默认值**，[`experimental/reward_loop/reward_manager/dapo.py:33-39`](../../verl/experimental/reward_loop/reward_manager/dapo.py#L33) 取不到就是 `None`，**等价于关闭**。所以 `reward_manager.name=dapo` 但不写 overlong_buffer 时，DAPO manager 拿到的是个"空壳"，和 naive manager 没有实质差别（只少了 num_examine 打印逻辑）。

**罚分公式**（[`reward_manager/dapo.py:108-114`](../../verl/experimental/reward_loop/reward_manager/dapo.py#L108)）：

```
expected_len    = max_resp_len - overlong_buffer.len      # 你 16384 → 12288
exceed_len      = valid_response_length - expected_len
overlong_reward = min(-exceed_len / overlong_buffer.len × penalty_factor, 0)
reward         += overlong_reward                          # ≤ 0,只扣不奖
```

**推荐取值**（按你 `data.max_response_length=16384` 的情况）：

| 字段 | 推荐值 | 含义 |
|---|---|---|
| `enable` | `True` | 关 = 不做长度惩罚 |
| `len` | `4096`（约 `max_resp_len / 4`） | 缓冲带宽。`< 12288` 不罚；`12288~16384` 线性扣 0~`penalty_factor` |
| `penalty_factor` | `1.0` | 满罚 = 扣 1 分。如果你的 reward 是 0/1，等于把超长样本压成 0；若 reward 范围更大可适当调高 |
| `log` | `False` 训练中 / `True` 调试 | True 会在 `reward_extra_info` 里加 `overlong_reward` 和 `overlong` 两列方便观察 |
| `max_resp_len` | `16384`（必须 = `data.max_response_length`） | 用来算 `expected_len` |

**长度建议的边界条件**：
- 如果 `response_length/clip_ratio`（被硬截断的样本比例）> 5%，**先开 overlong**，再谈 entropy。截断带来的 reward 噪声会污染 group_std。
- 如果开 overlong 之后模型迅速变短（`response_length/mean` 下降 30%+），说明 `penalty_factor=1.0` 太狠，降到 0.5。
- 反之如果开了基本看不到效果，可能是模型本来就很少触及 `expected_len`，那不开也行。

#### 7.2.1 你的脚本现状诊断

你 [`my-run_qwen3_5-27b-megatron.sh:30`](../scripts/my-run_qwen3_5-27b-megatron.sh#L30) 写了 `REWARD_MANAGER_NAME=dapo` 但**没有**任何 `reward.reward_kwargs.overlong_buffer_cfg.*` override。

→ 你拿到的是"DAPO 壳，零长度惩罚"，等价于 naive。如果你的训练中观察到长 response 截断比例不低（`response_length/clip_ratio` 在 tensorboard 里有这条），建议在 `REWARD` 数组里追加：

```bash
REWARD+=(
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=True       # 头几次跑建议 True 方便观察
    +reward.reward_kwargs.max_resp_len=16384
)
```

#### 这条为什么能防 collapse

entropy collapse 的**晚期症状**是 rollout 全一样 → group_std=0。`filter_groups` 自动跳过这些组，**避免坍塌的 group 继续推动 policy 朝坍塌方向更新**。

**但这是兜底，不是治本**。filter_groups 帮你"延缓崩盘"，真正的因还是上面 1-6 类工具。

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
- 大部分 group 全 1，被 `filter_groups` 过滤掉
- 实际有效训练样本少
- entropy 下降快（成功的 response 模式被反复强化）

→ 移除太简单的样本，保留中等难度（30-70% 成功率）。

GRPO 比 PPO 对此宽容（看 [rl-algorithms.md](./rl-algorithms.md)），但极端时仍需处理。

---

## 三、工具按强度 / 风险分级

按"激进 → 保守"排序：

| 级别 | 工具 | 风险 | 推荐使用时机 |
|---|---|---|---|
| **🟢 verl 已默认** | `loss_agg_mode=token-mean`、`grad_clip=1.0` | 无 | 不用动 |
| **🟢 必开（零成本）** | `algorithm.filter_groups.enable=True` + Clip-Higher | 无 | 始终开 |
| **🟢 必开（零成本）** | rollout sampling 中性值（temperature=1, top_p=1, top_k=-1） | 无 | 始终开 |
| **🟡 推荐**（GRPO 默认）| `use_kl_loss=True`（`kl_loss_coef=0.01`，verl 默认 0.001） | 学习速度略降 | 训练长 / 想稳定 |
| **🟡 推荐** | `entropy_coeff=0.001` | 略影响 reward 上限 | 看到 entropy 偏低就开 |
| **🟡 推荐** | DAPO reward manager + `overlong_buffer` | 无 | 长 response 任务（需配合长度惩罚） |
| **🟠 微调**| 降 lr (1e-6 → 5e-7) | 收敛慢 | entropy 快塌时 |
| **🟠 微调**| 增大 mini_batch_size | rollout 数据利用率降 | 训练不稳 |
| **🔴 极端**| `entropy_coeff=0.01` | 学不动 | 数据极端时 |
| **🔴 极端**| 重新过滤数据 | 重跑数据 pipeline | 万不得已 |

---

## 四、推荐组合（按你 27B 的情况）

### 当前你的"激进 baseline"
```bash
# 全部关掉,只剩 PPO clip 兜底
actor.use_kl_loss=False              # verl 默认就是 False
actor.entropy_coeff=0                # verl 默认就是 0
algorithm.use_kl_in_reward=False     # verl 默认就是 False
# clip_ratio_low=0.2 / clip_ratio_high=0.2 (verl 默认对称,无 clip-higher)
# loss_agg_mode=token-mean (verl 默认,这条 DAPO 推荐其实已经在用)
```
观察到 entropy 0.37 → 0.13，快塌。

**注意**：你 baseline 里其实只缺三件套里的 **Clip-Higher**、**Dynamic Sampling (filter_groups)**、**Overlong Reward**，token-level loss 已经是默认行为，不用补。

### 推荐"GRPO 标准配方"（一步到位）

```bash
# 在 ACTOR 数组里改 / 加这些:
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01             # verl 默认 0.001,GRPO 论文常用 0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl       # 你已经有(也是 verl 默认)
actor_rollout_ref.actor.entropy_coeff=0.001           # 0 → 0.001
actor_rollout_ref.actor.clip_ratio_low=0.2            # 显式设(等于 verl 默认)
actor_rollout_ref.actor.clip_ratio_high=0.28          # DAPO clip-higher,关键
# loss_agg_mode=token-mean 是 verl 默认,无需添加

# Dynamic Sampling (DAPO 三件套之一)
algorithm.filter_groups.enable=True
algorithm.filter_groups.metric=acc
algorithm.filter_groups.max_num_gen_batches=10

# Overlong Reward Shaping (DAPO 四件套之一,可选,长 response 任务才需要)
reward_model.reward_manager=dapo
reward_model.overlong_buffer.enable=True
reward_model.overlong_buffer.len=4096
reward_model.overlong_buffer.penalty_factor=1.0
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
- 加 `algorithm.filter_groups.enable=True` + Overlong Reward
- （`loss_agg_mode=token-mean` 是 verl 默认，没什么可加的）

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
| `filter_groups` 过滤掉的 group 占比 | 全 0/1 group 比例（开 `algorithm.filter_groups.enable=True` 才有此指标） | > 30% 持续 |

**多个指标一起恶化** = 实锤 collapse，应该立即停止并加 entropy/KL 防御。

### 健康状态的判据（不是单看 entropy）

**重要**：不要只看 `actor/entropy` 一条曲线就判断 collapse。真正危险的是"熵低 + 组内回答趋同 + advantage 信号消失"三者同时发生。如果只是 entropy 变低但验证集稳定上升、rollout 仍有组内差异，**不一定要强行拉高 entropy**。

**健康状态**：`response_length`、`actor/entropy`、`reward`、`val accuracy` **同步改善**（不是"越长越好"也不是"entropy 越高越好"）。

### Reward 拆分日志（防止 reward hacking 掩盖 collapse）

如果你的 reward 是单一标量（如 `blzk_rule` 里的 0/1 acc），熵坍塌时很难分辨是模型钻了哪个 reward 空子。建议在自定义 reward function 里返回 `dict`：

```python
return {
    "score": final_reward,
    "acc": acc,
    "format_score": format_score,
    "final_answer_found": found,
    "overlong": is_overlong,       # 配合 7.2.1 overlong_buffer_cfg.log=True
    "parse_error": parse_err,
    "repetition": rep_ratio,
}
```

verl 的 reward manager 会把 dict 里除 `score` 外的所有字段记到 `reward_extra_info`，tensorboard 上能逐条看分布。熵坍塌时通常能看到某个字段（往往是 format_score 或 repetition）异常高，那就是钻空子的方向。

---

## 五点五、按优先级的排查顺序

如果你已经看到 entropy 在快塌，按这个顺序改（**严禁一次改多个**，每次一个变量）：

1. **确认 rollout 采样是中性**：`temperature=1.0, top_p=1.0, top_k=-1, do_sample=True`。不是中性的先改这个，比加任何 entropy bonus 都有效。
2. **统计 group reward std=0 的比例**：通过 `critic/advantages/std` 间接看。比例高就开 `algorithm.filter_groups.enable=True` 或加大 `rollout.n` 到 16。
3. **看 `response_length/clip_ratio`**：> 5% 就先开 7.2 的 overlong shaping，处理截断 reward 噪声。
4. **开 Clip-Higher**：`clip_ratio_low=0.2, clip_ratio_high=0.28`。这条是 DAPO 四件套里对 entropy 维持最直接的。
5. **token-mean 是 verl 默认**：确认没人把它改成 `seq-mean-token-mean`。
6. **如果 `pg_clipfrac` 长期 > 30% 或 `ppo_kl` 飙**：降 LR、减 `ppo_epochs`、增 `mini_batch_size`，**不要继续加探索**（已经在硬撞 clip 墙了）。
7. **拆 reward 日志**：确认不是 parser、格式分、截断、重复模板在骗分。
8. **才考虑开 KL/entropy loss**：`use_kl_loss=True` + `kl_loss_coef=0.01`、`entropy_coeff=0.001`。这两条最直接但也最容易让模型"假装多样"，所以放在排查链最后。
9. **仍不稳**：考虑数据难度课程（保留 pass rate 20%~80% 的样本）、SFT rehearsal、换 GSPO 等算法。

**反模式**（不要做的事）：
- 看 entropy 一塌就立刻把 `entropy_coeff` 加到 0.01：会让模型学不动，且掩盖真实原因。
- 把 rollout `temperature` 调到 1.2+ 来"救"：模型 logits 已经尖了，温度救不回来，反而拖累训练效率。
- 同时打开 `use_kl_loss=True` 和 `algorithm.use_kl_in_reward=True`：verl 会打 NOTICE，二者效果重复且互相干扰。

---

## 六、总结

### 公式视角版

> Entropy collapse 的本质是 `H(π) = -Σ p log p` 里 p 的概率质量被集中到极少数 token。从公式看，防 collapse 有 6 个干预维度：**直接拉 H 大**（entropy bonus）、**限制 π 偏移 ref**（KL）、**减慢 π 变化速度**（lr / ppo_epochs / batch）、**改 clip 形状给低概率 token 翻身机会**（DAPO clip-higher）、**多样化反馈信号**（rollout.n）、**runtime 自动过滤坍塌组**（DAPO Dynamic Sampling）。

### DAPO 视角版

> DAPO 论文的四件套实际是从不同维度防 entropy collapse：**Clip-Higher** 直接干预 clip 形状（`clip_ratio_low/high`），**Dynamic Sampling** 在坍塌后兜底过滤（`algorithm.filter_groups`），**Token-Level Loss** 防止长 response 的多样性被稀释（`loss_agg_mode=token-mean`，**已是 verl 默认**），**Overlong Reward Shaping** 改 reward landscape 让长度区分度变细（DAPO reward manager + `overlong_buffer`）。**四件套都不是直接奖励 entropy**，但都是为了同一个目标——**让概率分布不要过快塌缩**。这是把 "entropy collapse" 从单一指标问题升级成"训练动力学"问题的完整工程方案。
>
> **verl 实现侧的关键拆分**：四件套不是一个开关搞定，而是 4 处独立配置——`clip_ratio_high`（actor）、`algorithm.filter_groups.enable`（algorithm）、`loss_agg_mode`（actor，默认已开）、`reward_model.reward_manager=dapo` + `overlong_buffer`（reward）。"reward_manager=dapo" 只负责长度惩罚，**不**负责过滤坍塌组。


