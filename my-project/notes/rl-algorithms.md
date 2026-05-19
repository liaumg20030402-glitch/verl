# RL 算法对比与训练数据难度要求

涵盖 PPO / GRPO / DPO / DAPO，重点解释**为什么 PPO 需要难度过滤而 GRPO 不需要**。

## 直接回答你上级的话

> "PPO 要进行 XXX 采样" —— 大概率是指 **PPO 需要 critic 网络估 V(s)，V(s) 学好的前提是 reward 有方差**。如果数据要么太简单（reward 都是 1）要么太难（reward 都是 0），V(s) 学到的就是常数，**advantage 永远等于 0，模型学不到东西**。

> "GRPO 不用难度过滤" —— GRPO **没有 critic**，advantage 用同组（同一 prompt 的 N 条 response）的 reward 均值/方差归一化算的。即使全数据集**平均**很难/很简单，只要**同组内**有方差，advantage 就有信号。**而且 DAPO/GRPO 的"Dynamic Sampling"会在运行时自动过滤掉 group_std=0 的样本**，不用预过滤。

---

## 一、四种主流算法核心区别

| 算法 | 全称 | 是否用 critic | 是否需要 rollout | 训练数据要求 |
|---|---|---|---|---|
| **PPO** | Proximal Policy Optimization | ✓（V 网络）| ✓ | **难度均衡**（success rate 30-70%）|
| **GRPO** | Group Relative Policy Optimization | ✗ | ✓（rollout.n 条/prompt）| 宽松，组内有方差即可 |
| **DPO** | Direct Preference Optimization | ✗ | ✗（离线训练）| 需要 (chosen, rejected) 偏好对 |
| **DAPO** | Decoupled Clip + Dynamic Sampling PPO | ✗（GRPO 改进版）| ✓ | 宽松，运行时动态过滤 |

---

## 二、PPO：经典 RLHF 的基础

### 核心公式

PPO 优化：
```
L_PPO = -E[ min(ratio × A, clip(ratio, 1-ε, 1+ε) × A) ]

其中：
  ratio = π_new(a|s) / π_old(a|s)
  A = advantage(s, a) = Q(s, a) - V(s)
                     ≈ r(s, a) - V(s)   (单步)
```

**关键**：A 用一个**额外的 critic 网络 V(s)** 来估计。

### Critic 网络 V(s) 的作用

V(s) 学习"给定 prompt s 后的预期 reward"。训练目标：
```
L_critic = MSE( V(s),  实际拿到的 reward )
```

每个 prompt 训练时反复见，V(s) 慢慢收敛到这个 prompt 上的平均 reward。

### 为什么 PPO 需要难度过滤

设想三种数据集：

**情况 A：太简单（success rate 95%）**
```
prompt 1: 5 次采样,5 次 reward=1
        V(prompt 1) ≈ 1
        advantage = 1 - 1 = 0   ❌ 没梯度
```

**情况 B：太难（success rate 5%）**
```
prompt 2: 20 次采样,1 次 reward=1,其他 19 次 0
        V(prompt 2) ≈ 0.05
        99% 的样本: advantage = 0 - 0.05 = -0.05  (微弱负梯度)
        5% 的样本:  advantage = 1 - 0.05 = +0.95  (强正梯度)
        → 信号极度稀疏,1/20 概率才出现可学样本
```

**情况 C：难度适中（success rate 30-70%）**
```
prompt 3: 8 次采样,3 次 1,5 次 0
        V(prompt 3) ≈ 0.375
        正样本 advantage = +0.625  ✓
        负样本 advantage = -0.375  ✓
        → 信号清晰,正负都有学习方向
```

**所以 PPO 训练前的"难度过滤"通常做这两件事**：
1. **去掉太简单的样本**（模型已经全对，没学习空间）
2. **去掉太难的样本**（模型从来不对，正梯度信号丢失）
3. **保留中间难度**（success rate ~30-70%）

实操：先用 base model rollout 估算每条数据的 success rate，按 rate 筛掉两端。

### 为什么这叫"XXX 采样"

可能你上级说的是几种叫法之一：
- **"重要性采样"（importance sampling）**：PPO 用 ratio 修正 π_old vs π_new，但和难度过滤本身没直接关系
- **"拒绝采样"（rejection sampling）**：训练前用 success rate 筛掉极端样本，是难度过滤的别名
- **"难度采样"**：直接描述行为
- **"on-policy 采样"**：泛指 PPO 用当前 policy 采样，和难度无关

最贴切的应该是 **"拒绝采样筛掉难度极端的样本"**，是 RLHF PPO 工程经验。

---

## 三、GRPO：DeepSeek-Math 的 critic-free 方案

### 核心公式

GRPO 干脆**不要 critic**：
```
每个 prompt 采 N 条 response（rollout.n）
计算每条的 reward: r_1, r_2, ..., r_N

group_mean = mean(r_1, ..., r_N)
group_std  = std(r_1, ..., r_N)

A_i = (r_i - group_mean) / group_std        ← 组内归一化
```

然后用和 PPO 一样的 clip loss：
```
L_GRPO = -E[ min(ratio × A_i, clip(ratio, 1-ε, 1+ε) × A_i) ]
```

只是 A_i 来自组内归一化，不是 V(s) 估计。

### 为什么 GRPO 对数据要求更宽松

**关键洞察**：advantage 是**组内相对值**，不依赖**全数据集**的难度分布。

**情况 A 的样本（PPO 会丢的"太简单"）**：
```
prompt 1: 8 次采样,8 次 reward=1
group_mean = 1, group_std = 0
A_i = (1 - 1) / 0  = NaN  ⚠️ 除零

→ verl 会过滤这种 group(看下面 DAPO 的 Dynamic Sampling)
```

**情况 B 的样本（PPO 会丢的"太难"）**：
```
prompt 2: 8 次采样,8 次 reward=0
group_mean = 0, group_std = 0
A_i = (0 - 0) / 0 = NaN  ⚠️ 同样除零

→ 过滤掉
```

**情况 C 的样本（中等难度）**：
```
prompt 3: 8 次采样,3 次 1,5 次 0
group_mean = 0.375, group_std ≈ 0.484
正样本: A_i = (1 - 0.375) / 0.484 = +1.29
负样本: A_i = (0 - 0.375) / 0.484 = -0.77

→ 自动归一化为标准化的 advantage ✓
```

### 关键区别

| | PPO（用 V(s)） | GRPO（用 group）|
|---|---|---|
| advantage 来源 | 全数据集学到的 V(s) 估计值 | **同 prompt 8 条 response 的均值/方差** |
| 对单个 prompt 难度敏感 | ✓ 极敏感 | ✗ 不敏感（看组内即可）|
| 对全数据集分布敏感 | ✓ V(s) 要靠数据学 | ✗ 不需要 |
| 极端难度处理 | **预过滤** | **运行时过滤**（std=0 → 跳过）|

**所以你上级的判断是对的**：换 GRPO 后**可以不用预过滤难度**，让 GRPO 在 rollout 阶段自动暴露 group_std=0 的样本，DAPO manager 过滤掉，剩下的都是有信号的。

### 代价：N 倍的 rollout 成本

GRPO 每个 prompt 要采 N 条（你用 8），rollout 成本是 PPO（每 prompt 1 条）的 **8 倍**。所以 GRPO 适合：
- rollout 成本可接受（中小模型 / 短序列）
- 想避免 critic 训练的复杂度
- reward 可以直接计算（不需要 V(s) 帮做信用分配）

PPO 仍然有用的场景：
- rollout 极贵（大模型 / 长序列）
- 长 horizon 任务需要 V(s) 做信用分配（多步规划）
- 工程上 N 倍 rollout 撑不住

---

## 四、DPO：完全跳过 RL，直接做偏好优化

### 核心思想

DPO（Direct Preference Optimization, Stanford 2023）**不需要 rollout、不需要 reward model**，只需要**偏好对** `(prompt, chosen, rejected)`：

```
L_DPO = -E[ log σ( β × ( log π(chosen) - log π(rejected) ) 
                 - β × ( log π_ref(chosen) - log π_ref(rejected) ) ) ]
```

直觉：**push up chosen 的概率，push down rejected 的概率**，同时不能离 ref model 太远（β 控制）。

### 数据要求

- 必须有**预先标注的偏好对**：人工标注、或者别的模型生成的
- 没法在线探索（offline 算法）

### 优劣对比

| | DPO | PPO/GRPO |
|---|---|---|
| 数据 | (prompt, chosen, rejected) 偏好对 | (prompt, ground_truth) 或 reward function |
| Rollout | ❌ 不需要 | ✅ 必须 |
| Reward model | ❌ 不需要 | rule 或 RM 都行 |
| 计算量 | **小**（一次 forward 算两条 logprob）| **大**（rollout + 训练）|
| 模型能力 | 受限于偏好数据质量 | 能探索新行为 |
| 收敛稳定性 | 易 reward hacking、过拟合 | clip 保护 |
| 适用 | 对齐 / IFT 后期 | math / code / 可验证任务 |

### DPO 为什么不适合"可验证 reward"的任务

DPO 隐式假设：**偏好关系 = 模型该学的方向**。但对 math/code 这种**有正确答案**的任务：
- "5+3 = 8" 是正确，"5+3 = 7" 是错误
- DPO 需要把这两个写成偏好对喂给它
- 数据制作慢、覆盖不全

GRPO/PPO 用 reward function 直接打分，**模型自己探索 N 种答案，挑对的鼓励**，更灵活。

---

## 五、DAPO：GRPO 的工业级改进（你正在用）

字节 Seed 团队 2025 年提出。**DAPO 不是新的优化算法**，而是 GRPO + 四个工程 trick：

### 四件套

#### ① Clip-Higher（非对称 clip）

经典 PPO clip 上下对称：`clip(ratio, 1-ε, 1+ε)`。

DAPO 改成：`clip(ratio, 1-ε_low, 1+ε_high)`，且 `ε_high > ε_low`。

**直觉**：低概率 token（ratio 容易>1）的探索更松，高概率 token（ratio 容易 < 1）的收紧更严。**让低概率"好动作"有机会被强化**。

#### ② Dynamic Sampling

**在线过滤掉 reward 全 0 或全 1 的 group**：
```python
if group_std == 0:
    skip this group, 不参与训练
```

这就是 GRPO 的"自动难度过滤"机制，由 DAPO manager 实现。

**实际效果**：dataset 里残留的极端难度样本不会污染训练，**模型只在"有信号"的 group 上学习**。所以你上级说的"GRPO 不用预过滤"就是因为 DAPO 这个 trick 帮你过滤了。

#### ③ Token-Level Loss

PPO/GRPO 默认按 sequence 平均 loss：
```
loss = mean_over_responses( mean_over_tokens(per_token_loss) )
```

短 response 和长 response 的 per-sample 权重一样，**长 response 的 token 被稀释**。

DAPO 改成 token-level：
```
loss = mean_over_all_tokens(per_token_loss)
```

长 response 因为 token 多，**对总 loss 贡献更大**，避免"短 response 偷懒"。

#### ④ Overlong Reward Shaping

response 接近 `max_response_length` 时，按距离施加渐进负 reward，**避免硬截断的 0/1 跳变**。

详见 [dapo_vs_naive_reward_manager.md](./dapo_vs_naive_reward_manager.md)。

### DAPO 的角色定位

```
GRPO（DeepSeek-Math, 2024）
    + Clip-Higher
    + Dynamic Sampling
    + Token-Level Loss
    + Overlong Reward Shaping
= DAPO（字节, 2025）
```

**所以 DAPO 是 GRPO 的工业增强版**。你脚本里 `REWARD_MANAGER_NAME=dapo` 就在用这套。

---

## 六、为什么你用 GRPO/DAPO 不需要难度过滤

整合上面的讨论：

1. **没有 critic**：advantage 不依赖 V(s) 学到的全数据集分布。
2. **组内归一化**：每个 prompt 独立计算 advantage，**全数据集难度分布不影响单 prompt 的训练信号**。
3. **DAPO Dynamic Sampling 在线过滤**：rollout 完后发现某 prompt 8 条 response 全对/全错（group_std=0），**直接跳过这条**，不参与训练。
4. **运行时 vs 预处理**：PPO 必须在数据准备阶段做难度过滤（critic 训练需要稳定数据），GRPO/DAPO 把这事推到运行时自动做。

**实操建议**：
- 用 DAPO/GRPO 时，**不需要预过滤难度**
- 但需要监控 `judge_reason` 分布：如果 `key_mode_not_matched` / `json_not_found` 占比一直高（>50%），说明数据**结构性偏难**，模型从来格式都不对，这时考虑：
  - 先做轻量 SFT 让模型先学会格式
  - 再切换 GRPO 训对答内容

## 七、算法选型决策树

```
你的任务有可验证的标准答案吗？
├─ 是
│   └─ rollout 成本能接受吗？
│       ├─ 是 → GRPO/DAPO （首选）
│       └─ 否（大模型 + 长序列）→ PPO with critic
│
└─ 否（开放式对齐 / 偏好任务）
    └─ 有偏好对数据吗？
        ├─ 是 → DPO（最快）
        └─ 否 → 训 reward model + PPO/GRPO（最复杂）
```

## 八、面试金句

### 难度过滤问题

> PPO 需要 critic 网络 V(s) 来估 advantage，**V(s) 学习的前提是 reward 有方差**。如果数据全简单（reward 都是 1）或全难（reward 都是 0），V(s) 收敛到常数，advantage = 0，模型学不到东西。所以 PPO **预处理时需要按 success rate 过滤数据，保留中等难度（30-70%）**。
>
> GRPO **没有 critic**，advantage = (r_i - group_mean) / group_std 是**组内相对值**，不依赖全数据集分布。即使整个 dataset 偏难，只要同 prompt 的 8 条 response 内部有方差，advantage 就有信号。**DAPO 的 Dynamic Sampling 进一步在运行时过滤掉 group_std=0 的样本**，从根源上免除了预过滤的必要。
>
> 这是 **GRPO 相对 PPO 在工程上的核心优势**：把"难度过滤"从数据预处理推到算法本身。

### 算法演化

> RLHF 算法演化主线：**PPO → DPO → GRPO → DAPO**。
> - PPO：经典但需要 critic + 数据预处理
> - DPO：跳过 RL，离线训练偏好对，简单但不能探索
> - GRPO：去掉 critic，用组内归一化代替 V(s)，rollout 成本翻 N 倍但训练简单
> - DAPO：GRPO + 四个 trick（clip-higher、dynamic sampling、token-level loss、overlong shaping），工业级稳定版
>
> **DeepSeek-R1、字节 Seed、Qwen2.5-Math 这些 SOTA 都在用 GRPO/DAPO 系**，PPO 在 RL post-training 里逐渐成为历史选项。

### 数据难度的本质

> 不论 PPO 还是 GRPO，**学习信号最终来自 reward 的方差**。PPO 让 V(s) 学方差（容易塌缩到 0），GRPO 直接用组内方差（不会塌缩，全 0/全 1 时自动跳过）。**所以 GRPO 在"data-centric"工程上的优势是把方差需求从"数据集层面"降到"prompt-group 层面"**。

---

## 九、verl 里这些算法的开关

```bash
# GRPO（你当前用的）
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01

# 切到 PPO（需要 critic 配置)
algorithm.adv_estimator=gae        # generalized advantage estimation
# + 加 critic 相关参数(critic.model.path 等)

# DAPO 的四件套
reward.reward_manager.name=dapo                          # ④ Overlong Shaping + ② Dynamic Sampling
actor_rollout_ref.actor.clip_ratio_low=0.2               # ① Clip-Higher 下界
actor_rollout_ref.actor.clip_ratio_high=0.28             # ① Clip-Higher 上界
actor_rollout_ref.actor.loss_agg_mode=token-mean         # ③ Token-Level Loss
```

verl 默认 GRPO + dapo manager，**你脚本现在就是 DAPO**，只是没显式调四件套，可以加上面后两行让 clip-higher 和 token-level loss 也生效。
