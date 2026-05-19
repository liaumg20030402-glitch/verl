# Entropy / KL / Total Loss 完整公式手册

聚焦 PPO/GRPO 训练 actor 时的 loss 构成，涵盖：
1. Entropy 是什么、怎么算
2. KL loss 怎么算（含 verl 支持的 4 种估计器）
3. Total loss 完整公式
4. entropy_coeff / kl_loss_coef 各自的作用
5. 实战调参建议

---

## 一、Entropy 的定义和直觉

### 数学定义

对于一个概率分布 P 在离散集合 X 上：

```
H(P) = -Σ_x P(x) · log P(x)
```

- P(x) ∈ [0, 1] 是 x 的概率
- log 通常取自然对数（nats 单位）

### 在 LLM 里的具体形式

LLM 每一步输出一个**词表大小的分布**（vocab_size 维），假设词表大小 V，模型在某个位置的输出概率是 `p = softmax(logits)`：

```
单位置 entropy = -Σ_{v=1..V} p(v) · log p(v)
```

一条 response 的整体 entropy 一般用**所有位置的 token-level entropy 取平均**：

```python
# 伪代码
log_probs = log_softmax(logits, dim=-1)  # shape [response_len, vocab_size]
probs = exp(log_probs)
per_token_entropy = -(probs * log_probs).sum(dim=-1)  # shape [response_len]
response_entropy = per_token_entropy.mean()           # scalar
```

verl 用的就是这个 token-level 平均（具体见 [verl/utils/torch_functional.py](../../verl/utils/torch_functional.py)）。

### 直觉值参考（LLM 场景）

| entropy 量级 | 含义 |
|---|---|
| ~ log(vocab_size) ≈ 11-12（V=151936）| 均匀分布,完全不确定（理论极限）|
| 3 ~ 5 | base model 早期训练 |
| 1.5 ~ 3 | base model / SFT 后 |
| 0.8 ~ 1.5 | RLHF 健康中期 |
| **0.3 ~ 1.0** | **RLHF 健康后期（推荐区间）**|
| < 0.3 | 偏低,接近 collapse |
| < 0.1 | **entropy collapse**,rollout 全一样 |

你 27B 训练观察到 0.13，已经在第 5 档危险区。

---

## 二、Entropy Bonus 是什么

### 数学动机

PPO 目标是**最大化奖励**：
```
maximize  E[ R(s, a) ]
```

但只追求奖励会让 policy **过快收敛到少数高概率 token**（exploitation 压倒 exploration）。所以经典 RL 引入 **entropy 正则项**，鼓励 policy 保持探索性：

```
maximize  E[ R(s, a) ] + β · H(π)
                         ↑
                  鼓励 entropy 大
```

换成最小化 loss：

```
minimize  -E[ R(s, a) ] - β · H(π)
                          ↑
                     负号 = entropy bonus
```

### entropy_coeff 是什么

就是上面公式里的 **β**，verl 配置项叫 `actor.entropy_coeff`：

```bash
actor_rollout_ref.actor.entropy_coeff=0.001  # β = 0.001
```

| 取值 | 效果 |
|---|---|
| `0` | 完全关闭 entropy bonus(你目前的设置)|
| `0.001 ~ 0.005` | 温和奖励 entropy,RLHF 常见 |
| `0.01` | DAPO 论文取值,较强 |
| `> 0.05` | 过强,模型学不动 |
| `< 0` | 反向,惩罚 entropy(罕用)|

### 为什么加 entropy bonus 能防 collapse

- entropy 越大 → loss 越小 → optimizer 倾向**保留 entropy**
- 即使 reward 强烈推荐某个 token，entropy bonus 在背后**拉回均匀化**
- 两股力量博弈，stabilize 在某个**有方差**的分布上

---

## 三、KL Loss 是什么

KL 散度衡量两个分布的差异。RL 训练里用它**约束 actor 不要离 reference model 太远**。

### KL 散度的数学定义

```
KL(P || Q) = Σ_x P(x) · log(P(x) / Q(x))
           = E_{x ~ P}[ log P(x) - log Q(x) ]
           = E_{x ~ P}[ log_ratio(x) ]
```

其中 `log_ratio = log P(x) - log Q(x)`。

注意 **KL 不对称**：`KL(P||Q) ≠ KL(Q||P)`。

### 在 RL 里的应用

通常算 **`KL(π_θ || π_ref)`**（actor 相对 reference 的 KL）：

```
KL_loss = E_{a ~ π_θ}[ log π_θ(a|s) - log π_ref(a|s) ]
```

但这个期望我们没法精确算（vocab 太大），用 **从 rollout 采样的 a** 做单点估计。这就涉及 **KL 估计器**的选择。

### verl 支持的 4 种 KL 估计器

verl 配置 `actor.kl_loss_type` 决定用哪种（[源码](../../verl/utils/kl_penalty.py)）：

```python
def kl_penalty(logprob: torch.Tensor, ref_logprob: torch.Tensor, kl_penalty: str) -> torch.Tensor:
    log_ratio = logprob - ref_logprob   # 关键中间量

    if kl_penalty == "kl":
        return log_ratio                  # k1: 直接用 log_ratio
    
    if kl_penalty == "abs":
        return log_ratio.abs()            # k2: 绝对值,等价 |log P - log Q|

    if kl_penalty == "mse":
        return 0.5 * log_ratio ** 2       # MSE: 平方
    
    if kl_penalty == "low_var_kl":        # k3 (推荐!)
        return torch.exp(log_ratio) - log_ratio - 1.0
```

#### 4 种估计器对照

| 名称 | 公式 | 是否无偏 | 方差 | 推荐 |
|---|---|---|---|---|
| **`kl`** (k1) | `log_ratio` | ✅ 无偏 | 大 | 简单但噪声大 |
| **`abs`** (k2) | `|log_ratio|` | ❌ 有偏（高估）| 中 | 工程性能稍稳 |
| **`mse`** | `0.5 × log_ratio²` | ❌ 有偏 | 中 | 不常用 |
| **`low_var_kl`** (k3) | `exp(log_ratio) - log_ratio - 1` | ✅ 无偏 | **低** | **John Schulman 推荐**,verl 默认 |

#### k3 (low_var_kl) 为什么是最优选择

[John Schulman's blog](http://joschu.net/blog/kl-approx.html) 证明：

```
k3 = exp(log_ratio) - log_ratio - 1
   = π_θ/π_ref - log(π_θ/π_ref) - 1
```

数学上有几个好性质：
1. **无偏**：`E[k3] = KL(π_θ || π_ref)` 严格成立
2. **始终 ≥ 0**：`exp(x) - x - 1 ≥ 0` 对所有 x，所以是非负的（一致性）
3. **方差小**：实测比 k1 低一个数量级

**verl 默认用 low_var_kl 就是 k3**，你脚本里设的：
```bash
actor_rollout_ref.actor.kl_loss_type=low_var_kl   # ← k3
```

是正确选择。

### KL loss 在 verl 里的完整计算

```python
# verl/workers/actor/megatron_actor.py:562-570
if self.config.use_kl_loss:
    ref_log_prob = data["ref_log_prob"]       # ref 模型对 rollout 的 logprob
    kld = kl_penalty(
        logprob=log_prob,                      # 当前 actor 对同一段 token 的 logprob
        ref_logprob=ref_log_prob,
        kl_penalty=self.config.kl_loss_type    # 默认 low_var_kl
    )
    # kld shape: [batch, response_len]
    
    kl_loss = agg_loss(
        loss_mat=kld,
        loss_mask=response_mask,                # 只算 response 部分,prompt 不算
        loss_agg_mode=self.config.loss_agg_mode # 默认 token-mean
    )
    # kl_loss: scalar
    
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

注意：
- KL 只在 response token 上算（prompt 不算，因为 prompt 是用户给的，model 没贡献）
- 聚合方式（mean / sum）由 `loss_agg_mode` 控制

### kl_loss_coef 是什么

公式里的**系数**，控制 KL 约束强度：

```bash
actor_rollout_ref.actor.kl_loss_coef=0.01   # 你脚本里的值
```

| 取值 | 效果 |
|---|---|
| `0.001` | 极弱约束,几乎自由 |
| `0.01`（默认）| GRPO 论文标准值 |
| `0.05` | 较强,policy 改动小 |
| `0.1+` | 过强,模型学不动 |

---

## 四、Total Loss 完整公式

PPO/GRPO actor 的完整 loss：

```
L_total = L_policy + kl_loss_coef × L_kl - entropy_coeff × H
```

### 详细展开

```
L_policy = -E[ min(ratio × A, clip(ratio, 1-ε, 1+ε) × A) ]   # PPO clip,核心
                                                                  
其中 ratio = π_θ(a|s) / π_θ_old(a|s)
      A = advantage = (r - group_mean) / group_std   ← GRPO

L_kl    = mean_over_response_tokens[ kl_penalty(log π_θ, log π_ref) ]
        ≈ KL(π_θ || π_ref)

H       = mean_over_response_tokens[ -Σ_v p(v) log p(v) ]
        = 当前 policy 在 response 各位置的平均 entropy
```

### 符号方向

注意各项的**符号**：

| 项 | 系数 | 符号 | 优化方向 |
|---|---|---|---|
| `L_policy` | 1 | 自带负号 | 最大化 ratio × A（让 advantage 高的行为更可能）|
| `L_kl` | `+ kl_loss_coef` | 正 | **惩罚** KL,让 actor 不离 ref 太远 |
| `H` | `- entropy_coeff` | 负 | **奖励** entropy,鼓励探索 |

总目标：**最大化 reward**（通过 -L_policy）+ **最小化 KL**（贴近 ref）+ **最大化 entropy**（保留多样性）。三者博弈。

### Python 代码片段（对应 verl 实现）

```python
# 1. PPO policy gradient loss (带 clip)
log_ratio = log_prob_new - log_prob_old
ratio = torch.exp(log_ratio)
pg_loss1 = -ratio * advantages
pg_loss2 = -torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages
pg_loss = torch.max(pg_loss1, pg_loss2).mean()    # ← max 不是 min,因为已经取负

# 2. KL loss (k3)
log_ratio_ref = log_prob_new - log_prob_ref
kld = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
kl_loss = kld.mean()    # 假设 mask 已应用

# 3. Entropy
probs = torch.exp(log_probs)
entropy = -(probs * log_probs).sum(dim=-1).mean()

# 4. Total loss
total_loss = pg_loss + kl_loss_coef * kl_loss - entropy_coeff * entropy

# 5. Backward
total_loss.backward()
```

---

## 五、KL 加的位置：loss 里 vs reward 里

verl 支持**两种 KL 约束位置**，互斥：

### 方式 A：`use_kl_loss=True`（loss 里加，GRPO 默认）

如上面完整公式，**KL 直接加进 loss**。verl 默认走这条路径。

```bash
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
algorithm.use_kl_in_reward=False   # ← 这条关掉
```

### 方式 B：`use_kl_in_reward=True`（reward 里扣，经典 RLHF）

```
shaped_reward = reward - β × KL(π_θ || π_ref)
advantage = compute_advantage(shaped_reward)
policy_loss = -log_prob × advantage    # 不另外加 KL 项
```

KL 通过 advantage 间接影响梯度。

```bash
actor_rollout_ref.actor.use_kl_loss=False
algorithm.use_kl_in_reward=True
algorithm.kl_ctrl.kl_coef=0.01
```

### 两者对比

| | A: loss 里加（推荐 GRPO）| B: reward 里扣（经典 PPO）|
|---|---|---|
| 梯度路径 | KL 直接产生梯度 | 通过 advantage 缩放 |
| 信号强度 | 每条 token 都受约束 | advantage=0 的 token 无约束 |
| 调参 | `kl_loss_coef` 直接对应 | `kl_coef` 进 reward,量级耦合 |
| GRPO 论文 | ✅ | ❌ |

**verl 两个都开会打 warning**（双重收税）。

---

## 六、实战：你 27B Dense 的 loss 该怎么配

### 当前你的配置（激进版）

```bash
actor_rollout_ref.actor.use_kl_loss=False    # KL 关
actor_rollout_ref.actor.kl_loss_coef=0.01    # 配了值但没用上(因为 use_kl_loss=False)
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.entropy_coeff=0      # entropy bonus 关
algorithm.use_kl_in_reward=False             # reward 里也没 KL
```

**等价于**：
```
L_total = L_policy   # 只有 PPO clip,啥都不约束
```

结果：entropy 快速下降（你截图看到的 0.37 → 0.13），有 collapse 风险。

### 推荐改为（GRPO 论文默认）

```bash
actor_rollout_ref.actor.use_kl_loss=True       # ← 改成 True
actor_rollout_ref.actor.kl_loss_coef=0.01      # 保持
actor_rollout_ref.actor.kl_loss_type=low_var_kl # 保持(k3 estimator)
actor_rollout_ref.actor.entropy_coeff=0.001    # ← 加点 entropy bonus
algorithm.use_kl_in_reward=False               # 保持
```

**等价于**：
```
L_total = L_policy + 0.01 × KL(π_θ || π_ref) - 0.001 × H(π_θ)
```

### 训练时盯的指标

| 指标 | 健康值 | 异常含义 |
|---|---|---|
| `actor/pg_loss` | 在 0 附近震荡（小）| NaN → 训崩 |
| `actor/kl_loss` | 0 → 0.5~2.0 缓慢上升 | 飙到 10+ → KL 失控,policy 跑飞 |
| `actor/entropy` | 缓慢下降到 0.3~1.0 平台 | 跌破 0.1 → collapse |
| `actor/pg_clipfrac` | 10~30% | > 50% → clip 太频繁,clip_high 不够 |

---

## 七、总结

### 公式版

> PPO/GRPO 的 actor total loss 是三项组合：`L_total = L_policy + kl_loss_coef × KL - entropy_coeff × H`。其中 **L_policy 是 PPO clip loss**（最大化 ratio × advantage 同时被 clip 限制策略变化幅度），**KL 项约束 actor 不离 reference model 太远**（防 reward hacking 和 entropy collapse），**entropy 项鼓励分布保持平**（防探索能力丢失）。

### 估计器选择

> verl 默认用 **k3 estimator（`low_var_kl`）**算 KL：`exp(log_ratio) - log_ratio - 1`。和朴素的 `log_ratio` 估计相比，它**无偏 + 始终非负 + 方差低一个量级**，是 [John Schulman 推荐](http://joschu.net/blog/kl-approx.html)的做法。

### 系数调优

> `entropy_coeff` 和 `kl_loss_coef` 分别从两个不同维度防 entropy collapse：**KL 约束是相对的（不能离 ref 太远）**，**entropy bonus 是绝对的（必须保持分布平）**。前者像安全带，后者像油门反向调节。GRPO 论文都开（典型值 kl=0.01, entropy=0.001），DAPO 也是。**两个都关掉的激进配置**容易在长 epoch 训练里 collapse。
